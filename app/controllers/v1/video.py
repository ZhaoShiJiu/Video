import glob
import os
import pathlib
import shutil
from typing import Optional, Union

from fastapi import BackgroundTasks, Depends, Path, Query, Request, UploadFile
from fastapi.params import File
from fastapi.responses import FileResponse, StreamingResponse
from loguru import logger

from app.config import config
from app.controllers import base
from app.controllers.manager.base_manager import TaskQueueFullError
from app.controllers.manager.memory_manager import InMemoryTaskManager
from app.controllers.manager.redis_manager import RedisTaskManager
from app.controllers.v1.base import new_router
from app.models.exception import HttpException
from app.models.schema import (
    AudioRequest,
    BgmRetrieveResponse,
    BgmUploadResponse,
    SubtitleRequest,
    TaskDeletionResponse,
    TaskQueryRequest,
    TaskQueryResponse,
    TaskResponse,
    TaskVideoRequest,
    VideoMaterialUploadResponse,
    VideoMaterialRetrieveResponse
)
from app.services import state as sm
from app.services import task as tm
from app.utils import file_security, utils

# 认证依赖项
# router = new_router(dependencies=[Depends(base.verify_token)])
router = new_router()

_enable_redis = config.app.get("enable_redis", False)
_redis_host = config.app.get("redis_host", "localhost")
_redis_port = config.app.get("redis_port", 6379)
_redis_db = config.app.get("redis_db", 0)
_redis_password = config.app.get("redis_password", None)
_max_concurrent_tasks = config.app.get("max_concurrent_tasks", 5)
_max_queued_tasks = config.app.get("max_queued_tasks", 100)

redis_url = f"redis://:{_redis_password}@{_redis_host}:{_redis_port}/{_redis_db}"
# 根据配置选择合适的任务管理器
if _enable_redis:
    task_manager = RedisTaskManager(
        max_concurrent_tasks=_max_concurrent_tasks,
        redis_url=redis_url,
        max_queued_tasks=_max_queued_tasks,
    )
else:
    task_manager = InMemoryTaskManager(
        max_concurrent_tasks=_max_concurrent_tasks,
        max_queued_tasks=_max_queued_tasks,
    )


def _sanitize_upload_filename(filename: str, request_id: str) -> str:
    # 浏览器或客户端有时会附带目录信息，甚至可能夹带 ../ 这类穿越片段。
    # 这里只保留纯文件名，避免上传接口把文件写到目标目录之外。
    normalized_name = (filename or "").replace("\\", "/").split("/")[-1].strip()
    if not normalized_name or normalized_name in {".", ".."}:
        raise HttpException(
            task_id=request_id,
            status_code=400,
            message=f"{request_id}: invalid filename",
        )
    return normalized_name


def _resolve_path_within_directory(base_dir: str, unsafe_path: str, request_id: str) -> str:
    try:
        return file_security.resolve_path_within_directory(base_dir, unsafe_path)
    except ValueError as exc:
        logger.warning(
            f"reject unsafe file path, request_id: {request_id}, path: {unsafe_path}, "
            f"error: {str(exc)}"
        )
        raise HttpException(
            task_id=request_id,
            status_code=404 if str(exc) == "file does not exist" else 403,
            message=f"{request_id}: invalid file path",
        )

def _task_file_to_uri(file: str, endpoint: str, task_dir: str, request_id: str) -> str:
    if not isinstance(file, str):
        return file

    if file.startswith(("http://", "https://")):
        return file

    try:
        resolved_path = file_security.resolve_path_within_directory(task_dir, file)
    except ValueError as exc:
        # 任务状态理论上只应保存任务目录内的产物路径。这里不再继续拼接 URL，
        # 避免把异常路径包装成可访问链接；同时保留原值，便于排查历史脏数据。
        logger.warning(
            f"skip unsafe task output path, request_id: {request_id}, path: {file}, "
            f"error: {str(exc)}"
        )
        return file

    relative_path = os.path.relpath(resolved_path, task_dir).replace("\\", "/")
    uri_path = f"tasks/{relative_path}"
    if endpoint:
        return f"{endpoint.rstrip('/')}/{uri_path}"
    return f"/{uri_path}"


@router.post("/videos", response_model=TaskResponse, summary="Generate a short video")
def create_video(
    background_tasks: BackgroundTasks, request: Request, body: TaskVideoRequest
):
    return create_task(request, body, stop_at="video")


@router.post("/subtitle", response_model=TaskResponse, summary="Generate subtitle only")
def create_subtitle(
    background_tasks: BackgroundTasks, request: Request, body: SubtitleRequest
):
    return create_task(request, body, stop_at="subtitle")


@router.post("/audio", response_model=TaskResponse, summary="Generate audio only")
def create_audio(
    background_tasks: BackgroundTasks, request: Request, body: AudioRequest
):
    return create_task(request, body, stop_at="audio")


def create_task(
    request: Request,
    body: Union[TaskVideoRequest, SubtitleRequest, AudioRequest],
    stop_at: str,
):
    task_id = utils.get_uuid()
    request_id = base.get_task_id(request)
    try:
        task = {
            "task_id": task_id,
            "request_id": request_id,
            "params": body.model_dump(),
        }
        sm.state.update_task(task_id)
        task_manager.add_task(tm.start, task_id=task_id, params=body, stop_at=stop_at)
        logger.success(f"Task created: {utils.to_json(task)}")
        return utils.get_response(200, task)
    except TaskQueueFullError as e:
        sm.state.delete_task(task_id)
        logger.warning(
            f"reject task because queue is full, request_id: {request_id}, task_id: {task_id}"
        )
        raise HttpException(
            task_id=task_id, status_code=429, message=f"{request_id}: {str(e)}"
        )
    except ValueError as e:
        raise HttpException(
            task_id=task_id, status_code=400, message=f"{request_id}: {str(e)}"
        )

@router.get("/tasks", response_model=TaskQueryResponse, summary="Get all tasks")
def get_all_tasks(request: Request, page: int = Query(1, ge=1), page_size: int = Query(10, ge=1)):
    tasks, total = sm.state.get_all_tasks(page, page_size)

    response = {
        "tasks": tasks,
        "total": total,
        "page": page,
        "page_size": page_size,
    }
    return utils.get_response(200, response)



@router.get(
    "/tasks/{task_id}", response_model=TaskQueryResponse, summary="Query task status"
)
def get_task(
    request: Request,
    task_id: str = Path(..., description="Task ID"),
    query: TaskQueryRequest = Depends(),
):
    request_id = base.get_task_id(request)
    endpoint = config.app.get("endpoint", "").rstrip("/")
    task = sm.state.get_task(task_id)
    if task:
        task_dir = utils.task_dir()
        response_task = dict(task)

        if "videos" in task:
            response_task["videos"] = [
                _task_file_to_uri(v, endpoint, task_dir, request_id)
                for v in task["videos"]
            ]
        if "combined_videos" in task:
            response_task["combined_videos"] = [
                _task_file_to_uri(v, endpoint, task_dir, request_id)
                for v in task["combined_videos"]
            ]
        return utils.get_response(200, response_task)

    raise HttpException(
        task_id=task_id, status_code=404, message=f"{request_id}: task not found"
    )


@router.delete(
    "/tasks/{task_id}",
    response_model=TaskDeletionResponse,
    summary="Delete a generated short video task",
)
def delete_video(request: Request, task_id: str = Path(..., description="Task ID")):
    request_id = base.get_task_id(request)
    task = sm.state.get_task(task_id)
    if task:
        tasks_dir = utils.task_dir()
        current_task_dir = os.path.join(tasks_dir, task_id)
        if os.path.exists(current_task_dir):
            shutil.rmtree(current_task_dir)

        sm.state.delete_task(task_id)
        logger.success(f"video deleted: {utils.to_json(task)}")
        return utils.get_response(200)

    raise HttpException(
        task_id=task_id, status_code=404, message=f"{request_id}: task not found"
    )


@router.get(
    "/musics", response_model=BgmRetrieveResponse, summary="Retrieve local BGM files"
)
def get_bgm_list(request: Request):
    suffix = "*.mp3"
    song_dir = utils.song_dir()
    files = glob.glob(os.path.join(song_dir, suffix))
    bgm_list = []
    for file in files:
        filename = os.path.basename(file)
        bgm_list.append(
            {
                "name": filename,
                "size": os.path.getsize(file),
                # 只返回文件名，避免把服务器绝对路径暴露给调用方。
                # 服务端后续会把该文件名解析回 songs 白名单目录。
                "file": filename,
            }
        )
    response = {"files": bgm_list}
    return utils.get_response(200, response)


@router.post(
    "/musics",
    response_model=BgmUploadResponse,
    summary="Upload the BGM file to the songs directory",
)
def upload_bgm_file(request: Request, file: UploadFile = File(...)):
    request_id = base.get_task_id(request)
    safe_filename = _sanitize_upload_filename(file.filename, request_id)
    # check file ext
    if safe_filename.lower().endswith("mp3"):
        song_dir = utils.song_dir()
        save_path = os.path.join(song_dir, safe_filename)
        # save file
        with open(save_path, "wb+") as buffer:
            # If the file already exists, it will be overwritten
            file.file.seek(0)
            buffer.write(file.file.read())
        response = {"file": safe_filename}
        return utils.get_response(200, response)

    raise HttpException(
        "", status_code=400, message=f"{request_id}: Only *.mp3 files can be uploaded"
    )

@router.get(
    "/video_materials", response_model=VideoMaterialRetrieveResponse, summary="Retrieve local video materials"
)
def get_video_materials_list(request: Request):
    allowed_suffixes = ("mp4", "mov", "avi", "flv", "mkv", "jpg", "jpeg", "png")
    # 优先使用配置的外部素材目录，未配置则回退到项目默认目录
    _material_dir = config.app.get("material_directory", "").strip()
    local_videos_dir = _material_dir or utils.storage_dir("local_videos", create=True)
    files = []
    for suffix in allowed_suffixes:
        files.extend(glob.glob(os.path.join(local_videos_dir, f"*.{suffix}")))
        # 外部素材目录通常按类别分放在子目录中，需要递归扫描一级子目录
        if _material_dir:
            files.extend(glob.glob(os.path.join(local_videos_dir, f"*/*.{suffix}")))
    # 文件系统枚举顺序不稳定，直接返回会导致"顺序拼接"在不同机器或不同
    # 时刻表现不一致。这里统一按文件名排序，至少保证服务端返回顺序可预测。
    files.sort(key=lambda file_path: os.path.basename(file_path).lower())
    video_materials_list = []
    for file in files:
        filename = os.path.basename(file)
        # 外部素材目录下，返回相对于根目录的路径（如 "001-PNG/xxx.png"），
        # 以便 preprocess_video 能正确定位子目录中的文件。
        # 默认目录下保持原有行为（只返回文件名），向后兼容。
        if _material_dir:
            try:
                display_path = os.path.relpath(file, local_videos_dir)
            except ValueError:
                display_path = filename
        else:
            display_path = filename
        video_materials_list.append(
            {
                "name": filename,
                "size": os.path.getsize(file),
                # 与 BGM 一样，只返回文件名；创建任务时再在 local_videos
                # 白名单目录内解析，避免 API 泄露宿主机绝对路径。
                "file": display_path,
            }
        )
    response = {"files": video_materials_list}
    return utils.get_response(200, response)


@router.post(
    "/video_materials",
    response_model=VideoMaterialUploadResponse,
    summary="Upload the video material file to the local videos directory",
)
def upload_video_material_file(request: Request, file: UploadFile = File(...)):
    request_id = base.get_task_id(request)
    safe_filename = _sanitize_upload_filename(file.filename, request_id)
    # check file ext
    allowed_suffixes = ("mp4", "mov", "avi", "flv", "mkv", "jpg", "jpeg", "png")
    normalized_filename = safe_filename.lower()
    # 统一按小写扩展名校验，兼容 .MOV 这类大写后缀文件。
    if normalized_filename.endswith(allowed_suffixes):
        # 优先上传到配置的外部素材目录，未配置则回退到项目默认目录
        _material_dir = config.app.get("material_directory", "").strip()
        local_videos_dir = _material_dir or utils.storage_dir("local_videos", create=True)
        save_path = os.path.join(local_videos_dir, safe_filename)
        # save file
        with open(save_path, "wb+") as buffer:
            # If the file already exists, it will be overwritten
            file.file.seek(0)
            buffer.write(file.file.read())
        response = {"file": safe_filename}
        return utils.get_response(200, response)

    raise HttpException(
        "", status_code=400, message=f"{request_id}: Only files with extensions {', '.join(allowed_suffixes)} can be uploaded"
    )

@router.get("/stream/{file_path:path}")
async def stream_video(request: Request, file_path: str):
    request_id = base.get_task_id(request)
    tasks_dir = utils.task_dir()
    video_path = _resolve_path_within_directory(tasks_dir, file_path, request_id)
    range_header = request.headers.get("Range")
    video_size = os.path.getsize(video_path)
    start, end = 0, video_size - 1

    length = video_size
    if range_header:
        range_ = range_header.split("bytes=")[1]
        start, end = [int(part) if part else None for part in range_.split("-")]
        if start is None:
            start = video_size - end
            end = video_size - 1
        if end is None:
            end = video_size - 1
        length = end - start + 1

    def file_iterator(file_path, offset=0, bytes_to_read=None):
        with open(file_path, "rb") as f:
            f.seek(offset, os.SEEK_SET)
            remaining = bytes_to_read or video_size
            while remaining > 0:
                bytes_to_read = min(4096, remaining)
                data = f.read(bytes_to_read)
                if not data:
                    break
                remaining -= len(data)
                yield data

    response = StreamingResponse(
        file_iterator(video_path, start, length), media_type="video/mp4"
    )
    response.headers["Content-Range"] = f"bytes {start}-{end}/{video_size}"
    response.headers["Accept-Ranges"] = "bytes"
    response.headers["Content-Length"] = str(length)
    response.status_code = 206  # Partial Content

    return response


@router.get("/download/{file_path:path}")
async def download_video(request: Request, file_path: str):
    """
    download video
    :param request: Request request
    :param file_path: video file path, eg: /cd1727ed-3473-42a2-a7da-4faafafec72b/final-1.mp4
    :return: video file
    """
    request_id = base.get_task_id(request)
    tasks_dir = utils.task_dir()
    video_path = _resolve_path_within_directory(tasks_dir, file_path, request_id)
    file_path = pathlib.Path(video_path)
    filename = file_path.stem
    extension = file_path.suffix
    headers = {"Content-Disposition": f"attachment; filename={filename}{extension}"}
    return FileResponse(
        path=video_path,
        headers=headers,
        filename=f"{filename}{extension}",
        media_type=f"video/{extension[1:]}",
    )


# =============================================================================
# AI Image Tagging API
# =============================================================================

from app.models.schema import (
    ImageTags,
    TagDeleteRequest,
    TagGenerateRequest,
)
from app.services import tagging


def _get_material_dir() -> str:
    """Get the configured material directory for tagging operations."""
    _material_dir = config.app.get("material_directory", "").strip()
    if _material_dir:
        base_dir = _material_dir
    else:
        base_dir = utils.storage_dir("local_videos", create=True)
    return base_dir


def _normalize_material_path(base_dir: str, file_path: str) -> str:
    """Resolve a file path relative to the material directory."""
    if os.path.isabs(file_path):
        fp = file_path
    else:
        fp = os.path.join(base_dir, file_path)
    fp = os.path.normpath(fp)
    if not os.path.isfile(fp):
        raise HttpException(
            task_id="",
            status_code=404,
            message=f"File not found: {file_path}",
        )
    return fp


@router.get(
    "/materials/tags",
    summary="Get tag statistics overview",
)
def get_tag_statistics(request: Request):
    base_dir = _get_material_dir()
    try:
        stats = tagging.get_tag_statistics(base_dir)
        return utils.get_response(200, stats)
    except Exception as e:
        logger.error(f"Failed to get tag statistics: {e}")
        raise HttpException(
            task_id="",
            status_code=500,
            message=f"Failed to get tag statistics: {str(e)}",
        )


@router.post(
    "/materials/tags/generate",
    summary="Trigger batch image tagging",
)
def trigger_batch_tagging(
    background_tasks: BackgroundTasks,
    request: Request,
    body: TagGenerateRequest,
):
    task_id = utils.get_uuid()
    base_dir = _get_material_dir()

    logger.info(
        f"Starting batch tagging: task_id={task_id}, force={body.force}, "
        f"max_concurrent={body.max_concurrent}"
    )

    # Check if tagging is enabled
    if not config.tagging.get("enabled", True):
        raise HttpException(
            task_id=task_id,
            status_code=400,
            message="Image tagging is disabled. Set [tagging] enabled = true in config.toml",
        )

    # Check if API key is configured
    api_key = config.tagging.get("vision_api_key_override", "").strip()
    if not api_key:
        api_key = config.app.get("qwen_api_key", "")
    if not api_key:
        raise HttpException(
            task_id=task_id,
            status_code=400,
            message="No API key configured. Set qwen_api_key in [app] or vision_api_key_override in [tagging].",
        )

    # Initialize task state
    from app.models import const

    sm.state.update_task(
        task_id,
        state=const.TASK_STATE_PROCESSING,
        progress=0,
        total=0,
        tagged=0,
        skipped=0,
        failed=0,
        current_file="",
        errors=[],
    )

    # Run batch tagging in background
    background_tasks.add_task(
        tagging.batch_tag_images,
        base_dir=base_dir,
        force=body.force,
        max_concurrent=body.max_concurrent,
        task_id=task_id,
    )

    return utils.get_response(
        200,
        {
            "task_id": task_id,
        },
        "Tagging task started",
    )


@router.get(
    "/materials/tags/status",
    summary="Query batch tagging progress",
)
def get_tagging_status(
    request: Request,
    task_id: str = Query(..., description="Task ID from /materials/tags/generate"),
):
    task = sm.state.get_task(task_id)
    if task:
        return utils.get_response(200, task)

    raise HttpException(
        task_id=task_id,
        status_code=404,
        message="Task not found",
    )


@router.get(
    "/materials/tags/search",
    summary="Search materials by tags",
)
def search_materials(
    request: Request,
    characters: Optional[str] = Query(None, description="Comma-separated character names"),
    emotions: Optional[str] = Query(None, description="Comma-separated emotion names"),
    events: Optional[str] = Query(None, description="Comma-separated event names"),
    keyword: Optional[str] = Query(None, description="Fuzzy search in description"),
    match: str = Query("any", description="Match mode: 'any' or 'all'"),
    limit: int = Query(20, ge=1, le=100, description="Max results to return"),
):
    base_dir = _get_material_dir()

    # Parse comma-separated parameters
    char_list = [c.strip() for c in characters.split(",") if c.strip()] if characters else None
    emo_list = [e.strip() for e in emotions.split(",") if e.strip()] if emotions else None
    evt_list = [e.strip() for e in events.split(",") if e.strip()] if events else None
    kw = keyword.strip() if keyword else None

    if match not in ("any", "all"):
        raise HttpException(
            task_id="",
            status_code=400,
            message="match must be 'any' or 'all'",
        )

    try:
        results = tagging.search_materials_by_tags(
            base_dir=base_dir,
            characters=char_list,
            emotions=emo_list,
            events=evt_list,
            keyword=kw,
            match_mode=match,
        )
        return utils.get_response(
            200,
            {
                "results": results[:limit],
                "total": len(results),
            },
        )
    except Exception as e:
        logger.error(f"Failed to search materials by tags: {e}")
        raise HttpException(
            task_id="",
            status_code=500,
            message=f"Search failed: {str(e)}",
        )


@router.delete(
    "/materials/tags",
    summary="Delete tags for specified images",
)
def delete_material_tags(
    request: Request,
    body: TagDeleteRequest,
):
    base_dir = _get_material_dir()
    deleted = 0
    not_found = 0

    for fp in body.file_paths:
        try:
            full_path = _normalize_material_path(base_dir, fp)
            sidecar = tagging.get_sidecar_path(full_path, base_dir)
            alt_path = full_path + ".tags.json"
            found = os.path.isfile(sidecar) or os.path.isfile(alt_path)
            if found:
                tagging.delete_tags(full_path, base_dir)
                deleted += 1
            else:
                not_found += 1
        except HttpException:
            not_found += 1
        except HttpException:
            not_found += 1

    return utils.get_response(
        200,
        {
            "deleted": deleted,
            "not_found": not_found,
        },
    )


@router.get(
    "/materials/tags/{file_path:path}",
    summary="Get tags for a single image",
)
def get_single_image_tags(
    request: Request,
    file_path: str = Path(..., description="Image file path relative to material directory"),
):
    base_dir = _get_material_dir()
    full_path = _normalize_material_path(base_dir, file_path)

    tags = tagging.load_tags(full_path, base_dir)
    if tags is None:
        raise HttpException(
            task_id="",
            status_code=404,
            message=f"No tags found for: {file_path}",
        )

    return utils.get_response(200, tags.model_dump())
