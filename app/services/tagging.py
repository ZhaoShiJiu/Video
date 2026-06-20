"""
AI Image Tagging Service

Core logic for:
- Image hash computation (MD5)
- Sidecar JSON tag file management
- Single/batch image tagging via Qwen3-VL-Flash
- Tag-based material search & statistics
"""
import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional

from loguru import logger

from app.config import config
from app.models.const import FILE_TYPE_IMAGES
from app.models.schema import ImageTags
from app.services import llm
from app.services import state as sm

def compute_image_hash(image_path: str) -> str:
    """
    Compute MD5 hash of an image file for change detection.

    Uses chunked reading for large files to avoid memory issues.
    Returns a 32-character hex string.
    """
    hash_md5 = hashlib.md5()
    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def _get_tags_dir() -> str:
    """Get the unified writable directory for all sidecar tag files."""
    from app.utils import utils
    tags_dir = os.path.join(utils.storage_dir("tags", create=True))
    return tags_dir


def _normalize_path_for_tags(image_path: str) -> str:
    """
    Convert an absolute image path to a safe relative path under storage/tags/.

    Uses the absolute path with separators replaced so tags from different
    material directories never collide.
    """
    abs_path = os.path.abspath(image_path)
    # Replace OS separators and colons (Windows) to create safe filename
    safe = abs_path.replace(":", "_").replace("\\", "/")
    # Remove leading slash
    safe = safe.lstrip("/")
    return safe


def get_sidecar_path(image_path: str, base_dir: str = "") -> str:
    """
    Get the sidecar tags file path.

    All tags are stored under storage/tags/, keyed by the absolute image path.
    This eliminates the dual-write (side-by-side vs fallback) inconsistency
    that caused partial-tag-display bugs across sessions.
    """
    tags_dir = _get_tags_dir()
    safe_name = _normalize_path_for_tags(image_path)
    tags_path = os.path.join(tags_dir, safe_name + ".tags.json")

    # Ensure parent directory exists
    tags_parent = os.path.dirname(tags_path)
    if not os.path.isdir(tags_parent):
        os.makedirs(tags_parent, exist_ok=True)

    return tags_path


def load_tags(image_path: str, base_dir: str = "") -> Optional[ImageTags]:
    """Load existing tags from sidecar file under storage/tags/. Returns None if missing or corrupted."""
    sidecar = get_sidecar_path(image_path, base_dir)

    if not os.path.isfile(sidecar):
        return None
    try:
        with open(sidecar, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ImageTags(**data)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Corrupted tags file {sidecar}: {e}")
        return None


def save_tags(image_path: str, tags: ImageTags, base_dir: str = "") -> None:
    """Write tags to sidecar JSON file under storage/tags/."""
    sidecar = get_sidecar_path(image_path, base_dir)
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(tags.model_dump(), f, ensure_ascii=False, indent=2)


def delete_tags(image_path: str, base_dir: str = "") -> None:
    """Delete sidecar tags file from storage/tags/."""
    path = get_sidecar_path(image_path, base_dir)
    if os.path.isfile(path):
        os.remove(path)


# Track whether old side-by-side tags have been migrated to storage/tags/
_migration_done = False


def migrate_old_tags(base_dir: str) -> int:
    """
    One-time migration: move all old side-by-side .tags.json files
    (from before unified storage) into storage/tags/.

    Returns the number of tags migrated.
    """
    global _migration_done
    if _migration_done:
        return 0

    images = _find_all_images(base_dir)
    migrated = 0

    for img_path in images:
        old_sidecar = img_path + ".tags.json"
        if not os.path.isfile(old_sidecar):
            continue

        # Already have tags in the unified location? Skip (unified wins)
        unified = get_sidecar_path(img_path, base_dir)
        if os.path.isfile(unified):
            try:
                os.remove(old_sidecar)
            except OSError:
                pass
            continue

        # Move old sidecar to unified location
        try:
            unified_parent = os.path.dirname(unified)
            if not os.path.isdir(unified_parent):
                os.makedirs(unified_parent, exist_ok=True)
            shutil.move(old_sidecar, unified)
            migrated += 1
        except OSError as e:
            logger.warning(f"Failed to migrate {old_sidecar}: {e}")

    _migration_done = True
    if migrated > 0:
        logger.info(f"Migrated {migrated} old side-by-side tag files to storage/tags/")
    return migrated


def _find_all_images(base_dir: str) -> List[str]:
    """Scan base_dir (and one level of subdirectories) for image files."""
    allowed_ext = FILE_TYPE_IMAGES  # ["jpg", "jpeg", "png", "bmp"]
    images = []

    if not os.path.isdir(base_dir):
        return images

    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith(tuple(f".{ext}" for ext in allowed_ext)):
                images.append(os.path.join(root, f))
        # Only scan one level deep for external material directories
        # to match existing material scanning conventions

    return images


def find_images_needing_tags(
    base_dir: str,
    force: bool = False,
) -> List[str]:
    """
    Scan material directory and return absolute paths of images needing tags.

    Logic:
    - force=True: return all images (regardless of existing tags)
    - force=False:
      · No sidecar file → needs tagging
      · Sidecar exists but file_hash mismatches → needs tagging
      · Sidecar exists and hash matches → skip
    """
    images = _find_all_images(base_dir)

    if force:
        return images

    needing = []
    for img_path in images:
        existing = load_tags(img_path, base_dir)
        if existing is None:
            needing.append(img_path)
        else:
            try:
                current_hash = compute_image_hash(img_path)
                if existing.file_hash != current_hash:
                    needing.append(img_path)
            except OSError as e:
                logger.warning(f"Cannot hash {img_path}: {e}")
                needing.append(img_path)

    return needing


def tag_single_image(
    image_path: str,
    base_dir: str,
) -> ImageTags:
    """
    Tag a single image via AI.

    Flow:
    1. Compute file hash
    2. Call llm.analyze_image() for tags
    3. Build ImageTags model (file_path relative to base_dir)
    4. Write sidecar file
    5. Return ImageTags
    """
    file_hash = compute_image_hash(image_path)
    rel_path = os.path.relpath(image_path, base_dir)

    result = llm.analyze_image(image_path)

    tags = ImageTags(
        file_path=rel_path,
        file_hash=file_hash,
        characters=result.get("characters", []),
        emotions=result.get("emotions", ["平静"]),
        events=result.get("events", ["无明显事件"]),
        description=result.get("description", ""),
        colors=result.get("colors", []),
        model=config.tagging.get("vision_model", "qwen3-vl-flash"),
        created_at=datetime.now().isoformat(),
    )

    save_tags(image_path, tags, base_dir)
    return tags


def _update_progress(
    task_id: str,
    completed: int,
    total: int,
    current_file: str = "",
    tagged: int = 0,
    skipped: int = 0,
    failed: int = 0,
):
    """Update the task progress in the state store."""
    if total > 0:
        progress = int(completed / total * 95) + 5
    else:
        progress = 100
    progress = min(100, max(0, progress))

    sm.state.update_task(
        task_id,
        state=-1,  # Will be set by caller; keep existing state
        progress=progress,
        total=total,
        tagged=tagged,
        skipped=skipped,
        failed=failed,
        current_file=current_file,
    )


def batch_tag_images(
    base_dir: str,
    force: bool = False,
    max_concurrent: int = 3,
    task_id: Optional[str] = None,
) -> dict:
    """
    Batch-tag images in the material directory.

    Returns: {"total": N, "tagged": N, "skipped": N, "failed": N, "errors": [...]}

    - Uses ThreadPoolExecutor for concurrency control
    - Each image is independently tagged; one failure doesn't affect others
    - Progress is reported via task_id to the state system
    - API calls are spaced by request_interval seconds
    """
    from app.models import const

    images = find_images_needing_tags(base_dir, force=force)
    total = len(images)
    tagged = 0
    skipped = 0
    failed = 0
    errors = []
    completed = 0

    request_interval = config.tagging.get("request_interval", 0.3)

    if task_id:
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_PROCESSING,
            progress=5,
            total=total,
            tagged=0,
            skipped=0,
            failed=0,
            current_file="",
            errors=[],
        )

    if total == 0:
        if task_id:
            sm.state.update_task(
                task_id,
                state=const.TASK_STATE_COMPLETE,
                progress=100,
                total=0,
                tagged=0,
                skipped=0,
                failed=0,
                current_file="",
                errors=[],
            )
        return {"total": 0, "tagged": 0, "skipped": 0, "failed": 0, "errors": []}

    _last_request_time = 0.0

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {
            executor.submit(tag_single_image, img, base_dir): img
            for img in images
        }
        for future in as_completed(futures):
            img = futures[future]
            completed += 1
            try:
                future.result()
                tagged += 1
            except Exception as e:
                failed += 1
                errors.append({"file": img, "error": str(e)})
                logger.error(f"Failed to tag {img}: {e}")

            # Update progress
            current_rel = os.path.relpath(img, base_dir)
            if task_id:
                sm.state.update_task(
                    task_id,
                    state=const.TASK_STATE_PROCESSING,
                    progress=int(completed / total * 90) + 5 if total > 0 else 100,
                    total=total,
                    tagged=tagged,
                    skipped=skipped,
                    failed=failed,
                    current_file=current_rel,
                    errors=errors,
                )

            # Rate limit spacing
            time.sleep(request_interval)

    if task_id:
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            total=total,
            tagged=tagged,
            skipped=skipped,
            failed=failed,
            current_file="",
            errors=errors,
        )

    logger.info(
        f"Batch tagging complete: total={total}, tagged={tagged}, "
        f"skipped={skipped}, failed={failed}"
    )
    return {
        "total": total,
        "tagged": tagged,
        "skipped": skipped,
        "failed": failed,
        "errors": errors,
    }


def _load_all_tags(base_dir: str) -> List[ImageTags]:
    """Load all tags from a material directory."""
    tags_list = []
    images = _find_all_images(base_dir)
    for img_path in images:
        t = load_tags(img_path, base_dir)
        if t is not None:
            tags_list.append(t)
    return tags_list


def search_materials_by_tags(
    base_dir: str,
    characters: Optional[List[str]] = None,
    emotions: Optional[List[str]] = None,
    events: Optional[List[str]] = None,
    keyword: Optional[str] = None,
    match_mode: str = "any",
) -> List[dict]:
    """
    Search for matching image materials by tags.

    Dimensions:
    - characters: exact match from closed enumeration
    - emotions: exact match from closed enumeration
    - events: exact match from closed enumeration
    - keyword: fuzzy match in description

    match_mode:
    - "any": any dimension hit counts
    - "all": all specified dimensions must have at least one hit

    Returns list of matched materials sorted by match_score (descending).

    Scoring:
    - Each matched character: +1
    - Each matched emotion: +1
    - Each matched event: +1
    - Keyword hit in description: +2
    """
    results = []
    all_tags = _load_all_tags(base_dir)

    has_filters = bool(characters or emotions or events or keyword)

    for tags in all_tags:
        score = 0
        match_detail = {
            "characters_matched": [],
            "emotions_matched": [],
            "events_matched": [],
            "keyword_matched": False,
        }

        # Character matching
        if characters:
            matched = [c for c in characters if c in tags.characters]
            score += len(matched)
            match_detail["characters_matched"] = matched
        else:
            match_detail["characters_matched"] = []

        # Emotion matching
        if emotions:
            matched = [e for e in emotions if e in tags.emotions]
            score += len(matched)
            match_detail["emotions_matched"] = matched
        else:
            match_detail["emotions_matched"] = []

        # Event matching
        if events:
            matched = [ev for ev in events if ev in tags.events]
            score += len(matched)
            match_detail["events_matched"] = matched
        else:
            match_detail["events_matched"] = []

        # Keyword search in description
        if keyword:
            kw = keyword.strip()
            if kw and kw.lower() in tags.description.lower():
                score += 2
                match_detail["keyword_matched"] = True

        # "all" mode: skip if any specified dimension has zero hits
        if match_mode == "all":
            if characters and not match_detail["characters_matched"]:
                continue
            if emotions and not match_detail["emotions_matched"]:
                continue
            if events and not match_detail["events_matched"]:
                continue
            if keyword and not match_detail["keyword_matched"]:
                continue

        if not has_filters or score > 0:
            results.append({
                "file_path": tags.file_path,
                "characters": tags.characters,
                "emotions": tags.emotions,
                "events": tags.events,
                "description": tags.description,
                "colors": tags.colors,
                "match_score": score,
                "match_detail": match_detail,
            })

    # Sort by score descending
    results.sort(key=lambda r: r["match_score"], reverse=True)
    return results


def get_tag_statistics(base_dir: str) -> dict:
    """
    Get tag statistics for the material library.

    Returns:
    {
        "total_images": N,
        "tagged_count": N,
        "untagged_count": N,
        "character_distribution": {...},
        "emotion_distribution": {...},
        "event_distribution": {...},
        "color_distribution": {...},
        "avg_tags_per_image": N,
    }
    """
    # One-time migration: move old side-by-side .tags.json into storage/tags/
    migrate_old_tags(base_dir)

    all_tags = _load_all_tags(base_dir)
    all_images = _find_all_images(base_dir)

    total = len(all_images)
    tagged = len(all_tags)

    char_dist: Dict[str, int] = {}
    emo_dist: Dict[str, int] = {}
    evt_dist: Dict[str, int] = {}
    col_dist: Dict[str, int] = {}

    for tags in all_tags:
        for c in tags.characters:
            char_dist[c] = char_dist.get(c, 0) + 1
        for e in tags.emotions:
            emo_dist[e] = emo_dist.get(e, 0) + 1
        for ev in tags.events:
            evt_dist[ev] = evt_dist.get(ev, 0) + 1
        for co in tags.colors:
            col_dist[co] = col_dist.get(co, 0) + 1

    # Sort by frequency descending
    char_dist = dict(sorted(char_dist.items(), key=lambda x: x[1], reverse=True))
    emo_dist = dict(sorted(emo_dist.items(), key=lambda x: x[1], reverse=True))
    evt_dist = dict(sorted(evt_dist.items(), key=lambda x: x[1], reverse=True))
    col_dist = dict(sorted(col_dist.items(), key=lambda x: x[1], reverse=True))

    total_tags = sum(
        len(t.characters) + len(t.emotions) + len(t.events) for t in all_tags
    )

    return {
        "total_images": total,
        "tagged_count": tagged,
        "untagged_count": total - tagged,
        "character_distribution": char_dist,
        "emotion_distribution": emo_dist,
        "event_distribution": evt_dist,
        "color_distribution": col_dist,
        "avg_tags_per_image": round(total_tags / tagged, 1) if tagged > 0 else 0,
    }


def match_materials_by_tags(
    script_text: str,
    tagged_dir: str,
    top_k: int = 5,
) -> List[str]:
    """
    Given a video script, auto-match the most relevant tagged images.

    Workflow:
    1. Call LLM to analyze the script: what characters, emotions, events are relevant?
    2. Call search_materials_by_tags() to find matching materials
    3. Return top_k material paths sorted by match_score descending

    Falls back to empty list if no tagged images match.
    """
    if not script_text or not os.path.isdir(tagged_dir):
        return []

    # Check if there are any tagged images
    all_tags = _load_all_tags(tagged_dir)
    if not all_tags:
        logger.info("No tagged images found, skipping tag-based material matching")
        return []

    # Use LLM to extract tag-like queries from script
    try:
        script_analysis_prompt = f"""你是一名《蜡笔小新》动画视频脚本分析助手。

给定一段视频脚本，请分析脚本中涉及的角色、情绪和剧情事件，并以 JSON 格式返回。

角色只能从以下列表中选择（空数组表示不限）：
{', '.join(llm._CHARACTER_CANDIDATES.keys())}

情绪只能从以下列表中选择（空数组表示不限）：
{', '.join(llm._EMOTION_CANDIDATES)}

事件只能从以下列表中选择：
{', '.join(e for evts in llm._EVENT_CATEGORIES.values() for e in evts)}

脚本内容：
{script_text[:1000]}

请返回严格的 JSON 格式（不要 Markdown 代码块）：
{{"characters": [], "emotions": [], "events": [], "keyword": ""}}
keyword 是从脚本中提取的 1-3 个关键场景描述词，用空格分隔。"""
    except Exception as e:
        logger.warning(f"Failed to build script analysis prompt: {e}")
        return []

    prompt = script_analysis_prompt
    response_text = ""
    for attempt in range(3):
        try:
            response_text = llm._generate_response(prompt=prompt)
            if not response_text or "Error: " in response_text:
                continue
            parsed = llm._parse_tags_json(response_text)
            # Extract search params
            characters = parsed.get("characters", [])
            emotions = parsed.get("emotions", [])
            events = parsed.get("events", [])
            keyword = parsed.get("keyword", "")

            if not (characters or emotions or events or keyword):
                return []

            results = search_materials_by_tags(
                base_dir=tagged_dir,
                characters=characters if characters else None,
                emotions=emotions if emotions else None,
                events=events if events else None,
                keyword=keyword if keyword else None,
                match_mode="any",
            )

            # Return top_k file paths
            top_results = results[:top_k]
            paths = [r["file_path"] for r in top_results]
            logger.info(
                f"Tag-based material matching: script analysis → "
                f"chars={characters}, emotions={emotions}, events={events}, "
                f"keyword={keyword!r} → {len(paths)} matches"
            )
            return paths
        except Exception as e:
            logger.warning(f"Tag-based matching attempt {attempt + 1} failed: {e}")
            import time as _time

            _time.sleep(0.5)

    logger.warning("Tag-based material matching failed after all attempts")
    return []
