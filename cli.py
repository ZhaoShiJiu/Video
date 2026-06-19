import argparse
import json
import os
from typing import Sequence

from loguru import logger

from app.models.schema import MaterialInfo, VideoParams
from app.services import task as tm
from app.utils import utils


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"video-count must be >= 1, got {parsed}")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MoneyPrinterTurbo command line video generation"
    )
    parser.add_argument("--video-subject", required=True, help="video subject")
    parser.add_argument("--video-script", default="", help="custom script")
    parser.add_argument("--video-terms", default=None, help="comma-separated terms")
    parser.add_argument(
        "--video-source",
        default="pexels",
        choices=["pexels", "pixabay", "coverr", "local"],
        help="video material source",
    )
    parser.add_argument(
        "--video-materials",
        default="",
        help="comma-separated local material paths",
    )
    parser.add_argument(
        "--stop-at",
        default="video",
        choices=["script", "terms", "audio", "subtitle", "materials", "video"],
        help="pipeline stop stage",
    )
    parser.add_argument(
        "--video-count", type=_positive_int, default=1, help="output video count (>=1)"
    )
    parser.add_argument("--video-aspect", default="9:16", help="video aspect ratio")
    parser.add_argument("--voice-name", default="", help="tts voice name")
    parser.add_argument(
        "--subtitle-enabled",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="enable subtitles (default: enabled, use --no-subtitle-enabled to disable)",
    )
    parser.add_argument("--task-id", default="", help="custom task id")
    return parser.parse_args(argv)


def build_video_params(args: argparse.Namespace) -> VideoParams:
    video_terms = args.video_terms
    if video_terms:
        video_terms = [term.strip() for term in video_terms.split(",") if term.strip()]

    video_materials = None
    materials_arg = args.video_materials or ""
    if materials_arg.strip():
        video_materials = [
            # Actual duration will be detected during video processing; use 0 as placeholder.
            MaterialInfo(provider="local", url=item.strip(), duration=0)
            for item in materials_arg.split(",")
            if item.strip()
        ]

    return VideoParams(
        video_subject=args.video_subject,
        video_script=args.video_script,
        video_terms=video_terms,
        video_source=args.video_source,
        video_materials=video_materials,
        video_count=args.video_count,
        video_aspect=args.video_aspect,
        voice_name=args.voice_name,
        subtitle_enabled=args.subtitle_enabled,
    )


def run_cli(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    params = build_video_params(args)
    task_id = args.task_id or utils.get_uuid()
    logger.info(f"start cli task: {task_id}, stop_at: {args.stop_at}")
    result = tm.start(task_id=task_id, params=params, stop_at=args.stop_at)
    if not result:
        logger.error("video generation failed")
        return 1

    print(json.dumps({"task_id": task_id, "result": result}, ensure_ascii=False))
    return 0


def run_tag_images(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for 'tag-images' subcommand."""
    parser = argparse.ArgumentParser(
        description="AI image tagging for local material library"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force re-tag all images (ignore existing tags)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Max concurrent API calls (default: from config.toml)",
    )
    parser.add_argument(
        "--directory",
        default=None,
        help="Material directory (default: from config.toml)",
    )
    args = parser.parse_args(argv)

    from app.config import config as _cfg
    from app.services import tagging

    base_dir = args.directory or _cfg.app.get("material_directory", "").strip()
    if not base_dir:
        base_dir = utils.storage_dir("local_videos", create=True)
    if not os.path.isdir(base_dir):
        logger.error(f"Material directory not found: {base_dir}")
        return 1

    max_concurrent = args.max_concurrent or _cfg.tagging.get("max_concurrent", 3)

    logger.info(f"Starting tag-images: directory={base_dir}, force={args.force}, max_concurrent={max_concurrent}")

    # First, show what needs tagging
    images = tagging.find_images_needing_tags(base_dir, force=args.force)
    logger.info(f"Images needing tags: {len(images)}")

    if not images:
        logger.info("All images are already tagged. Use --force to re-tag.")
        return 0

    # Run batch tagging (synchronous in CLI)
    result = tagging.batch_tag_images(
        base_dir=base_dir,
        force=args.force,
        max_concurrent=max_concurrent,
        task_id=None,
    )

    logger.success(
        f"Tagging complete: total={result['total']}, tagged={result['tagged']}, "
        f"skipped={result['skipped']}, failed={result['failed']}"
    )

    if result["errors"]:
        logger.warning(f"Errors ({len(result['errors'])}):")
        for err in result["errors"]:
            logger.warning(f"  - {err['file']}: {err['error']}")

    return 0 if result["failed"] == 0 else 1


if __name__ == "__main__":
    import sys

    # Detect subcommand
    if len(sys.argv) > 1 and sys.argv[1] == "tag-images":
        raise SystemExit(run_tag_images(sys.argv[2:]))
    raise SystemExit(run_cli())
