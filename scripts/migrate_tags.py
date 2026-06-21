"""
Migrate old tags from previous material directory to new location.

Background:
  Tags are stored under storage/tags/, keyed by the absolute path of the
  image file (with separators replaced for safety). When images move to a
  new directory, the old tag files are still valid (same file_hash), but
  they are stored at the wrong path — the system looks for them under the
  new absolute-path key and won't find them.

What this script does:
  For each old tag file, compute the new image absolute path and copy the
  tag to the corresponding new location. No Vision API calls needed.

Usage:
  python scripts/migrate_tags.py

Dry-run mode (no files copied, just report what would happen):
  python scripts/migrate_tags.py --dry-run
"""

import json
import os
import shutil
import sys

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
STORAGE_TAGS = os.path.join(PROJECT_ROOT, "storage", "tags")

# Old path → new path mapping
OLD_MATERIAL_DIR = "/materials"  # Previous config.toml value
NEW_MATERIAL_DIR = "D:/CODE/vedio-library/251208-蜡笔小新-总合集"  # Current value

DRY_RUN = "--dry-run" in sys.argv


def normalize_path(abs_path: str) -> str:
    """Mirrors tagging._normalize_path_for_tags()."""
    safe = abs_path.replace(":", "_").replace("\\", "/")
    safe = safe.lstrip("/")
    return safe


def main():
    # The old tags were created on a different system where OLD_MATERIAL_DIR
    # normalized to "materials" (no Windows drive letter prefix). We detect
    # the actual old tag directory by trying both the computed path and the
    # raw directory name.
    candidates = [
        os.path.join(STORAGE_TAGS, normalize_path(os.path.abspath(OLD_MATERIAL_DIR))),
        os.path.join(STORAGE_TAGS, OLD_MATERIAL_DIR.strip("/")),
    ]
    old_tag_root = None
    for c in candidates:
        if os.path.isdir(c):
            old_tag_root = c
            break

    new_normalized = normalize_path(os.path.abspath(NEW_MATERIAL_DIR))
    new_tag_root = os.path.join(STORAGE_TAGS, new_normalized)

    if not os.path.isdir(old_tag_root):
        print(f"[ERROR] Old tag directory not found: {old_tag_root}")
        print("Nothing to migrate.")
        return

    if not os.path.isdir(NEW_MATERIAL_DIR):
        print(f"[ERROR] New material directory not found: {NEW_MATERIAL_DIR}")
        print("Please verify the images exist at the expected location.")
        return

    migrated = 0
    skipped = 0
    errors = []

    for root, dirs, files in os.walk(old_tag_root):
        for fname in files:
            if not fname.endswith(".tags.json"):
                continue

            old_tag_path = os.path.join(root, fname)

            # Compute the new tag path by replacing the old prefix with new
            rel = os.path.relpath(old_tag_path, old_tag_root)
            new_tag_path = os.path.join(new_tag_root, rel)

            # Verify corresponding image exists
            # The tag stores file_path relative to base_dir (e.g. "001-PNG/xxx.png")
            try:
                with open(old_tag_path, "r", encoding="utf-8") as f:
                    tag_data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                errors.append(f"Failed to read {old_tag_path}: {e}")
                continue

            relative_image = tag_data.get("file_path", "")
            new_image_path = os.path.join(NEW_MATERIAL_DIR, relative_image)

            if not os.path.isfile(new_image_path):
                skipped += 1
                continue  # Image missing, skip tag

            if os.path.isfile(new_tag_path):
                skipped += 1
                continue  # Already exists

            if DRY_RUN:
                print(f"[DRY-RUN] Would copy: {old_tag_path} → {new_tag_path}")
                migrated += 1
            else:
                os.makedirs(os.path.dirname(new_tag_path), exist_ok=True)
                try:
                    shutil.copy2(old_tag_path, new_tag_path)
                    migrated += 1
                except OSError as e:
                    errors.append(f"Failed to copy {old_tag_path}: {e}")

    print(f"\n── Migration {'dry-run ' if DRY_RUN else ''}summary ──")
    print(f"  Migrated: {migrated}")
    print(f"  Skipped:  {skipped}")
    print(f"  Errors:   {len(errors)}")
    if errors:
        for e in errors[:10]:
            print(f"  ! {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")

    if DRY_RUN and migrated > 0:
        print("\nRun without --dry-run to apply the migration.")


if __name__ == "__main__":
    main()
