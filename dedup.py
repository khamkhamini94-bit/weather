"""Deduplicate weather dataset by MD5 hash across data/ class folders and dataset2/.

Rules:
- Exact duplicates: keep the copy in data/<ClassFolder>/ , remove others
- Near-duplicates (pHash): report only, don't auto-remove
- data/train/ and data/val/ are NOT scanned (they're intentional copies from split)
"""

import hashlib
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATASET2 = ROOT / "dataset2"

# Which directories to scan for duplicates (NOT train/val — those are intentional copies)
SCAN_DIRS = [
    DATA / "Cloudy",
    DATA / "Rain",
    DATA / "Shine",
    DATA / "snow",
    DATASET2,
]

# Priority: when duplicates found, keep the one in the highest-priority directory
# Lower number = higher priority
PRIORITY = {
    str(DATA / "Cloudy"): 0,
    str(DATA / "Rain"): 0,
    str(DATA / "Shine"): 0,
    str(DATA / "snow"): 0,
    str(DATASET2): 1,
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def md5(path: Path) -> str:
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def find_exact_duplicates(scan_dirs: list[Path]) -> dict[str, list[Path]]:
    """Return {hash: [paths]} for hashes with >1 occurrence."""
    hash_to_paths = defaultdict(list)
    for d in scan_dirs:
        if not d.is_dir():
            print(f"  SKIP (not found): {d}")
            continue
        for p in d.iterdir():
            if p.suffix.lower() in IMG_EXTS:
                try:
                    h = md5(p)
                    hash_to_paths[h].append(p)
                except OSError as e:
                    print(f"  WARN: cannot read {p}: {e}")

    return {h: paths for h, paths in hash_to_paths.items() if len(paths) > 1}


def find_near_duplicates(scan_dirs: list[Path], threshold: int = 5) -> list[tuple[Path, Path, int]]:
    """Find near-duplicate pairs using perceptual hash (pHash).

    threshold: Hamming distance cutoff (default 5 = very similar).
    Returns list of (path1, path2, distance).
    """
    try:
        from PIL import Image
        import imagehash
    except ImportError:
        print("  imagehash not installed; skipping near-duplicate check.")
        print("  Install with: pip install imagehash Pillow")
        return []

    all_images = []
    for d in scan_dirs:
        if not d.is_dir():
            continue
        for p in d.iterdir():
            if p.suffix.lower() in IMG_EXTS:
                all_images.append(p)

    # Compute pHash for all images
    phashes = {}
    for p in all_images:
        try:
            img = Image.open(p).convert("RGB")
            phashes[p] = imagehash.phash(img)
        except Exception as e:
            print(f"  WARN: cannot hash {p}: {e}")

    # Compare all pairs (O(n^2)) — for ~18k images this is ~162M comparisons, too slow.
    # Instead, we only compare within the same stem prefix to catch filename overlaps.
    # For cross-directory duplicates, we rely on MD5.
    near_pairs = []
    paths_by_stem = defaultdict(list)
    for p in phashes:
        paths_by_stem[p.stem].append(p)

    for stem, paths in paths_by_stem.items():
        if len(paths) < 2:
            continue
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                p1, p2 = paths[i], paths[j]
                d = phashes[p1] - phashes[p2]
                if 0 < d <= threshold:
                    near_pairs.append((p1, p2, d))

    return near_pairs


def main():
    print("=" * 60)
    print("Weather Dataset Deduplication")
    print("=" * 60)

    # ── Phase 1: exact duplicates ──
    print("\n[Phase 1] Computing MD5 hashes for exact duplicate detection...")
    dup_map = find_exact_duplicates(SCAN_DIRS)

    if not dup_map:
        print("  No exact duplicates found.")
    else:
        total_dup_groups = len(dup_map)
        total_dup_files = sum(len(paths) for paths in dup_map.values())
        total_removable = total_dup_files - total_dup_groups  # keep one per group
        print(f"  Found {total_dup_groups} duplicate groups ({total_dup_files} files involved)")
        print(f"  Can remove {total_removable} files (keeping one per group)")

        # Show details
        print("\n  Duplicate details:")
        for h, paths in dup_map.items():
            # Sort by priority
            paths_sorted = sorted(paths, key=lambda p: PRIORITY.get(str(p.parent), 99))
            keeper = paths_sorted[0]
            victims = paths_sorted[1:]
            print(f"\n    Hash: {h[:16]}...")
            print(f"    KEEP:  {keeper}")
            for v in victims:
                print(f"    DEL:   {v}")

        # ── Remove duplicates ──
        print(f"\n  Removing {total_removable} duplicate files...")
        removed = 0
        for h, paths in dup_map.items():
            paths_sorted = sorted(paths, key=lambda p: PRIORITY.get(str(p.parent), 99))
            for victim in paths_sorted[1:]:
                try:
                    victim.unlink()
                    removed += 1
                except OSError as e:
                    print(f"    ERROR removing {victim}: {e}")
        print(f"  Removed: {removed} files")

    # ── Phase 2: near-duplicates ──
    print("\n[Phase 2] Perceptual hash near-duplicate detection...")
    near = find_near_duplicates(SCAN_DIRS)
    if not near:
        print("  No near-duplicates found (or imagehash not available).")
    else:
        print(f"  Found {len(near)} near-duplicate pairs:")
        for p1, p2, d in near:
            print(f"    dist={d}: {p1}  <->  {p2}")
        print("  NOTE: near-duplicates NOT auto-removed. Review manually if needed.")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("Deduplication complete.")
    remaining = 0
    for d in SCAN_DIRS:
        if d.is_dir():
            n = sum(1 for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)
            print(f"  {d.name:12s}: {n:6d} images")
            remaining += n
    print(f"  {'TOTAL':12s}: {remaining:6d} images")
    print("=" * 60)


if __name__ == "__main__":
    main()
