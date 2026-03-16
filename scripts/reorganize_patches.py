"""Reorganize patches from old layout to new {country}/{state}/{hash}/ layout.

Old layouts:
  data/patches/all_countries/{state}/{hash}/{id}.npy
  data/patches/chicken_eggs/united_states/{state}/{hash}/{id}.npy

New layout:
  data/patches/{country}/{state}/{hash}/{id}.npy

Also updates patch_meta.csv with corrected patch_path values and
removes empty old directories.

Usage (on the remote machine):
    cd /workspace/farm-mapping
    /workspace/farm-venv-cpu/bin/python scripts/reorganize_patches.py

Dry-run first:
    /workspace/farm-venv-cpu/bin/python scripts/reorganize_patches.py --dry-run
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

PATCHES_ROOT = Path("data/patches")


def _load_candidates(candidates_dir: Path) -> dict[str, str]:
    """Build candidate_id -> country mapping from all candidate CSVs."""
    mapping: dict[str, str] = {}
    for csv_path in candidates_dir.rglob("*.csv"):
        df = pd.read_csv(csv_path, usecols=["id", "country"])
        for _, row in df.iterrows():
            mapping[str(row["id"])] = str(row["country"])
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Reorganize patches to {country}/{state}/{hash}/ layout")
    parser.add_argument("--dry-run", action="store_true", help="Print what would happen without moving files")
    parser.add_argument("--patches-root", default=str(PATCHES_ROOT))
    parser.add_argument("--candidates-dir", default="data/candidates")
    args = parser.parse_args()

    patches_root = Path(args.patches_root)
    meta_path = patches_root / "patch_meta.csv"

    if not meta_path.exists():
        print(f"No patch_meta.csv found at {meta_path}")
        return

    meta = pd.read_csv(meta_path)
    print(f"Loaded {len(meta)} rows from {meta_path}")

    # Build candidate_id -> country lookup
    cid_to_country = _load_candidates(Path(args.candidates_dir))
    print(f"Loaded {len(cid_to_country)} candidate -> country mappings")

    moved = 0
    skipped = 0
    missing = 0
    updated_paths: list[str] = []

    for idx, row in meta.iterrows():
        old_rel = row["patch_path"]
        old_abs = patches_root / old_rel

        candidate_id = str(row["candidate_id"])
        country = cid_to_country.get(candidate_id, "")
        state = str(row.get("state", "")) if pd.notna(row.get("state")) else ""
        imagery_hash = str(row.get("imagery_config_hash", "unknown"))

        country_part = country if country else "_"
        state_part = state if state else "_"
        new_rel = Path(country_part) / state_part / imagery_hash / f"{candidate_id}.npy"
        new_abs = patches_root / new_rel

        updated_paths.append(str(new_rel))

        if old_abs == new_abs:
            skipped += 1
            continue

        if not old_abs.exists():
            # Check if already at new location
            if new_abs.exists():
                skipped += 1
                continue
            missing += 1
            if missing <= 5:
                print(f"  MISSING: {old_abs}")
            continue

        if args.dry_run:
            print(f"  MOVE: {old_rel} -> {new_rel}")
            moved += 1
            continue

        new_abs.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_abs), str(new_abs))
        moved += 1

    # Update patch_meta.csv
    meta["patch_path"] = updated_paths
    if not args.dry_run:
        meta.to_csv(meta_path, index=False)
        print(f"Updated {meta_path}")

    print(f"\nDone: {moved} moved, {skipped} already correct, {missing} missing")

    # Cleanup empty directories
    if not args.dry_run:
        removed_dirs = 0
        for d in sorted(patches_root.rglob("*"), reverse=True):
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
                removed_dirs += 1
        print(f"Removed {removed_dirs} empty directories")


if __name__ == "__main__":
    main()
