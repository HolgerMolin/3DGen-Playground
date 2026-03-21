#!/usr/bin/env python3
"""Audit expected GaussianVerse objects for missing local files."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLASS_MAP = REPO_ROOT / "object_labels" / "object_to_class.json"
DEFAULT_GS_ROOT = Path(os.environ.get("GS_PATH", "/home/tiangexiang/gen3d/gaussianverse"))
DEFAULT_REQUIRED_FILES = ("point_cloud.ply", "gs2sphere.npy")


def scalar_sort_key(value: str) -> Tuple[int, object]:
    return (0, int(value)) if value.isdigit() else (1, value)


def object_sort_key(object_key: str) -> Tuple[Tuple[int, object], Tuple[int, object]]:
    cluster_id, object_id = object_key.split("/", 1)
    return scalar_sort_key(cluster_id), scalar_sort_key(object_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check which expected GaussianVerse objects are incomplete on local disk.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--class-map",
        type=Path,
        default=DEFAULT_CLASS_MAP,
        help="Path to object_to_class.json.",
    )
    parser.add_argument(
        "--gs-root",
        type=Path,
        default=DEFAULT_GS_ROOT,
        help="GaussianVerse root directory laid out as cluster_id/object_id.",
    )
    parser.add_argument(
        "--require",
        nargs="+",
        default=list(DEFAULT_REQUIRED_FILES),
        help="Required files that each local object directory must contain.",
    )
    parser.add_argument(
        "--nonempty",
        action="store_true",
        help="Treat zero-byte required files as missing.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many clusters/classes/file types to show in the console summary.",
    )
    parser.add_argument(
        "--show-samples",
        type=int,
        default=20,
        help="How many missing-dir and incomplete object keys to print.",
    )
    parser.add_argument(
        "--write-missing-dirs",
        type=Path,
        help="Optional path to write expected object keys whose directory is missing.",
    )
    parser.add_argument(
        "--write-incomplete",
        type=Path,
        help="Optional CSV path to write expected object keys missing required files.",
    )
    return parser.parse_args()


def load_class_map(class_map_path: Path) -> Dict[str, int]:
    with class_map_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected a dict in {class_map_path}, got {type(raw).__name__}")
    return {str(key): int(value) for key, value in raw.items()}


def counter_by_cluster(object_keys: Iterable[str]) -> Counter[str]:
    return Counter(object_key.split("/", 1)[0] for object_key in object_keys)


def write_lines(path: Path, values: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(f"{value}\n")


def write_incomplete_csv(path: Path, incomplete: Sequence[Tuple[str, List[str]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["object_key", "missing_files"])
        for object_key, missing_files in incomplete:
            writer.writerow([object_key, ";".join(missing_files)])


def inspect_object_dir(
    gs_root: Path,
    object_key: str,
    required_files: Sequence[str],
    nonempty: bool,
) -> Tuple[bool, List[str]]:
    cluster_id, object_id = object_key.split("/", 1)
    object_dir = gs_root / cluster_id / object_id
    if not object_dir.is_dir():
        return False, list(required_files)

    missing_files: List[str] = []
    for filename in required_files:
        file_path = object_dir / filename
        if not file_path.is_file():
            missing_files.append(filename)
            continue
        if nonempty and file_path.stat().st_size == 0:
            missing_files.append(filename)
    return True, missing_files


def print_counter_summary(
    title: str,
    values: Counter,
    limit: int,
    suffix_lookup: Dict[str, int] | None = None,
) -> None:
    print(title)
    if not values:
        print("  none")
        return

    for key, count in values.most_common(limit):
        if suffix_lookup is None:
            print(f"  {key}: {count:,}")
        else:
            total = suffix_lookup.get(str(key), 0)
            print(f"  {key}: {count:,} affected / {total:,} expected")


def main() -> int:
    args = parse_args()

    if not args.class_map.is_file():
        raise FileNotFoundError(f"Class map not found: {args.class_map}")
    if not args.gs_root.is_dir():
        raise FileNotFoundError(f"GaussianVerse root not found: {args.gs_root}")

    class_map = load_class_map(args.class_map)
    expected_keys = sorted(class_map, key=object_sort_key)

    missing_dirs: List[str] = []
    incomplete: List[Tuple[str, List[str]]] = []
    complete_count = 0
    missing_files_counter: Counter[str] = Counter()
    affected_by_class: Counter[int] = Counter()
    expected_by_cluster = counter_by_cluster(expected_keys)

    for object_key in expected_keys:
        has_dir, missing_files = inspect_object_dir(
            gs_root=args.gs_root,
            object_key=object_key,
            required_files=args.require,
            nonempty=args.nonempty,
        )
        if not has_dir:
            missing_dirs.append(object_key)
            affected_by_class[class_map[object_key]] += 1
            for filename in args.require:
                missing_files_counter[filename] += 1
            continue
        if missing_files:
            incomplete.append((object_key, missing_files))
            affected_by_class[class_map[object_key]] += 1
            missing_files_counter.update(missing_files)
            continue
        complete_count += 1

    affected_keys = missing_dirs + [object_key for object_key, _ in incomplete]
    affected_by_cluster = counter_by_cluster(affected_keys)

    print(f"class_map={args.class_map}")
    print(f"gs_root={args.gs_root}")
    print(f"required_files={','.join(args.require)}")
    print(f"expected_objects={len(expected_keys):,}")
    print(f"complete_objects={complete_count:,}")
    print(f"missing_dirs={len(missing_dirs):,}")
    print(f"incomplete_dirs={len(incomplete):,}")
    print(f"affected_objects={len(affected_keys):,}")
    completeness = 100.0 * complete_count / len(expected_keys) if expected_keys else 0.0
    print(f"local_completeness={completeness:.2f}%")
    print()

    print_counter_summary(
        title=f"top_missing_files (top {args.top_k})",
        values=missing_files_counter,
        limit=args.top_k,
    )
    print()
    print_counter_summary(
        title=f"top_affected_clusters (top {args.top_k})",
        values=affected_by_cluster,
        limit=args.top_k,
        suffix_lookup={cluster_id: count for cluster_id, count in expected_by_cluster.items()},
    )
    print()
    print_counter_summary(
        title=f"top_affected_classes (top {args.top_k})",
        values=affected_by_class,
        limit=args.top_k,
    )

    if args.show_samples > 0:
        print()
        print(f"sample_missing_dirs (first {min(args.show_samples, len(missing_dirs))})")
        for object_key in missing_dirs[: args.show_samples]:
            print(f"  {object_key}")

        print()
        print(f"sample_incomplete_dirs (first {min(args.show_samples, len(incomplete))})")
        for object_key, missing_files in incomplete[: args.show_samples]:
            print(f"  {object_key}: {', '.join(missing_files)}")

    if args.write_missing_dirs:
        write_lines(args.write_missing_dirs, missing_dirs)
    if args.write_incomplete:
        write_incomplete_csv(args.write_incomplete, incomplete)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
