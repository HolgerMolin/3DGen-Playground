#!/usr/bin/env python3
"""Compare object_to_class.json against local GaussianVerse object directories."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLASS_MAP = REPO_ROOT / "object_labels" / "object_to_class.json"
DEFAULT_GS_ROOT = Path(os.environ.get("GS_PATH", "/home/tiangexiang/gen3d/gaussianverse"))


def scalar_sort_key(value: str) -> Tuple[int, object]:
    return (0, int(value)) if value.isdigit() else (1, value)


def object_sort_key(object_key: str) -> Tuple[Tuple[int, object], Tuple[int, object]]:
    cluster_id, object_id = object_key.split("/", 1)
    return scalar_sort_key(cluster_id), scalar_sort_key(object_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count which GaussianVerse objects listed in object_to_class.json "
        "are missing from the local dataset root.",
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
        "--top-k",
        type=int,
        default=10,
        help="How many clusters/classes to show in the console summary.",
    )
    parser.add_argument(
        "--show-samples",
        type=int,
        default=20,
        help="How many missing and unexpected object keys to print.",
    )
    parser.add_argument(
        "--write-missing",
        type=Path,
        help="Optional path to write one missing object key per line.",
    )
    parser.add_argument(
        "--write-unexpected",
        type=Path,
        help="Optional path to write one local-only object key per line.",
    )
    parser.add_argument(
        "--write-cluster-summary",
        type=Path,
        help="Optional CSV path for per-cluster expected/local/missing counts.",
    )
    parser.add_argument(
        "--write-class-summary",
        type=Path,
        help="Optional CSV path for missing counts grouped by class id.",
    )
    return parser.parse_args()


def load_class_map(class_map_path: Path) -> Dict[str, int]:
    with class_map_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected a dict in {class_map_path}, got {type(raw).__name__}")
    return {str(key): int(value) for key, value in raw.items()}


def iter_local_object_keys(gs_root: Path) -> Iterator[str]:
    with os.scandir(gs_root) as cluster_entries:
        for cluster_entry in cluster_entries:
            if not cluster_entry.is_dir(follow_symlinks=False):
                continue
            cluster_id = cluster_entry.name
            with os.scandir(cluster_entry.path) as object_entries:
                for object_entry in object_entries:
                    if object_entry.is_dir(follow_symlinks=False):
                        yield f"{cluster_id}/{object_entry.name}"


def counter_by_cluster(object_keys: Iterable[str]) -> Counter[str]:
    return Counter(object_key.split("/", 1)[0] for object_key in object_keys)


def write_lines(path: Path, values: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(f"{value}\n")


def write_cluster_summary_csv(
    path: Path,
    expected_by_cluster: Counter[str],
    local_by_cluster: Counter[str],
    missing_by_cluster: Counter[str],
    unexpected_by_cluster: Counter[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clusters = sorted(
        set(expected_by_cluster)
        | set(local_by_cluster)
        | set(missing_by_cluster)
        | set(unexpected_by_cluster),
        key=scalar_sort_key,
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cluster_id", "expected", "local", "missing", "unexpected"])
        for cluster_id in clusters:
            writer.writerow(
                [
                    cluster_id,
                    expected_by_cluster.get(cluster_id, 0),
                    local_by_cluster.get(cluster_id, 0),
                    missing_by_cluster.get(cluster_id, 0),
                    unexpected_by_cluster.get(cluster_id, 0),
                ]
            )


def write_class_summary_csv(path: Path, missing_by_class: Counter[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "missing"])
        for class_id, count in sorted(
            missing_by_class.items(), key=lambda item: (-item[1], item[0])
        ):
            writer.writerow([class_id, count])


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
            print(f"  {key}: {count:,} missing / {total:,} expected")


def main() -> int:
    args = parse_args()

    if not args.class_map.is_file():
        raise FileNotFoundError(f"Class map not found: {args.class_map}")
    if not args.gs_root.is_dir():
        raise FileNotFoundError(f"GaussianVerse root not found: {args.gs_root}")

    class_map = load_class_map(args.class_map)
    expected_keys = set(class_map)
    local_keys = set(iter_local_object_keys(args.gs_root))

    missing_keys = sorted(expected_keys - local_keys, key=object_sort_key)
    unexpected_keys = sorted(local_keys - expected_keys, key=object_sort_key)
    matched_count = len(expected_keys & local_keys)

    expected_by_cluster = counter_by_cluster(expected_keys)
    local_by_cluster = counter_by_cluster(local_keys)
    missing_by_cluster = counter_by_cluster(missing_keys)
    unexpected_by_cluster = counter_by_cluster(unexpected_keys)
    missing_by_class = Counter(class_map[object_key] for object_key in missing_keys)

    print(f"class_map={args.class_map}")
    print(f"gs_root={args.gs_root}")
    print(f"expected_objects={len(expected_keys):,}")
    print(f"local_object_dirs={len(local_keys):,}")
    print(f"matched_objects={matched_count:,}")
    print(f"missing_local_objects={len(missing_keys):,}")
    print(f"unexpected_local_objects={len(unexpected_keys):,}")
    coverage = 100.0 * matched_count / len(expected_keys) if expected_keys else 0.0
    print(f"local_coverage={coverage:.2f}%")
    print()

    print_counter_summary(
        title=f"top_missing_clusters (top {args.top_k})",
        values=missing_by_cluster,
        limit=args.top_k,
        suffix_lookup={cluster_id: count for cluster_id, count in expected_by_cluster.items()},
    )
    print()
    print_counter_summary(
        title=f"top_missing_classes (top {args.top_k})",
        values=missing_by_class,
        limit=args.top_k,
    )

    if args.show_samples > 0:
        print()
        print(f"sample_missing_objects (first {min(args.show_samples, len(missing_keys))})")
        for object_key in missing_keys[: args.show_samples]:
            print(f"  {object_key}")

        print()
        print(f"sample_unexpected_local_objects (first {min(args.show_samples, len(unexpected_keys))})")
        for object_key in unexpected_keys[: args.show_samples]:
            print(f"  {object_key}")

    if args.write_missing:
        write_lines(args.write_missing, missing_keys)
    if args.write_unexpected:
        write_lines(args.write_unexpected, unexpected_keys)
    if args.write_cluster_summary:
        write_cluster_summary_csv(
            args.write_cluster_summary,
            expected_by_cluster=expected_by_cluster,
            local_by_cluster=local_by_cluster,
            missing_by_cluster=missing_by_cluster,
            unexpected_by_cluster=unexpected_by_cluster,
        )
    if args.write_class_summary:
        write_class_summary_csv(args.write_class_summary, missing_by_class)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
