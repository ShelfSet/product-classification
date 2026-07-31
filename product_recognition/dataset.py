"""Dataset scanning and leakage-safe train/test splitting helpers.

The dataset is expected to be organized as `brand/product/image`. Known product
folders become class labels in `brand__product` format, while configured
unknown folders are kept separate as negative examples for threshold tuning.

The important principle in this file is leakage prevention. Product crops often
come from larger source or shelf images, so a random crop-level split can put
near-duplicate crops into both memory and evaluation. To avoid inflated metrics,
the split logic groups crops by a normalized source id (`shelf_id`) and moves
whole source groups together.
"""

from __future__ import annotations

import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def extract_shelf_id(image_path: str) -> str:
    """Extract a source-image identifier used to prevent crop leakage.

    Input:
        image_path: Path to an image crop.
    Output:
        A normalized source id based on the image filename stem.
    """
    stem = Path(image_path).stem
    stem = re.sub(r"_crop_\d+$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_crop\d+$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_obj\d+$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"^(train|valid|val|test)_", "", stem, flags=re.IGNORECASE)
    return stem


def is_negative_folder_pair(
    brand_name: str,
    product_name: str,
    negative_brand_folders: Iterable[str],
    negative_product_folders: Iterable[str],
) -> bool:
    """Return whether a brand/product folder should be treated as unknown.

    Input:
        brand_name: Dataset brand folder name.
        product_name: Dataset product folder name.
        negative_brand_folders: Brand folder names reserved for unknown images.
        negative_product_folders: Product folder names reserved for unknown images.
    Output:
        True when either folder name marks the image group as unknown.
    """
    negative_brand_folders = set(negative_brand_folders)
    negative_product_folders = set(negative_product_folders)
    return brand_name.strip() in negative_brand_folders or product_name.strip() in negative_product_folders


def scan_dataset(
    dataset_root: str,
    negative_brand_folders: Iterable[str],
    negative_product_folders: Iterable[str],
) -> Tuple[Dict[str, List[str]], List[str]]:
    """Scan a dataset folder into known classes and unknown images.

    Input:
        dataset_root: Root folder with brand/product/image nesting.
        negative_brand_folders: Brand folders to treat as unknown.
        negative_product_folders: Product folders to treat as unknown.
    Output:
        A mapping of known labels to image paths, plus unknown image paths.
    """
    known_class_to_images = defaultdict(list)
    negative_images = []

    # The scanner only understands a simple brand/product/image hierarchy.
    # Keeping labels derived from folders makes the cache easy to inspect later.
    for brand_name in sorted(os.listdir(dataset_root)):
        brand_path = os.path.join(dataset_root, brand_name)
        if not os.path.isdir(brand_path):
            continue

        for product_name in sorted(os.listdir(brand_path)):
            product_path = os.path.join(brand_path, product_name)
            if not os.path.isdir(product_path):
                continue

            image_paths = []
            for fname in sorted(os.listdir(product_path)):
                if os.path.splitext(fname)[1].lower() in VALID_IMAGE_EXTS:
                    image_paths.append(os.path.join(product_path, fname))

            if not image_paths:
                continue

            # Unknown folders are not memory-bank classes. They are held out as
            # negatives so threshold optimization can learn when to reject.
            if is_negative_folder_pair(
                brand_name=brand_name,
                product_name=product_name,
                negative_brand_folders=negative_brand_folders,
                negative_product_folders=negative_product_folders,
            ):
                negative_images.extend(image_paths)
            else:
                known_class_to_images[f"{brand_name}__{product_name}"].extend(image_paths)

    return dict(known_class_to_images), negative_images


def summarize_known_classes(class_to_images: Dict[str, List[str]]) -> pd.DataFrame:
    """Create a class-count summary table for known product folders.

    Input:
        class_to_images: Mapping of known labels to image paths.
    Output:
        DataFrame with `label` and `count` columns.
    """
    rows = [{"label": label, "count": len(paths)} for label, paths in class_to_images.items()]
    if not rows:
        return pd.DataFrame(columns=["label", "count"])
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


def filter_known_classes(
    class_to_images: Dict[str, List[str]],
    min_images_per_class: int,
) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    """Drop known classes that do not have enough images for splitting.

    Input:
        class_to_images: Mapping of known labels to image paths.
        min_images_per_class: Minimum images required to keep a class.
    Output:
        Filtered class mapping and a DataFrame describing removed classes.
    """
    filtered = {
        label: paths
        for label, paths in class_to_images.items()
        if len(paths) >= min_images_per_class
    }
    removed = sorted(set(class_to_images) - set(filtered))
    removed_df = pd.DataFrame(
        [{"label": label, "count": len(class_to_images[label])} for label in removed]
    )
    if "label" not in removed_df.columns:
        return filtered, pd.DataFrame(columns=["label", "count"])
    if len(removed_df):
        removed_df = removed_df.sort_values("count", ascending=True).reset_index(drop=True)
    return filtered, removed_df


def _build_shelf_units(
    class_to_images: Dict[str, List[str]],
) -> Tuple[Dict[str, Dict[str, List[str]]], Dict[str, List[str]]]:
    """Group image paths by shelf/source id for leakage-safe splitting.

    Input:
        class_to_images: Mapping of known labels to image paths.
    Output:
        Shelf-to-label-paths mapping and label-to-shelf-ids mapping.
    """
    unit_to_label_paths = defaultdict(lambda: defaultdict(list))
    label_to_units = defaultdict(set)

    for label, paths in class_to_images.items():
        for path in paths:
            shelf_id = extract_shelf_id(path)
            unit_to_label_paths[shelf_id][label].append(path)
            label_to_units[label].add(shelf_id)

    return (
        {unit_id: dict(label_paths) for unit_id, label_paths in unit_to_label_paths.items()},
        {label: sorted(unit_ids) for label, unit_ids in label_to_units.items()},
    )


def _count_label_images_in_units(
    unit_ids: set,
    unit_to_label_paths: Dict[str, Dict[str, List[str]]],
) -> Counter:
    """Count class images contained in a set of shelf units.

    Input:
        unit_ids: Shelf/source ids to count.
        unit_to_label_paths: Shelf grouping created by `_build_shelf_units`.
    Output:
        Counter of label to image count.
    """
    counts = Counter()
    for unit_id in unit_ids:
        for label, paths in unit_to_label_paths[unit_id].items():
            counts[label] += len(paths)
    return counts


def _unit_label_counts(
    unit_id: str,
    unit_to_label_paths: Dict[str, Dict[str, List[str]]],
) -> Counter:
    """Count class images inside one shelf unit.

    Input:
        unit_id: Shelf/source id to inspect.
        unit_to_label_paths: Shelf grouping created by `_build_shelf_units`.
    Output:
        Counter of label to image count for that unit.
    """
    return Counter(
        {label: len(paths) for label, paths in unit_to_label_paths[unit_id].items()}
    )


def _can_move_unit_to_test(
    unit_id: str,
    test_units: set,
    total_label_counts: Counter,
    unit_to_label_paths: Dict[str, Dict[str, List[str]]],
    min_train_images_per_class: int = 1,
) -> bool:
    """Check whether moving a shelf unit preserves enough train images.

    Input:
        unit_id: Candidate shelf/source id for the test split.
        test_units: Shelf ids already assigned to test.
        total_label_counts: Total image counts per class.
        unit_to_label_paths: Shelf grouping created by `_build_shelf_units`.
        min_train_images_per_class: Minimum images left in memory per class.
    Output:
        True when the candidate can move to test without emptying a class.
    """
    current_test_counts = _count_label_images_in_units(test_units, unit_to_label_paths)
    candidate_counts = _unit_label_counts(unit_id, unit_to_label_paths)

    for label, add_count in candidate_counts.items():
        remaining_train = total_label_counts[label] - (current_test_counts[label] + add_count)
        if remaining_train < min_train_images_per_class:
            return False
    return True


def _items_from_units(
    unit_ids: set,
    unit_to_label_paths: Dict[str, Dict[str, List[str]]],
    split_name: str,
) -> List[dict]:
    """Convert grouped shelf units into flat split item records.

    Input:
        unit_ids: Shelf/source ids to convert.
        unit_to_label_paths: Shelf grouping created by `_build_shelf_units`.
        split_name: Split label to write into each item.
    Output:
        List of item dictionaries with image path, label, split, and shelf id.
    """
    items = []
    for unit_id in sorted(unit_ids):
        for label, paths in sorted(unit_to_label_paths[unit_id].items()):
            for path in sorted(paths):
                items.append(
                    {
                        "image_path": path,
                        "label": label,
                        "split": split_name,
                        "shelf_id": unit_id,
                    }
                )
    return items


def split_known_classes(
    class_to_images: Dict[str, List[str]],
    test_ratio: float,
    seed: int,
) -> Tuple[List[dict], List[dict]]:
    """Create leakage-safe memory and known-test item lists.

    Input:
        class_to_images: Mapping of known labels to image paths.
        test_ratio: Target fraction of images to hold out for known testing.
        seed: Random seed used for reproducible shelf selection.
    Output:
        Memory item records and known-test item records.
    """
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be between 0 and 1.")

    rng = random.Random(seed)
    unit_to_label_paths, label_to_units = _build_shelf_units(class_to_images)
    total_label_counts = Counter({label: len(paths) for label, paths in class_to_images.items()})
    target_test_counts = {}

    # Compute class-level targets, but cap each class so at least one image can
    # remain in the memory bank.
    for label, total in total_label_counts.items():
        if total < 2:
            target_test_counts[label] = 0
        else:
            target_test_counts[label] = min(max(1, int(round(total * test_ratio))), total - 1)

    labels_by_difficulty = sorted(
        class_to_images.keys(),
        key=lambda label: (len(label_to_units.get(label, [])), total_label_counts[label], label),
    )
    test_units = set()

    # Handle classes with fewer shelf/source units first. They have less freedom,
    # so choosing their test units early reduces the chance of an impossible split.
    for label in labels_by_difficulty:
        target = target_test_counts[label]
        if target <= 0:
            continue

        candidate_units = list(label_to_units.get(label, []))
        rng.shuffle(candidate_units)
        # Prefer small units that are not already in test. This keeps test close
        # to the target ratio and avoids pulling too many extra labels with a unit.
        candidate_units = sorted(
            candidate_units,
            key=lambda unit_id: (
                unit_id in test_units,
                len(unit_to_label_paths[unit_id].get(label, [])),
                sum(len(paths) for paths in unit_to_label_paths[unit_id].values()),
                unit_id,
            ),
        )

        for unit_id in candidate_units:
            if _count_label_images_in_units(test_units, unit_to_label_paths)[label] >= target:
                break
            if unit_id in test_units:
                continue
            if _can_move_unit_to_test(unit_id, test_units, total_label_counts, unit_to_label_paths):
                test_units.add(unit_id)

    # Safety pass: every class with a test target should get at least one test
    # unit if there is any valid way to do so.
    for label in labels_by_difficulty:
        if target_test_counts[label] <= 0:
            continue
        if _count_label_images_in_units(test_units, unit_to_label_paths)[label] > 0:
            continue

        candidate_units = list(label_to_units.get(label, []))
        rng.shuffle(candidate_units)
        candidate_units = sorted(
            candidate_units,
            key=lambda unit_id: (
                len(unit_to_label_paths[unit_id].get(label, [])),
                sum(len(paths) for paths in unit_to_label_paths[unit_id].values()),
                unit_id,
            ),
        )
        for unit_id in candidate_units:
            if unit_id in test_units:
                continue
            if _can_move_unit_to_test(unit_id, test_units, total_label_counts, unit_to_label_paths):
                test_units.add(unit_id)
                break

    all_units = set(unit_to_label_paths.keys())
    return (
        _items_from_units(all_units - test_units, unit_to_label_paths, "memory"),
        _items_from_units(test_units, unit_to_label_paths, "known_test"),
    )


def make_negative_eval_items(negative_paths: List[str], negative_label: str) -> List[dict]:
    """Create evaluation records for unknown/negative images.

    Input:
        negative_paths: Image paths from unknown folders.
        negative_label: Label stored as the ground truth for unknown images.
    Output:
        Negative-test item records.
    """
    return [
        {
            "image_path": path,
            "label": negative_label,
            "split": "negative_test",
            "shelf_id": extract_shelf_id(path),
        }
        for path in negative_paths
    ]


def build_shelf_split_diagnostics(memory_items: List[dict], known_test_items: List[dict]) -> pd.DataFrame:
    """Build a table showing whether shelf ids appear in both splits.

    Input:
        memory_items: Item records assigned to the memory bank.
        known_test_items: Item records assigned to known-test evaluation.
    Output:
        DataFrame with per-shelf split membership and leakage flags.
    """
    memory_shelves = {item.get("shelf_id") for item in memory_items if item.get("shelf_id")}
    test_shelves = {item.get("shelf_id") for item in known_test_items if item.get("shelf_id")}
    rows = []
    for shelf_id in sorted(memory_shelves | test_shelves):
        rows.append(
            {
                "shelf_id": shelf_id,
                "in_memory": shelf_id in memory_shelves,
                "in_test": shelf_id in test_shelves,
                "leakage": shelf_id in memory_shelves and shelf_id in test_shelves,
            }
        )
    return pd.DataFrame(rows)


def assert_no_shelf_leakage(memory_items: List[dict], known_test_items: List[dict]) -> None:
    """Validate that memory and known-test splits do not share shelf ids.

    Input:
        memory_items: Item records assigned to the memory bank.
        known_test_items: Item records assigned to known-test evaluation.
    Output:
        None. Raises ValueError when leakage is found.
    """
    diagnostics_df = build_shelf_split_diagnostics(memory_items, known_test_items)
    if len(diagnostics_df) == 0:
        return
    leakage_df = diagnostics_df[diagnostics_df["leakage"]]
    if len(leakage_df):
        examples = leakage_df["shelf_id"].head(20).tolist()
        raise ValueError(
            "Shelf leakage detected between memory and test splits. "
            f"Count: {len(leakage_df)}. Examples: {examples}"
        )
