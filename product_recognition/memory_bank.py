"""Memory-bank item preparation and pickle cache persistence.

The memory bank is the deployment artifact that connects dataset preparation
with inference. It stores the selected memory images, the evaluation splits used
for threshold tuning, optional precomputed memory embeddings, and summary
metadata in one trusted pickle file.

The cache keeps inference startup predictable: load one file, build a FAISS
index from cached embeddings, and classify images. Dataset filtering,
memory-image capping, leakage checks, and cache serialization all happen before
deployment.
"""

from __future__ import annotations

import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from .dataset import (
    assert_no_shelf_leakage,
    extract_shelf_id,
    filter_known_classes,
    make_negative_eval_items,
    scan_dataset,
    split_known_classes,
    summarize_known_classes,
)


def normalize_memory_item(
    item: dict,
    split_name: str = "memory",
    source_dataset: Optional[str] = None,
) -> dict:
    """Ensure a memory item has expected split, shelf, and source metadata.

    Input:
        item: Item dictionary with at least `image_path` and `label`.
        split_name: Default split name when the item has none.
        source_dataset: Optional dataset identifier to attach.
    Output:
        Normalized copy of the item.
    """
    normalized = dict(item)
    normalized["split"] = normalized.get("split", split_name)
    if not normalized.get("shelf_id"):
        normalized["shelf_id"] = extract_shelf_id(normalized["image_path"])
    if source_dataset is not None:
        normalized["source_dataset"] = source_dataset
    return normalized


def sort_memory_items(memory_items: Sequence[dict]) -> List[dict]:
    """Sort memory items deterministically by label, shelf, and path.

    Input:
        memory_items: Item records to sort.
    Output:
        New sorted list of item dictionaries.
    """
    return sorted(
        [dict(item) for item in memory_items],
        key=lambda item: (item.get("label", ""), item.get("shelf_id", ""), item.get("image_path", "")),
    )


def cap_memory_items_per_class(
    memory_items: Sequence[dict],
    max_images_per_class: Optional[int],
    seed: int,
) -> List[dict]:
    """Limit each class to a shelf-diverse maximum number of memory images.

    Input:
        memory_items: Candidate memory-bank item records.
        max_images_per_class: Maximum images kept per label, or None for no cap.
        seed: Random seed used when sampling across shelves.
    Output:
        Capped and sorted memory item records.
    """
    if max_images_per_class is None:
        return sort_memory_items(memory_items)
    if max_images_per_class <= 0:
        raise ValueError("max_images_per_class must be positive or None.")

    rng = random.Random(seed)
    label_to_items = defaultdict(list)
    for item in memory_items:
        normalized = normalize_memory_item(item)
        label_to_items[normalized["label"]].append(normalized)

    capped = []
    for label, items in sorted(label_to_items.items()):
        if len(items) <= max_images_per_class:
            capped.extend(sort_memory_items(items))
            continue

        # When a class has more images than the cap, sample across shelf/source
        # ids in rounds. This preserves visual diversity better than taking the
        # first N files or pure random samples.
        shelf_to_items = defaultdict(list)
        for item in items:
            shelf_to_items[item["shelf_id"]].append(item)
        for shelf_items in shelf_to_items.values():
            rng.shuffle(shelf_items)

        shelf_ids = sorted(shelf_to_items)
        rng.shuffle(shelf_ids)
        selected = []
        # Round-robin over shelves until the class cap is reached or every shelf
        # is exhausted.
        while len(selected) < max_images_per_class:
            added = False
            for shelf_id in shelf_ids:
                if len(selected) >= max_images_per_class:
                    break
                if shelf_to_items[shelf_id]:
                    selected.append(shelf_to_items[shelf_id].pop())
                    added = True
            if not added:
                break
        capped.extend(sort_memory_items(selected))

    return sort_memory_items(capped)


def create_memory_bank_from_dataset(
    dataset_root: str,
    negative_brand_folders: Iterable[str],
    negative_product_folders: Iterable[str],
    negative_label: str,
    min_images_per_class: int,
    test_ratio: float,
    seed: int,
    max_images_per_class: Optional[int],
) -> Tuple[List[dict], List[dict], List[dict], dict]:
    """Scan a dataset and create memory, known-test, and negative-test records.

    Input:
        dataset_root: Root folder with brand/product/image nesting.
        negative_brand_folders: Brand folders treated as unknown examples.
        negative_product_folders: Product folders treated as unknown examples.
        negative_label: Label assigned to unknown evaluation items.
        min_images_per_class: Minimum images required to keep a known class.
        test_ratio: Fraction of known images targeted for known-test evaluation.
        seed: Random seed for reproducible splitting and capping.
        max_images_per_class: Maximum memory images retained per class.
    Output:
        Memory items, known-test items, negative-test items, and metadata.
    """
    known_class_to_images, negative_images = scan_dataset(
        dataset_root=dataset_root,
        negative_brand_folders=negative_brand_folders,
        negative_product_folders=negative_product_folders,
    )
    known_summary_df = summarize_known_classes(known_class_to_images)
    # Very small classes cannot be split reliably and would create fragile
    # threshold metrics, so they are excluded from the MVP memory bank.
    filtered_known_class_to_images, removed_classes_df = filter_known_classes(
        known_class_to_images,
        min_images_per_class=min_images_per_class,
    )

    candidate_memory_items, known_test_items = split_known_classes(
        filtered_known_class_to_images,
        test_ratio=test_ratio,
        seed=seed,
    )
    # The split can create more memory candidates than we want to deploy. Apply
    # a per-class cap after leakage-safe splitting.
    memory_items = cap_memory_items_per_class(
        candidate_memory_items,
        max_images_per_class=max_images_per_class,
        seed=seed,
    )
    # This is a hard guardrail: evaluation must not contain crops from the same
    # source image as the memory bank.
    assert_no_shelf_leakage(memory_items, known_test_items)
    negative_test_items = make_negative_eval_items(negative_images, negative_label)

    # Metadata stays plain and serializable so the cache can be inspected
    # without loading the full item structures.
    metadata = {
        "dataset_root": dataset_root,
        "known_classes_found": len(known_class_to_images),
        "known_classes_kept": len(filtered_known_class_to_images),
        "removed_class_count": len(removed_classes_df),
        "negative_image_count": len(negative_images),
        "known_image_count": int(known_summary_df["count"].sum()) if len(known_summary_df) else 0,
        "candidate_memory_count": len(candidate_memory_items),
        "memory_count": len(memory_items),
        "known_test_count": len(known_test_items),
        "negative_test_count": len(negative_test_items),
    }
    return memory_items, known_test_items, negative_test_items, metadata


def save_memory_bank_cache(
    cache_dir: str,
    cache_name: str,
    memory_items: Sequence[dict],
    known_test_items: Optional[Sequence[dict]] = None,
    negative_test_items: Optional[Sequence[dict]] = None,
    memory_embeddings=None,
    memory_labels: Optional[Sequence[str]] = None,
    memory_paths: Optional[Sequence[str]] = None,
    metadata: Optional[dict] = None,
) -> Path:
    """Save memory-bank records and embeddings to a pickle cache.

    Input:
        cache_dir: Output folder for the cache file.
        cache_name: Cache filename or filename without `.pkl`.
        memory_items: Memory-bank records.
        known_test_items: Known-class evaluation records.
        negative_test_items: Unknown/negative evaluation records.
        memory_embeddings: Optional embeddings aligned with memory items.
        memory_labels: Optional labels aligned with memory embeddings.
        memory_paths: Optional image paths aligned with memory embeddings.
        metadata: Optional summary metadata.
    Output:
        Path to the written pickle cache.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    filename = cache_name if cache_name.endswith(".pkl") else f"{cache_name}.pkl"
    output_path = cache_path / filename

    # Keep the cache schema explicit. Optional embedding keys allow threshold
    # optimization to reuse cached vectors instead of recomputing them.
    payload = {
        "memory_items": [normalize_memory_item(item) for item in memory_items],
        "known_test_items": [normalize_memory_item(item, "known_test") for item in (known_test_items or [])],
        "negative_test_items": [normalize_memory_item(item, "negative_test") for item in (negative_test_items or [])],
        "memory_embeddings": memory_embeddings,
        "memory_labels": list(memory_labels) if memory_labels is not None else None,
        "memory_paths": list(memory_paths) if memory_paths is not None else None,
        "metadata": metadata or {},
    }

    with output_path.open("wb") as f:
        pickle.dump(payload, f)
    return output_path


def load_memory_bank_cache(cache_dir: str, cache_name: str) -> dict:
    """Load a memory-bank cache and fill optional legacy keys.

    Input:
        cache_dir: Folder containing the cache file.
        cache_name: Cache filename or filename without `.pkl`.
    Output:
        Cache payload dictionary with expected optional keys present.
    """
    filename = cache_name if cache_name.endswith(".pkl") else f"{cache_name}.pkl"
    input_path = Path(cache_dir) / filename
    with input_path.open("rb") as f:
        payload = pickle.load(f)

    required_keys = {"memory_items", "known_test_items", "negative_test_items"}
    missing = required_keys - set(payload)
    if missing:
        raise ValueError(f"Invalid memory-bank cache at {input_path}. Missing keys: {sorted(missing)}")

    # Older or manually created caches may not have every optional embedding key.
    # Set them to None so downstream code can use simple `payload.get(...)` logic.
    for key in [
        "metadata",
        "memory_embeddings",
        "memory_labels",
        "memory_paths",
        "known_test_embeddings",
        "known_test_labels",
        "known_test_paths",
        "negative_test_embeddings",
        "negative_test_labels",
        "negative_test_paths",
    ]:
        payload.setdefault(key, None if key != "metadata" else {})
    payload["cache_path"] = input_path
    return payload
