"""CLI for creating a memory-bank cache from the dataset folder.

This command scans the dataset, builds leakage-safe memory/evaluation splits,
embeds the selected memory images, and writes a single cache file used by
threshold optimization and inference.

The command prints CUDA status and item counts before embedding. Slow cache
creation is commonly caused by CPU-only PyTorch, a small batch size, or a large
number of memory-bank images.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..config import (
    DEFAULT_CACHE_DIR,
    DEFAULT_MAX_IMAGES_PER_CLASS,
    DEFAULT_MIN_IMAGES_PER_CLASS,
    DEFAULT_MODEL_NAME,
    DEFAULT_SEED,
    DEFAULT_TEST_RATIO,
    DEFAULT_UNKNOWN_LABEL,
)
from ..embeddings import cuda_status_lines, extract_embeddings_for_items, load_embedding_model, resolve_device
from ..memory_bank import create_memory_bank_from_dataset, save_memory_bank_cache


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for memory-bank creation.

    Input:
        Values passed through the command line.
    Output:
        argparse Namespace with dataset, cache, model, and split settings.
    """
    parser = argparse.ArgumentParser(description="Create a memory-bank cache from a product image dataset.")
    parser.add_argument("--dataset-root", default="data/dataset")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--cache-name", default=None)
    parser.add_argument("--min-images-per-class", type=int, default=DEFAULT_MIN_IMAGES_PER_CLASS)
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-images-per-class", type=int, default=DEFAULT_MAX_IMAGES_PER_CLASS)
    parser.add_argument("--unknown-label", default=DEFAULT_UNKNOWN_LABEL)
    parser.add_argument("--unknown-brand-folder", action="append", default=None)
    parser.add_argument("--unknown-product-folder", action="append", default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def cache_name_for(args: argparse.Namespace) -> str:
    """Return the explicit or default cache name.

    Input:
        args: Parsed CLI arguments.
    Output:
        Cache name without requiring a `.pkl` suffix.
    """
    if args.cache_name:
        return args.cache_name
    return f"product_memory_bank_{Path(args.dataset_root).name}_seed{args.seed}_max{args.max_images_per_class}"


def unknown_brand_folders(args: argparse.Namespace) -> set[str]:
    """Return brand-folder names treated as unknown examples.

    Input:
        args: Parsed CLI arguments.
    Output:
        Set of brand folder names to scan as unknown/negative data.
    """
    return set(args.unknown_brand_folder or [args.unknown_label])


def unknown_product_folders(args: argparse.Namespace) -> set[str]:
    """Return product-folder names treated as unknown examples.

    Input:
        args: Parsed CLI arguments.
    Output:
        Set of product folder names to scan as unknown/negative data.
    """
    return set(args.unknown_product_folder or ["unknown"])


def main() -> None:
    """Create the memory-bank cache and print a short summary.

    Input:
        Command-line arguments and dataset images on disk.
    Output:
        None. Writes a cache file and prints metadata to stdout.
    """
    args = parse_args()
    # Dataset scanning and splitting happen before model loading. If the dataset
    # layout is wrong, fail early without spending time loading model weights.
    memory_items, known_test_items, negative_test_items, metadata = create_memory_bank_from_dataset(
        dataset_root=args.dataset_root,
        negative_brand_folders=unknown_brand_folders(args),
        negative_product_folders=unknown_product_folders(args),
        negative_label=args.unknown_label,
        min_images_per_class=args.min_images_per_class,
        test_ratio=args.test_ratio,
        seed=args.seed,
        max_images_per_class=args.max_images_per_class,
    )

    resolved_device = resolve_device(args.device)
    print("Runtime device status:")
    for line in cuda_status_lines(args.device):
        print(f"  {line}")
    print(f"Memory-bank images to embed: {len(memory_items)}")
    print(f"Batch size: {args.batch_size}")

    # Only memory embeddings are stored in the cache by default. Known/negative
    # test embeddings can be generated later during threshold optimization.
    model = load_embedding_model(args.model_name, resolved_device)
    embeddings, labels, paths = extract_embeddings_for_items(
        items=memory_items,
        model=model,
        batch_size=args.batch_size,
        desc="Embedding memory-bank images",
    )

    metadata.update(
        {
            "operation": "create",
            "model_name": args.model_name,
            "device": resolved_device,
            "max_images_per_class": args.max_images_per_class,
            "memory_embedding_count": int(embeddings.shape[0]),
            "memory_embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        }
    )
    cache_path = save_memory_bank_cache(
        cache_dir=args.cache_dir,
        cache_name=cache_name_for(args),
        memory_items=memory_items,
        known_test_items=known_test_items,
        negative_test_items=negative_test_items,
        memory_embeddings=embeddings,
        memory_labels=labels,
        memory_paths=paths,
        metadata=metadata,
    )
    print("Saved memory-bank cache:", cache_path)
    print(pd.DataFrame([metadata]).T.rename(columns={0: "value"}).to_string())


if __name__ == "__main__":
    main()
