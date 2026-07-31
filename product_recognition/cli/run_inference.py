"""CLI for running product-recognition inference on images.

This command provides single-image and batch inference for the backend
integration API. It loads the same recognizer runtime dictionary that service
code keeps in memory, then runs either one image or a folder of images.
Single-image mode prints detailed JSON; folder mode prints a table or writes CSV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ..config import DEFAULT_CACHE_DIR, DEFAULT_MODEL_NAME, DEFAULT_TOP_K, DEFAULT_UNKNOWN_LABEL
from ..inference import list_image_paths, load_recognizer, predict_image, predict_images


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for inference.

    Input:
        Values passed through the command line.
    Output:
        argparse Namespace with cache, threshold, model, and image settings.
    """
    parser = argparse.ArgumentParser(description="Run product-recognition inference.")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--cache-name", required=True)
    parser.add_argument("--threshold-dir", default=None)
    parser.add_argument("--image-path", default=None)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--unknown-label", default=DEFAULT_UNKNOWN_LABEL)
    return parser.parse_args()


def detailed_output_path(output_csv: str) -> Path:
    """Return the detailed CSV path paired with the compact output path."""
    path = Path(output_csv)
    return path.with_name(f"{path.stem}_detailed{path.suffix or '.csv'}")


def compact_results_df(df: pd.DataFrame) -> pd.DataFrame:
    """Build the client-facing inference table."""
    required_cols = {"image_path", "final_pred"}
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        raise ValueError(f"Inference results are missing columns: {sorted(missing_cols)}")
    return df[["image_path", "final_pred"]].rename(columns={"image_path": "crop_path"})


def write_inference_outputs(df: pd.DataFrame, output_csv: str) -> None:
    """Write compact and detailed inference CSV outputs."""
    compact_path = Path(output_csv)
    detailed_path = detailed_output_path(output_csv)
    compact_results_df(df).to_csv(compact_path, index=False)
    df.to_csv(detailed_path, index=False)
    print("Saved compact inference results to:", compact_path)
    print("Saved detailed inference results to:", detailed_path)


def main() -> None:
    """Run single-image or folder inference from the command line.

    Input:
        Command-line arguments, cache artifacts, and image files on disk.
    Output:
        None. Prints predictions or writes them to a CSV file.
    """
    args = parse_args()
    if not args.image_path and not args.image_dir:
        raise ValueError("Provide --image-path or --image-dir.")

    # Load once and reuse for all requested images, matching the recommended
    # backend service pattern.
    recognizer = load_recognizer(
        cache_dir=args.cache_dir,
        cache_name=args.cache_name,
        threshold_dir=args.threshold_dir,
        model_name=args.model_name,
        device=args.device,
        top_k=args.top_k,
        unknown_label=args.unknown_label,
    )

    if args.image_path:
        # Single-image mode includes thresholds, neighbors, and rejection
        # reasons for debugging prediction behavior.
        result = predict_image(recognizer, args.image_path)
        print(json.dumps(result, indent=2, default=str))
        if args.output_csv:
            write_inference_outputs(pd.DataFrame([result]), args.output_csv)
        return

    # Folder mode supports batch evaluation and CSV generation.
    paths = list_image_paths(args.image_dir)
    df = predict_images(recognizer, paths)
    if args.output_csv:
        write_inference_outputs(df, args.output_csv)
    else:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
