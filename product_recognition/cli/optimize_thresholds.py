"""CLI for optimizing global and per-class score thresholds.

This is the second step in the workflow after memory-bank creation. It loads the
cache, scores known and unknown evaluation images against the memory bank, and
selects score thresholds used by inference. The command writes a deployable
`thresholds.csv` and a class-level diagnostic `thresholds_w_metrics.csv`.
"""

from __future__ import annotations

import argparse

import pandas as pd

from ..config import (
    DEFAULT_CACHE_DIR,
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEED,
    DEFAULT_TOP_K,
    DEFAULT_UNKNOWN_LABEL,
)
from ..thresholds import optimize_thresholds_from_cache


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for threshold optimization.

    Input:
        Values passed through the command line.
    Output:
        argparse Namespace with cache, model, and threshold settings.
    """
    parser = argparse.ArgumentParser(description="Optimize global and dynamic thresholds.")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--cache-name", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--unknown-label", default=DEFAULT_UNKNOWN_LABEL)
    parser.add_argument("--calibration-ratio", type=float, default=0.50)
    parser.add_argument("--score-min", type=float, default=0.65)
    parser.add_argument("--score-max", type=float, default=0.80)
    parser.add_argument("--score-step", type=float, default=0.01)
    parser.add_argument("--min-known-accept-rate", type=float, default=0.97)
    parser.add_argument("--dynamic-step", type=float, default=0.01)
    parser.add_argument("--dynamic-max-steps", type=int, default=15)
    parser.add_argument("--dynamic-target-accept-rate", type=float, default=0.98)
    parser.add_argument("--dynamic-min-class-samples", type=int, default=3)
    parser.add_argument("--dynamic-max-accepted-accuracy-drop", type=float, default=0.03)
    parser.add_argument("--dynamic-min-class-unknown-samples", type=int, default=1)
    parser.add_argument("--dynamic-max-added-unknown-accepts", type=int, default=0)
    parser.add_argument("--dynamic-max-unknown-false-accept-rate-increase", type=float, default=0.005)
    parser.add_argument("--dynamic-max-raise-steps", type=int, default=15)
    parser.add_argument("--dynamic-target-class-unknown-false-accept-rate", type=float, default=0.01)
    parser.add_argument("--dynamic-max-known-correct-accept-rate-drop", type=float, default=0.03)
    return parser.parse_args()


def main() -> None:
    """Run threshold optimization and print evaluation summaries.

    Input:
        Command-line arguments and a memory-bank cache on disk.
    Output:
        None. Writes two threshold artifacts and prints summary metrics.
    """
    args = parse_args()
    # The CLI maps arguments to the threshold optimizer and prints the generated
    # artifact paths.
    result = optimize_thresholds_from_cache(
        cache_dir=args.cache_dir,
        cache_name=args.cache_name,
        output_dir=args.output_dir,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
        top_k=args.top_k,
        unknown_label=args.unknown_label,
        calibration_ratio=args.calibration_ratio,
        score_min=args.score_min,
        score_max=args.score_max,
        score_step=args.score_step,
        min_known_accept_rate=args.min_known_accept_rate,
        dynamic_step=args.dynamic_step,
        dynamic_max_steps=args.dynamic_max_steps,
        dynamic_target_accept_rate=args.dynamic_target_accept_rate,
        dynamic_min_class_samples=args.dynamic_min_class_samples,
        dynamic_max_accepted_accuracy_drop=args.dynamic_max_accepted_accuracy_drop,
        dynamic_min_class_unknown_samples=args.dynamic_min_class_unknown_samples,
        dynamic_max_added_unknown_accepts=args.dynamic_max_added_unknown_accepts,
        dynamic_max_unknown_false_accept_rate_increase=(
            args.dynamic_max_unknown_false_accept_rate_increase
        ),
        dynamic_max_raise_steps=args.dynamic_max_raise_steps,
        dynamic_target_class_unknown_false_accept_rate=(
            args.dynamic_target_class_unknown_false_accept_rate
        ),
        dynamic_max_known_correct_accept_rate_drop=(
            args.dynamic_max_known_correct_accept_rate_drop
        ),
    )
    print("Saved threshold optimization results to:", result["output_dir"])
    print("Threshold table:", result["threshold_file"])
    print("Threshold metrics:", result["metrics_file"])
    print(pd.DataFrame([result["global_eval_summary"], result["dynamic_eval_summary"]]).to_string(index=False))


if __name__ == "__main__":
    main()
