"""Threshold calibration and evaluation for open-set product recognition.

The recognizer always retrieves the nearest known product class, so a separate
threshold policy is needed to decide when that nearest class is trustworthy. This
module learns that policy from two cached evaluation sets:

- known-test images, which should usually be accepted as their true class
- unknown/negative images, which should usually be rejected as `__Unknown__`

The optimization uses a conservative threshold policy. First it selects a single
global score threshold that balances known acceptance and unknown rejection. Then
it raises thresholds for classes with concentrated unknown false accepts and
optionally lowers thresholds for classes with safe known recovery. The final
deployable artifact is `thresholds.csv`; the extended per-class diagnostics live
in `thresholds_w_metrics.csv`.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DEFAULT_UNKNOWN_LABEL
from .embeddings import extract_embeddings_for_items, load_embedding_model, resolve_device
from .inference import aggregate_topk_predictions, build_faiss_index
from .memory_bank import load_memory_bank_cache


THRESHOLD_FILE = "thresholds.csv"
METRICS_FILE = "thresholds_w_metrics.csv"


def inclusive_grid(start: float, stop: float, step: float) -> np.ndarray:
    """Create an inclusive numeric grid for threshold sweeps.

    Input:
        start: First grid value.
        stop: Last grid value to include when reachable.
        step: Positive increment.
    Output:
        NumPy array of threshold candidates.
    """
    if step <= 0:
        raise ValueError("Grid step must be positive.")
    values = []
    current = start
    epsilon = step / 10.0
    while current <= stop + epsilon:
        values.append(round(float(current), 10))
        current += step
    return np.asarray(values, dtype="float64")


def labels_and_paths_from_items(items: Sequence[dict]) -> tuple[list[str], list[str]]:
    """Extract aligned labels and paths from item records.

    Input:
        items: Records containing `label` and `image_path`.
    Output:
        Labels and paths in matching order.
    """
    return [item["label"] for item in items], [item["image_path"] for item in items]


def validate_embeddings(name: str, embeddings, labels: Sequence[str], paths: Sequence[str]) -> None:
    """Validate that embeddings, labels, and paths are aligned.

    Input:
        name: Human-readable split name used in error messages.
        embeddings: Embedding matrix to validate.
        labels: Labels aligned with embedding rows.
        paths: Image paths aligned with embedding rows.
    Output:
        None. Raises ValueError when alignment is invalid.
    """
    if embeddings is None:
        raise ValueError(f"{name} embeddings are missing.")
    if embeddings.ndim != 2:
        raise ValueError(f"{name} embeddings must be 2D, got shape {embeddings.shape}.")
    if len(embeddings) != len(labels) or len(embeddings) != len(paths):
        raise ValueError(
            f"{name} alignment mismatch: {len(embeddings)} embeddings, "
            f"{len(labels)} labels, {len(paths)} paths."
        )


def get_or_extract_split_embeddings(
    payload: dict,
    split_name: str,
    items: Sequence[dict],
    model,
    model_name: str,
    device: str,
    batch_size: int,
):
    """Load cached split embeddings or extract them if missing.

    Input:
        payload: Memory-bank cache payload.
        split_name: Split prefix such as `memory` or `known_test`.
        items: Split item records.
        model: Existing model instance, or None to lazily load one.
        model_name: Embedding model id or local model path.
        device: Resolved runtime device.
        batch_size: Number of images embedded per model call.
    Output:
        Embeddings, labels, paths, and the loaded/reused model.
    """
    emb_key = f"{split_name}_embeddings"
    label_key = f"{split_name}_labels"
    path_key = f"{split_name}_paths"

    embeddings = payload.get(emb_key)
    labels = payload.get(label_key)
    paths = payload.get(path_key)

    if embeddings is not None:
        if labels is None or paths is None:
            labels, paths = labels_and_paths_from_items(items)
        validate_embeddings(split_name, embeddings, labels, paths)
        return embeddings, list(labels), list(paths), model

    if model is None:
        model = load_embedding_model(model_name, device)

    embeddings, labels, paths = extract_embeddings_for_items(
        items=items,
        model=model,
        batch_size=batch_size,
        desc=f"Embedding {split_name} images",
    )
    validate_embeddings(split_name, embeddings, labels, paths)
    return embeddings, labels, paths, model


def subset(values, idx):
    """Select array/list values by index.

    Input:
        values: NumPy array or Python sequence.
        idx: Integer indices to select.
    Output:
        Subset of values in the requested order.
    """
    if isinstance(values, np.ndarray):
        return values[idx]
    return [values[i] for i in idx]


def split_calibration_evaluation(
    embeddings: np.ndarray,
    labels: Sequence[str],
    paths: Sequence[str],
    calibration_ratio: float,
    seed: int,
    stratify: bool,
):
    """Split embeddings into calibration and holdout evaluation subsets.

    Input:
        embeddings: Embedding matrix.
        labels: Labels aligned with embeddings.
        paths: Paths aligned with embeddings.
        calibration_ratio: Fraction used for threshold selection.
        seed: Random seed for reproducible splitting.
        stratify: Whether to stratify by label when possible.
    Output:
        Calibration arrays/lists, evaluation arrays/lists, and split metadata.
    """
    if not 0.0 < calibration_ratio < 1.0:
        raise ValueError("calibration_ratio must be between 0 and 1.")

    indices = np.arange(len(labels))
    stratify_labels = None
    can_stratify = False
    if stratify:
        label_counts = Counter(labels)
        can_stratify = bool(label_counts) and min(label_counts.values()) >= 2
        if can_stratify:
            stratify_labels = list(labels)

    # Calibration chooses thresholds; evaluation estimates how those thresholds
    # perform on data not used for selection.
    calib_idx, eval_idx = train_test_split(
        indices,
        train_size=calibration_ratio,
        random_state=seed,
        shuffle=True,
        stratify=stratify_labels,
    )
    report = {
        "total_count": len(labels),
        "calibration_count": len(calib_idx),
        "evaluation_count": len(eval_idx),
        "stratified": can_stratify,
    }
    return (
        subset(embeddings, calib_idx),
        subset(labels, calib_idx),
        subset(paths, calib_idx),
        subset(embeddings, eval_idx),
        subset(labels, eval_idx),
        subset(paths, eval_idx),
        report,
    )


def build_score_table(
    embeddings: np.ndarray,
    labels: Sequence[str],
    paths: Sequence[str],
    index,
    memory_labels: Sequence[str],
    memory_paths: Sequence[str],
    top_k: int,
) -> pd.DataFrame:
    """Score embeddings against the memory bank and return prediction rows.

    Input:
        embeddings: Query embedding matrix.
        labels: True labels aligned with query embeddings.
        paths: Image paths aligned with query embeddings.
        index: FAISS index built from memory embeddings.
        memory_labels: Labels aligned with memory-bank rows.
        memory_paths: Paths aligned with memory-bank rows.
        top_k: Number of nearest neighbors to retrieve.
    Output:
        DataFrame with raw best-label scores for each query.
    """
    rows = []
    for emb, true_label, path in zip(embeddings, labels, paths):
        # This mirrors inference scoring exactly, but keeps true labels so the
        # optimizer can measure correctness and rejection behavior.
        out = aggregate_topk_predictions(
            query_embedding=emb,
            index=index,
            memory_labels=memory_labels,
            memory_paths=memory_paths,
            top_k=top_k,
        )
        rows.append(
            {
                "image_path": path,
                "true_label": true_label,
                "best_label": out["best_label"],
                "best_score": out["best_score"],
                "second_label": out["second_label"],
                "second_score": out["second_score"],
                "margin": out["margin"],
            }
        )
    return pd.DataFrame(rows)


def apply_global_threshold(
    score_df: pd.DataFrame,
    score_threshold: float,
    unknown_label: str,
) -> pd.DataFrame:
    """Apply one global score threshold to prediction scores.

    Input:
        score_df: Raw prediction score table.
        score_threshold: Minimum score required to accept a known label.
        unknown_label: Output label for rejected predictions.
    Output:
        DataFrame with `final_pred`, rejection flags, and threshold columns.
    """
    out = score_df.copy()
    out["is_rejected"] = out["best_score"] < score_threshold
    out["final_pred"] = np.where(out["is_rejected"], unknown_label, out["best_label"])
    out["effective_score_threshold"] = score_threshold
    out["threshold_offset"] = 0.0
    return out


def apply_dynamic_threshold(
    score_df: pd.DataFrame,
    global_score_threshold: float,
    class_score_thresholds: dict[str, float],
    unknown_label: str,
) -> pd.DataFrame:
    """Apply class-specific thresholds with a global fallback.

    Input:
        score_df: Raw prediction score table.
        global_score_threshold: Fallback accept/reject threshold.
        class_score_thresholds: Label-specific threshold overrides.
        unknown_label: Output label for rejected predictions.
    Output:
        DataFrame with dynamic thresholds and final predictions.
    """
    out = score_df.copy()
    out["effective_score_threshold"] = out["best_label"].map(class_score_thresholds)
    out["effective_score_threshold"] = out["effective_score_threshold"].fillna(global_score_threshold)
    out["is_rejected"] = out["best_score"] < out["effective_score_threshold"]
    out["final_pred"] = np.where(out["is_rejected"], unknown_label, out["best_label"])
    out["threshold_offset"] = out["effective_score_threshold"] - global_score_threshold
    return out


def summarize_open_set_from_thresholded(known_df: pd.DataFrame, negative_df: pd.DataFrame) -> dict:
    """Summarize known-class acceptance and unknown rejection metrics.

    Input:
        known_df: Thresholded predictions for known-class images.
        negative_df: Thresholded predictions for unknown/negative images.
    Output:
        Dictionary of accept/reject rates and count metrics.
    """
    num_known = len(known_df)
    num_negative = len(negative_df)
    known_accept_mask = ~known_df["is_rejected"]
    known_accept = int(known_accept_mask.sum())
    known_reject = int(known_df["is_rejected"].sum())
    known_correct_accept = int((known_accept_mask & (known_df["best_label"] == known_df["true_label"])).sum())
    negative_true_reject = int(negative_df["is_rejected"].sum())
    negative_false_accept = int((~negative_df["is_rejected"]).sum())

    return {
        "known_count": num_known,
        "negative_count": num_negative,
        "known_accept_rate": known_accept / num_known if num_known else 0.0,
        "known_false_reject_rate": known_reject / num_known if num_known else 0.0,
        "known_accuracy_on_all": known_correct_accept / num_known if num_known else 0.0,
        "known_accuracy_on_accepted": known_correct_accept / known_accept if known_accept else 0.0,
        "negative_true_reject_rate": negative_true_reject / num_negative if num_negative else 0.0,
        "negative_false_accept_rate": negative_false_accept / num_negative if num_negative else 0.0,
        "known_accept_count": known_accept,
        "known_reject_count": known_reject,
        "known_correct_accept_count": known_correct_accept,
        "negative_true_reject_count": negative_true_reject,
        "negative_false_accept_count": negative_false_accept,
    }


def evaluate_global_threshold_grid(
    known_score_df: pd.DataFrame,
    negative_score_df: pd.DataFrame,
    score_values: Iterable[float],
    unknown_label: str,
) -> pd.DataFrame:
    """Evaluate candidate global thresholds on calibration scores.

    Input:
        known_score_df: Raw known-image calibration scores.
        negative_score_df: Raw unknown-image calibration scores.
        score_values: Candidate score thresholds.
        unknown_label: Output label for rejected predictions.
    Output:
        DataFrame with one metric row per threshold.
    """
    rows = []
    for score_threshold in score_values:
        # Both known and negative calibration sets are thresholded for every
        # candidate so the sweep can trade off false rejects vs false accepts.
        known_df = apply_global_threshold(known_score_df, score_threshold, unknown_label)
        negative_df = apply_global_threshold(negative_score_df, score_threshold, unknown_label)
        rows.append(
            {
                "score_threshold": float(score_threshold),
                **summarize_open_set_from_thresholded(known_df, negative_df),
            }
        )
    return pd.DataFrame(rows)


def choose_best_global_threshold(grid_df: pd.DataFrame, min_known_accept_rate: float) -> pd.Series:
    """Select the best global threshold under the known-acceptance constraint.

    Input:
        grid_df: Threshold sweep metrics.
        min_known_accept_rate: Minimum acceptable known-image acceptance rate.
    Output:
        Selected threshold row.
    """
    df = grid_df.copy()
    filtered = df[df["known_accept_rate"] >= min_known_accept_rate].copy()
    if len(filtered):
        df = filtered
    # Prefer strong unknown rejection first, but only after preserving enough
    # known-image acceptance. Later sort keys break ties toward better accuracy.
    df = df.sort_values(
        by=[
            "negative_true_reject_rate",
            "known_accuracy_on_all",
            "known_accuracy_on_accepted",
            "known_false_reject_rate",
            "score_threshold",
        ],
        ascending=[False, False, False, True, False],
    ).reset_index(drop=True)
    return df.iloc[0]


def known_metrics_for_threshold(df: pd.DataFrame, score_threshold: float) -> dict:
    """Compute known-class metrics for one score threshold.

    Input:
        df: Raw score table for known-class images.
        score_threshold: Candidate score threshold.
    Output:
        Dictionary of known-image acceptance and accuracy metrics.
    """
    accepted = df["best_score"] >= score_threshold
    correct_accept = accepted & (df["best_label"] == df["true_label"])
    n = len(df)
    n_accept = int(accepted.sum())
    n_correct_accept = int(correct_accept.sum())
    return {
        "known_count": n,
        "known_accept_rate": n_accept / n if n else 0.0,
        "known_false_reject_rate": 1.0 - (n_accept / n if n else 0.0),
        "known_accuracy_on_all": n_correct_accept / n if n else 0.0,
        "known_accuracy_on_accepted": n_correct_accept / n_accept if n_accept else 0.0,
        "accepted_count": n_accept,
        "correct_accepted_count": n_correct_accept,
    }


def predicted_class_open_set_metrics(
    known_score_df: pd.DataFrame,
    negative_score_df: pd.DataFrame,
    label: str,
    score_threshold: float,
) -> dict:
    """Measure known precision and unknown risk for one predicted class.

    Input:
        known_score_df: Raw scores for all known calibration images.
        negative_score_df: Raw scores for all unknown calibration images.
        label: Predicted class whose runtime threshold is being evaluated.
        score_threshold: Candidate accept/reject threshold.
    Output:
        Predicted-class known and unknown acceptance metrics.
    """
    predicted_known_df = known_score_df[known_score_df["best_label"] == label]
    accepted_known = predicted_known_df["best_score"] >= score_threshold
    correct_known = predicted_known_df["true_label"] == label
    accepted_correct_known = accepted_known & correct_known
    accepted_error_known = accepted_known & ~correct_known

    predicted_unknown_df = negative_score_df[negative_score_df["best_label"] == label]
    accepted_unknown = predicted_unknown_df["best_score"] >= score_threshold

    predicted_known_accept_count = int(accepted_known.sum())
    predicted_known_correct_accept_count = int(accepted_correct_known.sum())
    known_error_accept_count = int(accepted_error_known.sum())
    class_unknown_count = len(predicted_unknown_df)
    unknown_false_accept_count = int(accepted_unknown.sum())
    total_unknown_count = len(negative_score_df)

    return {
        "predicted_known_count": len(predicted_known_df),
        "predicted_known_accept_count": predicted_known_accept_count,
        "predicted_known_correct_accept_count": predicted_known_correct_accept_count,
        "known_error_accept_count": known_error_accept_count,
        "known_precision_on_accepted": (
            predicted_known_correct_accept_count / predicted_known_accept_count
            if predicted_known_accept_count
            else 0.0
        ),
        "unknown_count": total_unknown_count,
        "class_unknown_count": class_unknown_count,
        "unknown_false_accept_count": unknown_false_accept_count,
        "unknown_false_accept_rate": (
            unknown_false_accept_count / total_unknown_count if total_unknown_count else 0.0
        ),
        "class_unknown_false_accept_rate": (
            unknown_false_accept_count / class_unknown_count if class_unknown_count else 0.0
        ),
    }


def build_class_thresholds(
    known_score_df: pd.DataFrame,
    global_score_threshold: float,
    step: float,
    max_steps: int,
    target_accept_rate: float,
    min_class_samples: int,
    max_accepted_accuracy_drop: float,
    negative_score_df: pd.DataFrame | None = None,
    min_class_unknown_samples: int = 1,
    max_added_unknown_accepts: int = 0,
    max_unknown_false_accept_rate_increase: float = 0.005,
    max_raise_steps: int = 15,
    target_class_unknown_false_accept_rate: float = 0.01,
    max_known_correct_accept_rate_drop: float = 0.03,
) -> pd.DataFrame:
    """Build unknown-aware class thresholds around the global value.

    Risky classes can rise above the global threshold before safe lowering is
    considered for classes without excess unknown risk.

    Input:
        known_score_df: Raw known-image calibration scores.
        global_score_threshold: Selected global threshold.
        step: Threshold adjustment size.
        max_steps: Maximum number of downward steps to test.
        target_accept_rate: Accept-rate target used for recovery.
        min_class_samples: Minimum samples before class-specific tuning.
        max_accepted_accuracy_drop: Allowed drop in predicted-class precision.
        negative_score_df: Raw unknown-image calibration scores.
        min_class_unknown_samples: Minimum unknowns attracted to a class.
        max_added_unknown_accepts: Allowed extra unknown accepts per class.
        max_unknown_false_accept_rate_increase: Aggregate unknown-rate budget.
        max_raise_steps: Maximum number of upward threshold steps to test.
        target_class_unknown_false_accept_rate: Target among attracted unknowns.
        max_known_correct_accept_rate_drop: Allowed correct-known rate loss.
    Output:
        DataFrame with selected threshold and metrics for each class.
    """
    if step <= 0:
        raise ValueError("Dynamic threshold step must be positive.")
    if max_steps < 0:
        raise ValueError("Dynamic threshold max_steps must be non-negative.")
    if max_raise_steps < 0:
        raise ValueError("Dynamic threshold max_raise_steps must be non-negative.")
    if min_class_samples < 1:
        raise ValueError("min_class_samples must be at least 1.")
    if min_class_unknown_samples < 0:
        raise ValueError("min_class_unknown_samples must be non-negative.")
    if max_added_unknown_accepts < 0:
        raise ValueError("max_added_unknown_accepts must be non-negative.")
    if max_accepted_accuracy_drop < 0:
        raise ValueError("max_accepted_accuracy_drop must be non-negative.")
    if max_unknown_false_accept_rate_increase < 0:
        raise ValueError("max_unknown_false_accept_rate_increase must be non-negative.")
    if not 0.0 <= target_class_unknown_false_accept_rate <= 1.0:
        raise ValueError("target_class_unknown_false_accept_rate must be between 0 and 1.")
    if not 0.0 <= max_known_correct_accept_rate_drop <= 1.0:
        raise ValueError("max_known_correct_accept_rate_drop must be between 0 and 1.")

    # Keep the optional input for direct callers, but the main optimizer always
    # supplies unknown scores. Without them, selection skips the unknown-support
    # guardrails while retaining correct-known recovery checks.
    use_unknown_safety = negative_score_df is not None
    if negative_score_df is None:
        negative_score_df = known_score_df.iloc[0:0].copy()

    lower_candidate_thresholds = [
        max(-1.0, round(global_score_threshold + k * step, 10))
        for k in range(-max_steps, 1)
    ]
    raise_candidate_thresholds = [
        min(1.0, round(global_score_threshold + k * step, 10))
        for k in range(1, max_raise_steps + 1)
    ]
    candidate_thresholds = sorted(set(lower_candidate_thresholds + raise_candidate_thresholds))
    rows = []
    global_candidate_rows = {}

    for label, class_df in known_score_df.groupby("true_label"):
        class_df = class_df.copy()
        n = len(class_df)
        global_metrics = known_metrics_for_threshold(class_df, global_score_threshold)
        global_open_set_metrics = predicted_class_open_set_metrics(
            known_score_df,
            negative_score_df,
            label,
            global_score_threshold,
        )

        # Test thresholds on both sides of the global value. Upward candidates
        # reduce class-specific unknown risk; downward candidates recover knowns.
        candidate_rows = []
        for threshold in candidate_thresholds:
            candidate_rows.append(
                {
                    "label": label,
                    "candidate_threshold": threshold,
                    "offset": threshold - global_score_threshold,
                    **known_metrics_for_threshold(class_df, threshold),
                    **predicted_class_open_set_metrics(
                        known_score_df,
                        negative_score_df,
                        label,
                        threshold,
                    ),
                }
            )

        candidates_df = pd.DataFrame(candidate_rows)
        correct_accept_delta = (
            candidates_df["correct_accepted_count"] - global_metrics["correct_accepted_count"]
        )
        candidates_df["recovered_correct_count"] = correct_accept_delta.clip(lower=0)
        candidates_df["lost_correct_count"] = (-correct_accept_delta).clip(lower=0)
        known_error_delta = (
            candidates_df["known_error_accept_count"]
            - global_open_set_metrics["known_error_accept_count"]
        )
        candidates_df["added_known_error_count"] = known_error_delta.clip(lower=0)
        candidates_df["removed_known_error_count"] = (-known_error_delta).clip(lower=0)
        unknown_accept_delta = (
            candidates_df["unknown_false_accept_count"]
            - global_open_set_metrics["unknown_false_accept_count"]
        )
        candidates_df["added_unknown_accept_count"] = unknown_accept_delta.clip(lower=0)
        candidates_df["removed_unknown_false_accept_count"] = (-unknown_accept_delta).clip(
            lower=0
        )
        candidates_df["meets_target_accept_rate"] = (
            candidates_df["known_accept_rate"] >= target_accept_rate
        )
        candidates_df["meets_target_class_unknown_false_accept_rate"] = (
            candidates_df["class_unknown_false_accept_rate"]
            <= target_class_unknown_false_accept_rate
        )
        candidates_df["global_threshold"] = global_score_threshold
        candidates_df["global_known_accept_rate"] = global_metrics["known_accept_rate"]
        candidates_df["global_known_accuracy_on_all"] = global_metrics["known_accuracy_on_all"]
        candidates_df["global_known_accuracy_on_accepted"] = global_metrics[
            "known_accuracy_on_accepted"
        ]
        candidates_df["global_known_precision_on_accepted"] = global_open_set_metrics[
            "known_precision_on_accepted"
        ]
        candidates_df["global_known_error_accept_count"] = global_open_set_metrics[
            "known_error_accept_count"
        ]
        candidates_df["global_unknown_false_accept_count"] = global_open_set_metrics[
            "unknown_false_accept_count"
        ]
        candidates_df["global_unknown_false_accept_rate"] = global_open_set_metrics[
            "unknown_false_accept_rate"
        ]
        candidates_df["global_class_unknown_false_accept_rate"] = global_open_set_metrics[
            "class_unknown_false_accept_rate"
        ]

        global_idx = (candidates_df["candidate_threshold"] - global_score_threshold).abs().idxmin()
        global_candidate_rows[label] = candidates_df.loc[global_idx].copy()
        selected = candidates_df.loc[
            global_idx
        ].copy()
        selected["selection_reason"] = "keep_global"

        # Small calibration samples are noisy, so use the global threshold unless
        # a class has enough examples to justify class-specific tuning.
        if n < min_class_samples:
            selected["selection_reason"] = "fallback_small_class_keep_global"
        elif (
            use_unknown_safety
            and global_open_set_metrics["class_unknown_count"] < min_class_unknown_samples
        ):
            # No attracted unknowns is missing evidence, not evidence that a lower
            # threshold is safe for this predicted class.
            selected["selection_reason"] = "fallback_insufficient_unknown_support_keep_global"
        else:
            global_precision = global_open_set_metrics["known_precision_on_accepted"]
            min_allowed_precision = max(0.0, global_precision - max_accepted_accuracy_drop)
            global_class_unknown_rate = global_open_set_metrics[
                "class_unknown_false_accept_rate"
            ]
            has_excess_unknown_risk = (
                use_unknown_safety
                and global_open_set_metrics["unknown_false_accept_count"] > 0
                and global_class_unknown_rate > target_class_unknown_false_accept_rate
            )

            if has_excess_unknown_risk:
                # Raising is class-local: reduce unknown false accepts while
                # preserving almost all correct known accepts for this class.
                min_allowed_known_correct_rate = max(
                    0.0,
                    global_metrics["known_accuracy_on_all"]
                    - max_known_correct_accept_rate_drop,
                )
                raise_safe = candidates_df[
                    (candidates_df["candidate_threshold"] >= global_score_threshold)
                    & (
                        candidates_df["known_accuracy_on_all"]
                        >= min_allowed_known_correct_rate
                    )
                    & (candidates_df["known_precision_on_accepted"] >= min_allowed_precision)
                    & (
                        candidates_df["unknown_false_accept_count"]
                        < global_open_set_metrics["unknown_false_accept_count"]
                    )
                ].copy()

                if len(raise_safe):
                    target_candidates = raise_safe[
                        raise_safe["meets_target_class_unknown_false_accept_rate"]
                    ].copy()
                    if len(target_candidates):
                        # Once the risk target is met, retain known performance
                        # and choose the smallest sufficient upward adjustment.
                        selected = target_candidates.sort_values(
                            by=[
                                "known_accuracy_on_all",
                                "candidate_threshold",
                                "unknown_false_accept_count",
                                "known_precision_on_accepted",
                            ],
                            ascending=[False, True, True, False],
                        ).iloc[0].copy()
                        selected["selection_reason"] = "raised_meets_unknown_false_accept_target"
                    else:
                        # When the target is infeasible under the known-recall
                        # constraint, still take the best safe risk reduction.
                        selected = raise_safe.sort_values(
                            by=[
                                "unknown_false_accept_count",
                                "known_accuracy_on_all",
                                "candidate_threshold",
                                "known_precision_on_accepted",
                            ],
                            ascending=[True, False, True, False],
                        ).iloc[0].copy()
                        selected["selection_reason"] = "raised_reduces_unknown_false_accepts"
                else:
                    # Do not lower a class whose existing unknown risk could not
                    # be reduced without exceeding the known-recall constraint.
                    selected["selection_reason"] = "keep_global_unresolved_unknown_risk"
            else:
                safe = candidates_df[
                    (candidates_df["candidate_threshold"] <= global_score_threshold)
                    & (candidates_df["known_precision_on_accepted"] >= min_allowed_precision)
                ].copy()
                if use_unknown_safety:
                    safe = safe[
                        safe["added_unknown_accept_count"] <= max_added_unknown_accepts
                    ].copy()

                if len(safe):
                    # A runtime threshold for this label can only recover known
                    # rows correctly predicted as this label. Never lower merely
                    # to accept additional wrong-label known examples.
                    improving = safe[safe["recovered_correct_count"] > 0].copy()
                    if len(improving):
                        selected = improving.sort_values(
                            by=[
                                "recovered_correct_count",
                                "added_unknown_accept_count",
                                "added_known_error_count",
                                "meets_target_accept_rate",
                                "known_precision_on_accepted",
                                "candidate_threshold",
                            ],
                            ascending=[False, True, True, False, False, False],
                        ).iloc[0].copy()
                        if (
                            global_metrics["known_accept_rate"] < target_accept_rate
                            and selected["known_accept_rate"] > global_metrics["known_accept_rate"]
                        ):
                            selected["selection_reason"] = "lowered_unknown_safe_recovers_accept_rate"
                        else:
                            selected["selection_reason"] = (
                                "lowered_unknown_safe_improves_accuracy_on_all"
                            )
                    elif use_unknown_safety:
                        precision_safe = candidates_df[
                            (candidates_df["candidate_threshold"] <= global_score_threshold)
                            & (candidates_df["known_precision_on_accepted"] >= min_allowed_precision)
                            & (candidates_df["recovered_correct_count"] > 0)
                        ]
                        if len(precision_safe):
                            selected["selection_reason"] = "keep_global_unknown_safety"

        # Retain the target in diagnostics; unknown-aware lowering also requires
        # a correct recovery rather than acceptance alone.
        selected["target_accept_rate"] = target_accept_rate
        selected["target_class_unknown_false_accept_rate"] = (
            target_class_unknown_false_accept_rate
        )
        selected["max_known_correct_accept_rate_drop"] = max_known_correct_accept_rate_drop
        rows.append(selected)

    # Candidate rows inherit their grid index. Reset it so aggregate-budget
    # rollback always addresses exactly one class row.
    out_df = pd.DataFrame(rows).reset_index(drop=True)
    out_df["pre_budget_selected_threshold"] = out_df["candidate_threshold"]
    out_df["rolled_back_for_global_unknown_budget"] = False

    # Per-class safety limits can accumulate across labels. Enforce one final
    # deployment-level budget and roll back the costliest lowerings first.
    global_unknown_false_accept_count = int(
        (negative_score_df["best_score"] >= global_score_threshold).sum()
    )
    unknown_count = len(negative_score_df)
    allowed_extra_unknown_accepts = int(
        np.floor(max_unknown_false_accept_rate_increase * unknown_count + 1e-12)
    )
    allowed_unknown_false_accept_count = min(
        unknown_count,
        global_unknown_false_accept_count + allowed_extra_unknown_accepts,
    )

    def dynamic_unknown_false_accept_count() -> int:
        class_thresholds = dict(zip(out_df["label"], out_df["candidate_threshold"]))
        effective_thresholds = negative_score_df["best_label"].map(class_thresholds)
        effective_thresholds = effective_thresholds.fillna(global_score_threshold)
        return int((negative_score_df["best_score"] >= effective_thresholds).sum())

    current_unknown_false_accept_count = dynamic_unknown_false_accept_count()
    while use_unknown_safety and current_unknown_false_accept_count > allowed_unknown_false_accept_count:
        rollback_candidates = out_df[
            (out_df["offset"] < 0) & (out_df["added_unknown_accept_count"] > 0)
        ].copy()
        if not len(rollback_candidates):
            break

        rollback_candidates["unknown_cost_per_recovered_correct"] = (
            rollback_candidates["added_unknown_accept_count"]
            / rollback_candidates["recovered_correct_count"].clip(lower=1)
        )
        rollback_idx = rollback_candidates.sort_values(
            by=[
                "unknown_cost_per_recovered_correct",
                "added_unknown_accept_count",
                "recovered_correct_count",
                "label",
            ],
            ascending=[False, False, True, True],
        ).index[0]
        label = out_df.at[rollback_idx, "label"]
        previous_threshold = out_df.at[rollback_idx, "pre_budget_selected_threshold"]
        replacement = global_candidate_rows[label]
        for col in replacement.index:
            out_df.at[rollback_idx, col] = replacement[col]
        out_df.at[rollback_idx, "selection_reason"] = "rollback_global_unknown_budget"
        out_df.at[rollback_idx, "pre_budget_selected_threshold"] = previous_threshold
        out_df.at[rollback_idx, "rolled_back_for_global_unknown_budget"] = True
        current_unknown_false_accept_count = dynamic_unknown_false_accept_count()

    out_df["calibration_global_unknown_false_accept_count"] = global_unknown_false_accept_count
    out_df["calibration_dynamic_unknown_false_accept_count"] = current_unknown_false_accept_count
    out_df["calibration_allowed_unknown_false_accept_count"] = allowed_unknown_false_accept_count
    out_df["calibration_unknown_count"] = unknown_count
    out_df["calibration_global_unknown_false_accept_rate"] = (
        global_unknown_false_accept_count / unknown_count if unknown_count else 0.0
    )
    out_df["calibration_dynamic_unknown_false_accept_rate"] = (
        current_unknown_false_accept_count / unknown_count if unknown_count else 0.0
    )
    out_df["min_class_unknown_samples"] = min_class_unknown_samples
    out_df["max_added_unknown_accepts"] = max_added_unknown_accepts
    out_df["max_unknown_false_accept_rate_increase"] = max_unknown_false_accept_rate_increase
    out_df["max_raise_steps"] = max_raise_steps
    out_df["target_class_unknown_false_accept_rate"] = (
        target_class_unknown_false_accept_rate
    )
    out_df["max_known_correct_accept_rate_drop"] = max_known_correct_accept_rate_drop
    out_df = out_df.rename(columns={"candidate_threshold": "selected_threshold"})
    ordered_cols = [
        "label",
        "known_count",
        "global_threshold",
        "selected_threshold",
        "offset",
        "selection_reason",
        "global_known_accept_rate",
        "known_accept_rate",
        "global_known_accuracy_on_all",
        "known_accuracy_on_all",
        "global_known_accuracy_on_accepted",
        "known_accuracy_on_accepted",
        "global_known_precision_on_accepted",
        "known_precision_on_accepted",
        "accepted_count",
        "correct_accepted_count",
        "recovered_correct_count",
        "lost_correct_count",
        "global_known_error_accept_count",
        "known_error_accept_count",
        "added_known_error_count",
        "removed_known_error_count",
        "class_unknown_count",
        "global_unknown_false_accept_count",
        "unknown_false_accept_count",
        "added_unknown_accept_count",
        "removed_unknown_false_accept_count",
        "global_class_unknown_false_accept_rate",
        "class_unknown_false_accept_rate",
        "pre_budget_selected_threshold",
        "rolled_back_for_global_unknown_budget",
        "calibration_global_unknown_false_accept_count",
        "calibration_dynamic_unknown_false_accept_count",
        "calibration_allowed_unknown_false_accept_count",
        "calibration_unknown_count",
        "calibration_global_unknown_false_accept_rate",
        "calibration_dynamic_unknown_false_accept_rate",
        "min_class_unknown_samples",
        "max_added_unknown_accepts",
        "max_unknown_false_accept_rate_increase",
        "max_raise_steps",
        "target_class_unknown_false_accept_rate",
        "max_known_correct_accept_rate_drop",
    ]
    remaining_cols = [col for col in out_df.columns if col not in ordered_cols]
    return out_df[ordered_cols + remaining_cols].sort_values(["offset", "label"]).reset_index(drop=True)


# Preserve the previous public name for callers written against the lower-only
# implementation. New code should use the bidirectional name above.
build_lower_only_class_thresholds = build_class_thresholds


def build_threshold_table(class_thresholds_df: pd.DataFrame) -> pd.DataFrame:
    """Build the deployable threshold table written to `thresholds.csv`.

    Input:
        class_thresholds_df: Per-class threshold metrics.
    Output:
        DataFrame with only label, global threshold, and selected threshold.
    """
    class_rows = class_thresholds_df[["label", "global_threshold", "selected_threshold"]].copy()
    return class_rows.reset_index(drop=True)


def build_threshold_metrics_table(
    class_thresholds_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the extended per-class threshold metrics table.

    Input:
        class_thresholds_df: Per-class threshold metrics.
    Output:
        DataFrame written to `thresholds_w_metrics.csv`.
    """
    preferred_cols = [
        "label",
        "global_threshold",
        "selected_threshold",
        "offset",
        "selection_reason",
        "known_count",
        "global_known_accept_rate",
        "known_accept_rate",
        "global_known_accuracy_on_all",
        "known_accuracy_on_all",
        "global_known_accuracy_on_accepted",
        "known_accuracy_on_accepted",
        "accepted_count",
        "correct_accepted_count",
    ]
    ordered_cols = [col for col in preferred_cols if col in class_thresholds_df.columns]
    remaining_cols = [col for col in class_thresholds_df.columns if col not in ordered_cols]
    return class_thresholds_df[ordered_cols + remaining_cols].copy()


def optimize_thresholds_from_cache(
    cache_dir: str,
    cache_name: str,
    output_dir: str,
    model_name: str,
    device: str = "auto",
    batch_size: int = 32,
    seed: int = 42,
    top_k: int = 5,
    unknown_label: str = DEFAULT_UNKNOWN_LABEL,
    calibration_ratio: float = 0.50,
    score_min: float = 0.65,
    score_max: float = 0.80,
    score_step: float = 0.01,
    min_known_accept_rate: float = 0.97,
    dynamic_step: float = 0.01,
    dynamic_max_steps: int = 15,
    dynamic_target_accept_rate: float = 0.98,
    dynamic_min_class_samples: int = 3,
    dynamic_max_accepted_accuracy_drop: float = 0.03,
    dynamic_min_class_unknown_samples: int = 1,
    dynamic_max_added_unknown_accepts: int = 0,
    dynamic_max_unknown_false_accept_rate_increase: float = 0.005,
    dynamic_max_raise_steps: int = 15,
    dynamic_target_class_unknown_false_accept_rate: float = 0.01,
    dynamic_max_known_correct_accept_rate_drop: float = 0.03,
) -> dict:
    """Optimize thresholds from a saved memory-bank cache.

    Input:
        cache_dir: Folder containing the memory-bank cache.
        cache_name: Cache filename or filename without `.pkl`.
        output_dir: Folder where threshold artifacts are written.
        model_name: Embedding model id or local model path.
        device: `auto`, `cpu`, or `cuda`.
        batch_size: Number of images embedded per model call when needed.
        seed: Random seed for calibration/evaluation splits.
        top_k: Number of memory neighbors used for scoring.
        unknown_label: Output label for rejected predictions.
        calibration_ratio: Fraction of test data used for threshold selection.
        score_min: Minimum threshold candidate.
        score_max: Maximum threshold candidate.
        score_step: Candidate threshold increment.
        min_known_accept_rate: Minimum known acceptance for global selection.
        dynamic_step: Per-class threshold adjustment size.
        dynamic_max_steps: Maximum downward per-class steps.
        dynamic_target_accept_rate: Per-class accept-rate recovery target.
        dynamic_min_class_samples: Minimum class samples for dynamic tuning.
        dynamic_max_accepted_accuracy_drop: Allowed predicted-class precision drop.
        dynamic_min_class_unknown_samples: Minimum unknowns attracted to a class.
        dynamic_max_added_unknown_accepts: Allowed extra unknown accepts per class.
        dynamic_max_unknown_false_accept_rate_increase: Aggregate unknown-rate budget.
        dynamic_max_raise_steps: Maximum upward per-class steps.
        dynamic_target_class_unknown_false_accept_rate: Target attracted-unknown rate.
        dynamic_max_known_correct_accept_rate_drop: Allowed correct-known rate loss.
    Output:
        Dictionary containing output path, summaries, and metadata.
    """
    np.random.seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # The cache provides the memory bank and fixed evaluation shelves. Loading
    # from the cache keeps threshold optimization reproducible.
    payload = load_memory_bank_cache(cache_dir, cache_name)
    memory_items = payload["memory_items"]
    known_test_items = payload["known_test_items"]
    negative_test_items = payload["negative_test_items"]

    if not known_test_items:
        raise ValueError("Memory-bank cache does not contain known_test_items.")
    if not negative_test_items:
        raise ValueError("Memory-bank cache does not contain negative_test_items.")

    resolved_device = resolve_device(device)
    model = None
    # Memory embeddings should usually be cached from create_memory_bank. Known
    # and negative split embeddings are extracted lazily only if the cache does
    # not already contain them.
    memory_embeddings, memory_labels, memory_paths, model = get_or_extract_split_embeddings(
        payload, "memory", memory_items, model, model_name, resolved_device, batch_size
    )
    known_embeddings, known_labels, known_paths, model = get_or_extract_split_embeddings(
        payload, "known_test", known_test_items, model, model_name, resolved_device, batch_size
    )
    negative_embeddings, negative_labels, negative_paths, model = get_or_extract_split_embeddings(
        payload, "negative_test", negative_test_items, model, model_name, resolved_device, batch_size
    )

    index = build_faiss_index(memory_embeddings)
    # Split the cached known/negative test data again into threshold-selection
    # calibration data and holdout evaluation data.
    (
        known_calib_embeddings,
        known_calib_labels,
        known_calib_paths,
        known_eval_embeddings,
        known_eval_labels,
        known_eval_paths,
        known_split_report,
    ) = split_calibration_evaluation(known_embeddings, known_labels, known_paths, calibration_ratio, seed, True)
    (
        negative_calib_embeddings,
        negative_calib_labels,
        negative_calib_paths,
        negative_eval_embeddings,
        negative_eval_labels,
        negative_eval_paths,
        negative_split_report,
    ) = split_calibration_evaluation(negative_embeddings, negative_labels, negative_paths, calibration_ratio, seed, False)

    known_calib_score_df = build_score_table(
        known_calib_embeddings, known_calib_labels, known_calib_paths, index, memory_labels, memory_paths, top_k
    )
    negative_calib_score_df = build_score_table(
        negative_calib_embeddings, negative_calib_labels, negative_calib_paths, index, memory_labels, memory_paths, top_k
    )
    known_eval_score_df = build_score_table(
        known_eval_embeddings, known_eval_labels, known_eval_paths, index, memory_labels, memory_paths, top_k
    )
    negative_eval_score_df = build_score_table(
        negative_eval_embeddings, negative_eval_labels, negative_eval_paths, index, memory_labels, memory_paths, top_k
    )

    # Select the global threshold from calibration scores, then raise risky
    # classes or lower safely recoverable classes using the same calibration data.
    score_values = inclusive_grid(score_min, score_max, score_step)
    global_sweep_df = evaluate_global_threshold_grid(
        known_calib_score_df, negative_calib_score_df, score_values, unknown_label
    )
    best_global = choose_best_global_threshold(global_sweep_df, min_known_accept_rate)
    global_score_threshold = float(best_global["score_threshold"])

    class_thresholds_df = build_class_thresholds(
        known_score_df=known_calib_score_df,
        global_score_threshold=global_score_threshold,
        step=dynamic_step,
        max_steps=dynamic_max_steps,
        target_accept_rate=dynamic_target_accept_rate,
        min_class_samples=dynamic_min_class_samples,
        max_accepted_accuracy_drop=dynamic_max_accepted_accuracy_drop,
        negative_score_df=negative_calib_score_df,
        min_class_unknown_samples=dynamic_min_class_unknown_samples,
        max_added_unknown_accepts=dynamic_max_added_unknown_accepts,
        max_unknown_false_accept_rate_increase=dynamic_max_unknown_false_accept_rate_increase,
        max_raise_steps=dynamic_max_raise_steps,
        target_class_unknown_false_accept_rate=(
            dynamic_target_class_unknown_false_accept_rate
        ),
        max_known_correct_accept_rate_drop=dynamic_max_known_correct_accept_rate_drop,
    )
    class_score_thresholds = dict(zip(class_thresholds_df["label"], class_thresholds_df["selected_threshold"]))

    # Evaluate both policies on the holdout split. The global policy is useful as
    # a baseline; the bidirectional class-specific policy is deployable.
    global_known_eval_df = apply_global_threshold(known_eval_score_df, global_score_threshold, unknown_label)
    global_negative_eval_df = apply_global_threshold(negative_eval_score_df, global_score_threshold, unknown_label)
    dynamic_known_eval_df = apply_dynamic_threshold(
        known_eval_score_df, global_score_threshold, class_score_thresholds, unknown_label
    )
    dynamic_negative_eval_df = apply_dynamic_threshold(
        negative_eval_score_df, global_score_threshold, class_score_thresholds, unknown_label
    )

    global_eval_summary = {
        "method": "global",
        "score_threshold": global_score_threshold,
        **summarize_open_set_from_thresholded(global_known_eval_df, global_negative_eval_df),
    }
    dynamic_eval_summary = {
        "method": "dynamic_class_specific",
        "global_score_threshold": global_score_threshold,
        "num_class_thresholds": len(class_thresholds_df),
        "num_classes_lowered": int((class_thresholds_df["offset"] < 0).sum()),
        "num_classes_raised": int((class_thresholds_df["offset"] > 0).sum()),
        **summarize_open_set_from_thresholded(dynamic_known_eval_df, dynamic_negative_eval_df),
    }

    metadata = {
        "cache_dir": cache_dir,
        "cache_name": cache_name,
        "cache_path": str(payload["cache_path"]),
        "output_dir": str(output_path),
        "threshold_file": THRESHOLD_FILE,
        "metrics_file": METRICS_FILE,
        "model_name": model_name,
        "device": resolved_device,
        "unknown_label": unknown_label,
        "top_k": top_k,
        "seed": seed,
        "calibration_ratio": calibration_ratio,
        "memory_count": len(memory_items),
        "known_test_count": len(known_test_items),
        "negative_test_count": len(negative_test_items),
        "memory_embedding_shape": list(memory_embeddings.shape),
        "known_embedding_shape": list(known_embeddings.shape),
        "negative_embedding_shape": list(negative_embeddings.shape),
        "known_split": known_split_report,
        "negative_split": negative_split_report,
        "score_values": score_values.tolist(),
        "dynamic_min_class_unknown_samples": dynamic_min_class_unknown_samples,
        "dynamic_max_added_unknown_accepts": dynamic_max_added_unknown_accepts,
        "dynamic_max_unknown_false_accept_rate_increase": (
            dynamic_max_unknown_false_accept_rate_increase
        ),
        "dynamic_max_raise_steps": dynamic_max_raise_steps,
        "dynamic_target_class_unknown_false_accept_rate": (
            dynamic_target_class_unknown_false_accept_rate
        ),
        "dynamic_max_known_correct_accept_rate_drop": (
            dynamic_max_known_correct_accept_rate_drop
        ),
        "global_eval_summary": global_eval_summary,
        "dynamic_eval_summary": dynamic_eval_summary,
        "cache_metadata": payload.get("metadata", {}),
    }

    # Keep deployment and analysis separate: `thresholds.csv` contains only the
    # required threshold columns, while `thresholds_w_metrics.csv` adds
    # class-level diagnostics.
    threshold_df = build_threshold_table(class_thresholds_df)
    metrics_df = build_threshold_metrics_table(class_thresholds_df)

    threshold_df.to_csv(output_path / THRESHOLD_FILE, index=False)
    metrics_df.to_csv(output_path / METRICS_FILE, index=False)

    return {
        "output_dir": str(output_path),
        "threshold_file": str(output_path / THRESHOLD_FILE),
        "metrics_file": str(output_path / METRICS_FILE),
        "global_eval_summary": global_eval_summary,
        "dynamic_eval_summary": dynamic_eval_summary,
        "metadata": metadata,
    }
