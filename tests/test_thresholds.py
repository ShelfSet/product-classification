"""Focused tests for unknown-aware class-specific threshold selection."""

from __future__ import annotations

import unittest

import pandas as pd

from product_recognition.thresholds import build_class_thresholds


class UnknownAwareClassThresholdTests(unittest.TestCase):
    """Verify known recovery, predicted-class safety, and global budgeting."""

    @staticmethod
    def build_thresholds(
        known_rows: list[dict],
        unknown_rows: list[dict],
        **overrides,
    ) -> pd.DataFrame:
        """Build thresholds with a compact deterministic candidate grid."""
        kwargs = {
            "known_score_df": pd.DataFrame(known_rows),
            "negative_score_df": pd.DataFrame(unknown_rows),
            "global_score_threshold": 0.71,
            "step": 0.01,
            "max_steps": 2,
            "target_accept_rate": 0.98,
            "min_class_samples": 3,
            "max_accepted_accuracy_drop": 0.03,
            "min_class_unknown_samples": 1,
            "max_added_unknown_accepts": 0,
            "max_unknown_false_accept_rate_increase": 0.005,
        }
        kwargs.update(overrides)
        return build_class_thresholds(**kwargs)

    def test_lowers_when_correct_known_is_recovered_without_unknown_cost(self):
        """Lower to the strictest threshold that safely recovers a known row."""
        known_rows = [
            {"true_label": "A", "best_label": "A", "best_score": 0.70},
            {"true_label": "A", "best_label": "A", "best_score": 0.75},
            {"true_label": "A", "best_label": "A", "best_score": 0.80},
        ]
        unknown_rows = [
            {"true_label": "__Unknown__", "best_label": "A", "best_score": 0.65},
            {"true_label": "__Unknown__", "best_label": "A", "best_score": 0.68},
        ]

        result = self.build_thresholds(known_rows, unknown_rows)
        row = result.iloc[0]

        self.assertAlmostEqual(row["selected_threshold"], 0.70)
        self.assertEqual(row["recovered_correct_count"], 1)
        self.assertEqual(row["added_unknown_accept_count"], 0)
        self.assertEqual(row["selection_reason"], "lowered_unknown_safe_recovers_accept_rate")

    def test_keeps_global_when_lowering_would_accept_an_unknown(self):
        """Reject a known-recovery candidate that adds an unknown false accept."""
        known_rows = [
            {"true_label": "A", "best_label": "A", "best_score": 0.70},
            {"true_label": "A", "best_label": "A", "best_score": 0.75},
            {"true_label": "A", "best_label": "A", "best_score": 0.80},
        ]
        unknown_rows = [
            {"true_label": "__Unknown__", "best_label": "A", "best_score": 0.705},
        ]

        result = self.build_thresholds(known_rows, unknown_rows)
        row = result.iloc[0]

        self.assertAlmostEqual(row["selected_threshold"], 0.71)
        self.assertEqual(row["selection_reason"], "keep_global_unknown_safety")

    def test_keeps_global_without_unknown_support_for_predicted_class(self):
        """Treat absent attracted unknowns as missing safety evidence."""
        known_rows = [
            {"true_label": "A", "best_label": "A", "best_score": 0.70},
            {"true_label": "A", "best_label": "A", "best_score": 0.75},
            {"true_label": "A", "best_label": "A", "best_score": 0.80},
        ]
        unknown_rows = [
            {"true_label": "__Unknown__", "best_label": "B", "best_score": 0.60},
        ]

        result = self.build_thresholds(known_rows, unknown_rows)
        row = result.iloc[0]

        self.assertAlmostEqual(row["selected_threshold"], 0.71)
        self.assertEqual(
            row["selection_reason"],
            "fallback_insufficient_unknown_support_keep_global",
        )

    def test_uses_predicted_class_precision_to_block_known_errors(self):
        """Evaluate wrong known predictions under the threshold they will use."""
        known_rows = [
            {"true_label": "A", "best_label": "A", "best_score": 0.70},
            {"true_label": "A", "best_label": "A", "best_score": 0.75},
            {"true_label": "A", "best_label": "A", "best_score": 0.80},
            {"true_label": "B", "best_label": "A", "best_score": 0.70},
        ]
        unknown_rows = [
            {"true_label": "__Unknown__", "best_label": "A", "best_score": 0.60},
        ]

        result = self.build_thresholds(known_rows, unknown_rows)
        row = result[result["label"] == "A"].iloc[0]

        self.assertAlmostEqual(row["selected_threshold"], 0.71)
        self.assertEqual(row["known_precision_on_accepted"], 1.0)

    def test_raises_risky_class_to_reject_attracted_unknown(self):
        """Raise to the smallest threshold that meets the class risk target."""
        known_rows = [
            {"true_label": "A", "best_label": "A", "best_score": 0.74},
            {"true_label": "A", "best_label": "A", "best_score": 0.75},
            {"true_label": "A", "best_label": "A", "best_score": 0.80},
        ]
        unknown_rows = [
            {"true_label": "__Unknown__", "best_label": "A", "best_score": 0.725},
            {"true_label": "__Unknown__", "best_label": "A", "best_score": 0.60},
        ]

        result = self.build_thresholds(known_rows, unknown_rows)
        row = result.iloc[0]

        self.assertAlmostEqual(row["selected_threshold"], 0.73)
        self.assertEqual(row["removed_unknown_false_accept_count"], 1)
        self.assertEqual(row["lost_correct_count"], 0)
        self.assertEqual(row["selection_reason"], "raised_meets_unknown_false_accept_target")

    def test_takes_best_safe_raise_when_unknown_target_is_infeasible(self):
        """Reduce unknown accepts even when the requested target cannot be met."""
        known_rows = [
            {"true_label": "A", "best_label": "A", "best_score": 0.74},
            {"true_label": "A", "best_label": "A", "best_score": 0.80},
            {"true_label": "A", "best_label": "A", "best_score": 0.80},
        ]
        unknown_rows = [
            {"true_label": "__Unknown__", "best_label": "A", "best_score": 0.725},
            {"true_label": "__Unknown__", "best_label": "A", "best_score": 0.75},
        ]

        result = self.build_thresholds(known_rows, unknown_rows)
        row = result.iloc[0]

        self.assertAlmostEqual(row["selected_threshold"], 0.73)
        self.assertEqual(row["unknown_false_accept_count"], 1)
        self.assertEqual(row["selection_reason"], "raised_reduces_unknown_false_accepts")

    def test_keeps_global_when_raise_exceeds_known_recall_drop(self):
        """Do not reduce unknown risk by sacrificing too many correct knowns."""
        known_rows = [
            {"true_label": "A", "best_label": "A", "best_score": 0.72},
            {"true_label": "A", "best_label": "A", "best_score": 0.72},
            {"true_label": "A", "best_label": "A", "best_score": 0.80},
        ]
        unknown_rows = [
            {"true_label": "__Unknown__", "best_label": "A", "best_score": 0.75},
        ]

        result = self.build_thresholds(known_rows, unknown_rows)
        row = result.iloc[0]

        self.assertAlmostEqual(row["selected_threshold"], 0.71)
        self.assertEqual(row["selection_reason"], "keep_global_unresolved_unknown_risk")

    def test_rolls_back_lowerings_that_exceed_global_unknown_budget(self):
        """Limit the aggregate cost when individually allowed costs accumulate."""
        known_rows = [
            {"true_label": "A", "best_label": "A", "best_score": 0.70},
            {"true_label": "A", "best_label": "A", "best_score": 0.75},
            {"true_label": "A", "best_label": "A", "best_score": 0.80},
            {"true_label": "B", "best_label": "B", "best_score": 0.70},
            {"true_label": "B", "best_label": "B", "best_score": 0.75},
            {"true_label": "B", "best_label": "B", "best_score": 0.80},
        ]
        unknown_rows = [
            {"true_label": "__Unknown__", "best_label": "A", "best_score": 0.705},
            {"true_label": "__Unknown__", "best_label": "B", "best_score": 0.705},
            {"true_label": "__Unknown__", "best_label": "C", "best_score": 0.20},
            {"true_label": "__Unknown__", "best_label": "C", "best_score": 0.10},
        ]

        result = self.build_thresholds(
            known_rows,
            unknown_rows,
            max_added_unknown_accepts=1,
            max_unknown_false_accept_rate_increase=0.25,
        )

        self.assertEqual(int((result["offset"] < 0).sum()), 1)
        self.assertEqual(int(result["rolled_back_for_global_unknown_budget"].sum()), 1)
        self.assertEqual(result["calibration_dynamic_unknown_false_accept_count"].iloc[0], 1)


if __name__ == "__main__":
    unittest.main()
