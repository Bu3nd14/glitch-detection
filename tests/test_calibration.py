from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from glitch_poc.calibration import (
    default_drain_timeout_seconds,
    proposal_for_four_classes,
    proposed_timeout_seconds,
)


class CalibrationTests(unittest.TestCase):
    def test_no_valid_results_has_no_proposal(self) -> None:
        self.assertIsNone(proposed_timeout_seconds([]))

    def test_fewer_than_twenty_uses_maximum_and_safety_margin(self) -> None:
        self.assertEqual(proposed_timeout_seconds([2.1, 3.2, 2.8]), 5)
        self.assertEqual(proposed_timeout_seconds([4.1, 8.1]), 13)

    def test_p95_and_clamps(self) -> None:
        values = [float(value) for value in range(1, 21)]
        self.assertEqual(proposed_timeout_seconds(values), 29)
        self.assertEqual(proposed_timeout_seconds([.01]), 5)
        self.assertEqual(proposed_timeout_seconds([100]), 60)

    def test_drain_budget_scales_only_accepted_pending_requests(self) -> None:
        self.assertEqual(default_drain_timeout_seconds(10, pending_requests=4), 43)
        self.assertEqual(default_drain_timeout_seconds(10, pending_requests=7), 73)
        self.assertEqual(default_drain_timeout_seconds(10, pending_requests=0), 0)

    def test_four_class_policy_requires_all_valid_classes(self) -> None:
        types = frozenset({"click", "dropout", "stutter", "clipping"})
        samples = [{"glitch_type": kind, "response_valid_grounded": True, "elapsed_s": 10.0} for kind in types]
        self.assertEqual(proposal_for_four_classes(samples, types), 15)
        self.assertIsNone(proposal_for_four_classes(samples[:-1], types))
