from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.calibrate_gemma_timeout import EXPECTED_TYPES, fixture_payloads


class CalibrationToolTests(unittest.TestCase):
    def test_fixture_selection_has_exactly_four_closed_canonical_payloads(self) -> None:
        payloads = fixture_payloads()
        self.assertEqual({event.glitch_type for event, _ in payloads}, EXPECTED_TYPES)
        self.assertTrue(all(event.lifecycle == "CLOSED" for event, _ in payloads))
