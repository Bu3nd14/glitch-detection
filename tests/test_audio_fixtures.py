import hashlib
import json
import sys
import unittest
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.generate_audio_fixtures import generate

ASSET_DIR = ROOT / "fixtures" / "audio"
SAMPLE_RATE = 48_000
FRAME_COUNT = 480_000
CLICK = (108_000, 108_096)
DROPOUT = (204_000, 218_400)
DROPOUT_ZERO = (204_240, 218_160)
STUTTER_SOURCE = (293_760, 297_600)
STUTTER = (297_600, 320_640)
CLIPPING = (393_600, 422_400)
CANONICAL_SHA256 = {
    "clean": "febd52acbb9f96dd8fe4d4eee34107d87349e52091b3ead0229c37d22ec95ec9",
    "corrupted": "2ae990d463d71ee800e50d4bd15952c0b5293701f41ea0640330c42d829924b9",
}


def read_frames(path: Path) -> list[tuple[int, int]]:
    with wave.open(str(path), "rb") as source:
        raw = source.readframes(source.getnframes())
    return [tuple(int.from_bytes(raw[index + channel * 2:index + channel * 2 + 2], "little", signed=True) for channel in range(2)) for index in range(0, len(raw), 4)]


class AudioFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.clean_path = ASSET_DIR / "poc_clean.wav"
        cls.corrupted_path = ASSET_DIR / "poc_corrupted.wav"
        cls.clean = read_frames(cls.clean_path)
        cls.corrupted = read_frames(cls.corrupted_path)
        cls.manifest = json.loads((ASSET_DIR / "poc_ground_truth.json").read_text())

    def test_wav_format_and_frame_count(self) -> None:
        for path in (self.clean_path, self.corrupted_path):
            with wave.open(str(path), "rb") as source:
                self.assertEqual((source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getnframes()), (2, 2, SAMPLE_RATE, FRAME_COUNT))

    def test_peak_limits(self) -> None:
        clean_peak = max(abs(value) for frame in self.clean for value in frame)
        corrupted_peak = max(abs(value) for frame in self.corrupted for value in frame)
        self.assertAlmostEqual(clean_peak, 9830, delta=1)
        self.assertLessEqual(corrupted_peak, 16384)

    def test_unchanged_outside_fault_union(self) -> None:
        faults = (CLICK, DROPOUT, STUTTER, CLIPPING)
        for index, (clean, corrupted) in enumerate(zip(self.clean, self.corrupted)):
            if not any(start <= index < end for start, end in faults):
                self.assertEqual(corrupted, clean, f"unexpected difference at frame {index}")

    def test_dropout_core_is_zero(self) -> None:
        self.assertTrue(all(frame == (0, 0) for frame in self.corrupted[DROPOUT_ZERO[0]:DROPOUT_ZERO[1]]))

    def test_stutter_repetitions_are_clean_source_block(self) -> None:
        source = self.clean[STUTTER_SOURCE[0]:STUTTER_SOURCE[1]]
        for repetition in range(6):
            start = STUTTER[0] + repetition * len(source)
            self.assertEqual(self.corrupted[start:start + len(source)], source)

    def test_clipping_has_plateaux(self) -> None:
        clipped = self.corrupted[CLIPPING[0]:CLIPPING[1]]
        for channel in (0, 1):
            for plateau_value in (-14745, 14745):
                plateau = sum(frame[channel] == plateau_value for frame in clipped)
                self.assertGreaterEqual(plateau / len(clipped), 0.10)

    def test_manifest_intervals_and_hashes_are_coherent(self) -> None:
        self.assertEqual(self.manifest["schema_version"], 1)
        self.assertEqual(
            self.manifest["format"],
            {
                "container": "WAV",
                "codec": "PCM s16le",
                "sample_rate_hz": 48_000,
                "channels": 2,
                "frames": 480_000,
                "duration_seconds": 10.0,
            },
        )
        self.assertEqual(list(self.manifest["files"]), ["clean", "corrupted"])
        for name, path in (("clean", self.clean_path), ("corrupted", self.corrupted_path)):
            file_record = self.manifest["files"][name]
            self.assertEqual(file_record["path"], path.name)
            self.assertEqual(file_record["sha256"], CANONICAL_SHA256[name])
            self.assertEqual(file_record["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())

        faults = self.manifest["faults"]
        self.assertEqual([fault["class"] for fault in faults], ["click", "dropout", "stutter", "clipping"])
        expected_intervals = [CLICK, DROPOUT, STUTTER, CLIPPING]
        for fault, (start, end) in zip(faults, expected_intervals):
            interval = fault["interval"]
            self.assertEqual(interval, {"unit": "frames", "semantics": "half-open", "start": start, "end": end})
            self.assertEqual(fault["seconds"], {"start": start / 48_000, "end": end / 48_000})

        self.assertEqual(faults[0]["replacement"], "both channels")
        self.assertEqual(faults[0]["formula"], "0.38*exp(-k/10)*cos(pi*k)+0.12*exp(-k/36)*cos(2*pi*180*k/48000)")
        self.assertEqual(faults[1]["fade_out_frames"], 240)
        self.assertEqual(faults[1]["zero_interval"], {"start": DROPOUT_ZERO[0], "end": DROPOUT_ZERO[1]})
        self.assertEqual(faults[1]["fade_in_frames"], 240)
        self.assertEqual(faults[2]["source_interval"], {"start": STUTTER_SOURCE[0], "end": STUTTER_SOURCE[1]})
        self.assertEqual(faults[2]["repetitions"], 6)
        self.assertEqual(faults[3]["transform"], "clip(6*x_clean, -0.45, +0.45)")

    def test_regeneration_is_bit_identical(self) -> None:
        temporary = ROOT / ".work" / "fixture-regeneration"
        temporary.mkdir(parents=True, exist_ok=True)
        try:
            regenerated = generate(temporary)
            for name, canonical in (("clean", self.clean_path), ("corrupted", self.corrupted_path), ("manifest", ASSET_DIR / "poc_ground_truth.json")):
                self.assertEqual(regenerated[name].read_bytes(), canonical.read_bytes())
        finally:
            for path in temporary.iterdir():
                path.unlink()
            temporary.rmdir()


if __name__ == "__main__":
    unittest.main()
