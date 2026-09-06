import hashlib
import json
import sys
import unittest
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.generate_audio_fixtures import (
    _riff_data_chunk,
    generate,
    generate_harvard,
    generate_rock,
)

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
ROCK_FRAME_COUNT = 768_000
ROCK_FAULTS = ((111_216, 111_264), (255_840, 267_360), (437_760, 460_800), (584_640, 610_560))
ROCK_ZERO = (256_080, 267_120)
ROCK_SOURCE = (430_080, 433_920)
ROCK_SHA256 = {
    "clean": "d4a590e25065817959b33024e5d35055c0e84311d746b564a4cea23724c97f43",
    "corrupted": "c94d7a90c241d54b2e0beb620a5b99090d8473d848348b016fd73331475fed5e",
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


class RockAudioFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.clean_path = ASSET_DIR / "poc_rock_v1_clean.wav"
        cls.corrupted_path = ASSET_DIR / "poc_rock_v1_corrupted.wav"
        cls.manifest_path = ASSET_DIR / "poc_rock_v1_ground_truth.json"
        cls.manifest = json.loads(cls.manifest_path.read_text())
        with wave.open(str(cls.clean_path), "rb") as source:
            cls.clean = np.frombuffer(source.readframes(ROCK_FRAME_COUNT), dtype="<i2").reshape(-1, 2)
        with wave.open(str(cls.corrupted_path), "rb") as source:
            cls.corrupted = np.frombuffer(source.readframes(ROCK_FRAME_COUNT), dtype="<i2").reshape(-1, 2)

    def test_format_duration_hash_and_manifest_schema(self) -> None:
        for name, path in (("clean", self.clean_path), ("corrupted", self.corrupted_path)):
            with wave.open(str(path), "rb") as source:
                self.assertEqual((source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getnframes()),
                                 (2, 2, SAMPLE_RATE, ROCK_FRAME_COUNT))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), ROCK_SHA256[name])
            self.assertEqual(self.manifest["files"][name]["sha256"], ROCK_SHA256[name])
        self.assertEqual(self.manifest["schema_version"], 2)
        self.assertEqual([item["class"] for item in self.manifest["faults"]], ["click", "dropout", "stutter", "clipping"])
        self.assertEqual([item["interval"]["start"] for item in self.manifest["faults"]], [item[0] for item in ROCK_FAULTS])
        self.assertIn("identifiability_limits", self.manifest)
        self.assertIn("clean_controls", self.manifest)
        dropout = next(item for item in self.manifest["faults"] if item["class"] == "dropout")
        self.assertEqual(dropout["expected_status"], "uncertain")
        self.assertEqual(dropout["allowed_statuses"], ["uncertain"])
        self.assertIn("not reliably identifiable", " ".join(self.manifest["identifiability_limits"]))
        hard_mute = next(item for item in self.manifest["clean_controls"] if item["name"] == "hard_mute")
        self.assertEqual(hard_mute["expected_status"], "uncertain")

    def test_corruption_is_identical_outside_faults_and_faults_are_exact(self) -> None:
        unchanged = np.ones(ROCK_FRAME_COUNT, dtype=bool)
        for start, end in ROCK_FAULTS:
            unchanged[start:end] = False
        self.assertTrue(np.array_equal(self.clean[unchanged], self.corrupted[unchanged]))
        self.assertTrue(np.array_equal(self.corrupted[ROCK_ZERO[0]:ROCK_ZERO[1]], np.zeros((ROCK_ZERO[1] - ROCK_ZERO[0], 2), dtype=np.int16)))
        source = self.clean[ROCK_SOURCE[0]:ROCK_SOURCE[1]]
        for start in range(437_760, 460_800, len(source)):
            self.assertTrue(np.array_equal(self.corrupted[start:start + len(source)], source))
        plateau = round(.46 * 32_767)
        clipped = self.corrupted[584_640:610_560]
        self.assertGreater(np.count_nonzero(clipped == plateau), 100)
        self.assertGreater(np.count_nonzero(clipped == -plateau), 100)

    def test_clean_invariants_and_regeneration_are_deterministic(self) -> None:
        values = self.clean.astype(np.float64) / 32_767
        self.assertLessEqual(np.max(np.abs(self.clean)), self.manifest["clean_invariants"]["peak_s16"])
        self.assertLessEqual(np.max(np.abs(np.diff(values, axis=0))), self.manifest["clean_invariants"]["max_derivative_float"])
        correlation = float(np.corrcoef(self.clean[:-24_000, 0], self.clean[24_000:, 0])[0, 1])
        self.assertAlmostEqual(correlation, self.manifest["clean_invariants"]["correlation_lag_500ms_left"])
        temporary = ROOT / ".work" / "rock-fixture-regeneration"
        temporary.mkdir(parents=True, exist_ok=True)
        try:
            regenerated = generate_rock(temporary)
            for name, canonical in (("clean", self.clean_path), ("corrupted", self.corrupted_path), ("manifest", self.manifest_path)):
                self.assertEqual(regenerated[name].read_bytes(), canonical.read_bytes())
        finally:
            for path in temporary.iterdir():
                path.unlink()
            temporary.rmdir()


class HarvardAudioFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = ASSET_DIR / "harvard.wav"
        cls.clean = ASSET_DIR / "poc_harvard_v1_clean.wav"
        cls.corrupted = ASSET_DIR / "poc_harvard_v1_corrupted.wav"
        cls.manifest_path = ASSET_DIR / "poc_harvard_v1_ground_truth.json"
        cls.manifest = json.loads(cls.manifest_path.read_text())
        cls.source_bytes = cls.source.read_bytes()
        cls.corrupted_bytes = cls.corrupted.read_bytes()
        with wave.open(str(cls.source), "rb") as source:
            cls.source_pcm = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").reshape(-1, 2)
        with wave.open(str(cls.corrupted), "rb") as source:
            cls.corrupted_pcm = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").reshape(-1, 2)

    def test_source_clean_riff_and_manifest_are_preserved(self) -> None:
        self.assertEqual(hashlib.sha256(self.source_bytes).hexdigest(), "971b4163670445c415c6b0fb6813c38093409ecac2f6b4d429ae3574d24ad470")
        self.assertEqual(self.clean.read_bytes(), self.source_bytes)
        data_offset, data_size = _riff_data_chunk(self.source_bytes)
        self.assertEqual(self.source_bytes[:data_offset], self.corrupted_bytes[:data_offset])
        self.assertEqual(self.source_bytes[data_offset + data_size:], self.corrupted_bytes[data_offset + data_size:])
        self.assertEqual(self.manifest["schema_version"], 3)
        self.assertEqual(self.manifest["source"]["sha256"], hashlib.sha256(self.source_bytes).hexdigest())
        self.assertEqual([item["name"] for item in self.manifest["controls"]],
                         ["plosive", "sibilant", "long_pause", "ambiguous_natural_pause", "repeated_word"])
        self.assertEqual(self.manifest["controls"][3]["expected_status"], "uncertain")
        self.assertEqual([(item["class"], item["status"]) for item in self.manifest["baseline_events"]],
                         [("clipping", "detected"), ("clipping", "detected"), ("dropout", "uncertain")])
        with wave.open(str(self.corrupted), "rb") as source:
            self.assertEqual((source.getframerate(), source.getnframes(), source.getnchannels(), source.getsampwidth()), (44_100, 809_508, 2, 2))

    def test_harvard_pcm_patches_are_exact_and_regenerate(self) -> None:
        faults = [(132_300, 132_344), (88_200, 98_784), (506_268, 527_436), (593_145, 610_785)]
        unchanged = np.ones(len(self.source_pcm), dtype=bool)
        for start, end in faults: unchanged[start:end] = False
        self.assertTrue(np.array_equal(self.source_pcm[unchanged], self.corrupted_pcm[unchanged]))
        self.assertTrue(np.array_equal(self.corrupted_pcm[88_421:98_563], np.zeros((10_142, 2), dtype=np.int16)))
        source = self.source_pcm[502_740:506_268]
        for start in range(506_268, 527_436, len(source)):
            self.assertTrue(np.array_equal(self.corrupted_pcm[start:start + len(source)], source))
        self.assertGreater(np.count_nonzero(self.corrupted_pcm[593_145:610_785] == round(.46 * 32_767)), 100)
        temporary = ROOT / ".work" / "harvard-fixture-regeneration"
        temporary.mkdir(parents=True, exist_ok=True)
        try:
            regenerated = generate_harvard(temporary)
            for name, canonical in (("clean", self.clean), ("corrupted", self.corrupted), ("manifest", self.manifest_path)):
                self.assertEqual(regenerated[name].read_bytes(), canonical.read_bytes())
        finally:
            for path in temporary.iterdir(): path.unlink()
            temporary.rmdir()


if __name__ == "__main__":
    unittest.main()
