from __future__ import annotations

import asyncio
import io
import shutil
import subprocess
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from rich.text import Text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from glitch_poc.cli import resolve_source
from glitch_poc.contracts import DSPEvent, GemmaAnnotation, PCMBlock
from glitch_poc.ingest import FFmpegProducer, ffmpeg_command, pcm_blocks
from glitch_poc.ollama import (
    GemmaQueueStats,
    GemmaWorker,
    OllamaConfig,
    annotation_payload,
    validate_local_ollama_url,
)
from glitch_poc.ring import PCMBlockRing
from glitch_poc.runtime import SliceRuntime, WallClockPacer
from glitch_poc.tui import GlitchTui


def block(sequence: int, start: int, frames: int = 4) -> PCMBlock:
    return PCMBlock.create("test", sequence, start, np.full((frames, 2), sequence, dtype=np.float32))


class TuiSpawnClient:
    """Module-level fake keeps the spawned executor picklable during a Textual pilot run."""
    config = OllamaConfig(queue_capacity=2, timeout_s=.1)

    def annotate(self, item: DSPEvent, values: object) -> GemmaAnnotation:
        return GemmaAnnotation("test", item.event_id, item.revision, "coherent", item.glitch_type, "low", (),
            "TUI spawn mock", True, 0.0, "gemma4:e4b", (), "gemma-grounded-v2", .1)


class PCMTests(unittest.TestCase):
    def test_parser_handles_non_block_aligned_reads_and_frame_accounting(self) -> None:
        raw = np.arange(20, dtype="<f4").tobytes()
        blocks = list(pcm_blocks(io.BytesIO(raw), stream_id="x", block_frames=3))
        self.assertEqual([(item.sequence, item.start_frame, item.frame_count) for item in blocks],
                         [(0, 0, 3), (1, 3, 3), (2, 6, 3), (3, 9, 1)])
        self.assertEqual(blocks[0].samples.tolist(), [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])

    def test_parser_rejects_trailing_partial_frame(self) -> None:
        with self.assertRaisesRegex(ValueError, "truncated PCM stream: 3"):
            list(pcm_blocks(io.BytesIO(b"\x00" * 3), stream_id="x"))

    def test_ring_is_bounded_and_reports_overwrite_discontinuity(self) -> None:
        ring = PCMBlockRing(2, 4)
        ring.push(block(0, 0)); ring.push(block(1, 4)); ring.push(block(2, 8))
        stats = ring.stats()
        self.assertEqual((stats.fill, stats.dropped_blocks, stats.dropped_frames), (2, 1, 4))
        retained = ring.pop()
        assert retained is not None
        self.assertEqual(retained.sequence, 1)
        # The next retained block carries the overwrite accounting to its consumer.
        latest = ring.pop()
        assert latest is not None
        self.assertTrue(latest.discontinuity)
        self.assertEqual(latest.missing_frames, 4)

    def test_pacer_uses_frame_clock_not_a_wall_clock_test_delay(self) -> None:
        now = [10.0]
        delays: list[float] = []
        pacer = WallClockPacer(10, clock=lambda: now[0], sleeper=delays.append)
        pacer.pace(block(0, 0))
        now[0] = 10.05
        pacer.pace(block(1, 2))
        self.assertEqual(len(delays), 1)
        self.assertAlmostEqual(delays[0], 0.15)

    def test_snapshot_has_decimated_waveform_and_metrics(self) -> None:
        ring = PCMBlockRing(2, 8)
        samples = np.column_stack((np.linspace(-1, 1, 8), np.linspace(-1, 1, 8))).astype(np.float32)
        ring.push(PCMBlock.create("x", 0, 0, samples))
        producer = type("Producer", (), {"state": "eof", "error": None})()
        runtime = SliceRuntime(ring, producer, waveform_points=4,
                               pacer=WallClockPacer(48_000, clock=lambda: 0.0, sleeper=lambda _: None))
        self.assertTrue(runtime.step())
        snapshot = runtime.snapshot()
        self.assertEqual((snapshot.position_frames, len(snapshot.waveform)), (8, 4))
        self.assertAlmostEqual(snapshot.peak, 1.0)

    def test_drain_paced_limits_consumer_work_per_ui_snapshot(self) -> None:
        ring = PCMBlockRing(4, 2)
        for sequence in range(3):
            ring.push(block(sequence, sequence * 2, 2))
        producer = type("Producer", (), {"state": "eof", "error": None})()
        runtime = SliceRuntime(ring, producer, waveform_points=2,
            pacer=WallClockPacer(48_000, clock=lambda: 0.0, sleeper=lambda _: None))
        self.assertEqual(runtime.drain_paced(2), 2)
        self.assertEqual(ring.stats().fill, 1)

    def test_background_consumer_advances_without_tui_refresh(self) -> None:
        ring = PCMBlockRing(2, 4)
        ring.push(block(0, 0))
        producer = type("Producer", (), {"state": "eof", "error": None})()
        runtime = SliceRuntime(ring, producer, pacer=WallClockPacer(48_000))
        runtime.start()
        try:
            deadline = time.monotonic() + 0.2
            while runtime.snapshot().position_frames == 0 and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertEqual(runtime.snapshot().position_frames, 4)
        finally:
            runtime.stop()
        snapshot = runtime.snapshot()
        self.assertFalse(runtime.consumer_alive)
        self.assertEqual((snapshot.position_frames, snapshot.ffmpeg_error), (4, None))

    def test_stop_interrupts_distant_pacing_wait_and_joins_consumer(self) -> None:
        ring = PCMBlockRing(3, 4)
        ring.push(block(0, 0))
        ring.push(block(1, 4_800_000))
        producer = type("Producer", (), {"state": "eof", "error": None})()
        runtime = SliceRuntime(ring, producer)
        runtime.start()
        deadline = time.monotonic() + 0.2
        while ring.stats().fill and time.monotonic() < deadline:
            time.sleep(0.001)
        self.assertEqual(ring.stats().fill, 0, "consumer did not reach the distant pacing wait")
        started = time.monotonic()
        runtime.stop()
        self.assertLess(time.monotonic() - started, 0.2)
        self.assertFalse(runtime.consumer_alive)


class IntegrationBoundaryTests(unittest.TestCase):
    def test_ffmpeg_command_is_one_pcm_stdout_decoder(self) -> None:
        command = ffmpeg_command("fixture.mp3")
        self.assertEqual(command.count("ffmpeg"), 1)
        self.assertIn("-re", command)
        self.assertEqual(command[-1], "pipe:1")
        self.assertEqual(command[command.index("-f") + 1], "f32le")

    def test_missing_ffmpeg_degrades_without_starting_a_process(self) -> None:
        ring = PCMBlockRing(2, 8)
        with patch("glitch_poc.ingest.shutil.which", return_value=None):
            producer = FFmpegProducer("missing.wav", ring, stream_id="x")
            self.assertFalse(producer.start())
        self.assertEqual(producer.state, "unavailable")

    def test_producer_discards_stderr_to_prevent_pipe_deadlock(self) -> None:
        ring = PCMBlockRing(2, 8)
        producer = FFmpegProducer("fixture.wav", ring, stream_id="x")
        with patch("glitch_poc.ingest.subprocess.Popen", side_effect=OSError("bad executable")) as popen:
            producer._run()
        self.assertIs(popen.call_args.kwargs["stderr"], subprocess.DEVNULL)
        self.assertEqual(producer.state, "error")

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is not installed")
    def test_ffmpeg_decodes_existing_fixture_to_f32le(self) -> None:
        fixture = ROOT / "fixtures" / "audio" / "poc_clean.wav"
        process = subprocess.run(ffmpeg_command(str(fixture), realtime=False), capture_output=True,
                                 check=True, timeout=15)
        decoded = next(pcm_blocks(io.BytesIO(process.stdout), stream_id="fixture", block_frames=64))
        self.assertEqual((decoded.sample_rate_hz, decoded.channels, decoded.frame_count), (48_000, 2, 64))
        self.assertEqual(decoded.samples.dtype, np.float32)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is not installed")
    def test_ffmpeg_decodes_rock_alias_and_external_path_with_spaces(self) -> None:
        rock = resolve_source(ROOT, "rock-corrupted")
        external_dir = ROOT / ".work" / "ffmpeg external audio"
        external_dir.mkdir(parents=True, exist_ok=True)
        external = external_dir / "song with spaces.wav"
        external.write_bytes((ROOT / "fixtures" / "audio" / "poc_clean.wav").read_bytes())
        try:
            for source in (rock, resolve_source(ROOT, str(external))):
                process = subprocess.run(ffmpeg_command(str(source.path), realtime=False), capture_output=True,
                                         check=True, timeout=20)
                decoded = next(pcm_blocks(io.BytesIO(process.stdout), stream_id=source.stream_id, block_frames=64))
                self.assertEqual(decoded.frame_count, 64)
                self.assertNotIn("song with spaces", decoded.stream_id)
        finally:
            external.unlink()
            external_dir.rmdir()

    def test_ollama_payload_has_no_raw_audio(self) -> None:
        from glitch_poc.contracts import DSPEvent
        event = DSPEvent("event-1", 1, "CLOSED", "uncertain", "loop", 0, 10, 10, 1.0,
                         ("loop",), ("evidence-1",), "poc-d2-v2", 0, False)
        payload = annotation_payload(event, (("evidence-1", {"stream_id": "observed", "start_frame": 0,
            "end_frame": 10, "sample_rate_hz": 48_000, "window_ms": 10,
            "features": {"correlation_lag_500ms": .99},
            "profile_id": "poc-d2-v2"}),))
        rendered = str(payload).lower()
        self.assertNotIn("audio", rendered)
        self.assertNotIn("waveform", rendered)

    def test_ollama_rejects_non_loopback_and_https_endpoints(self) -> None:
        for endpoint in ("https://127.0.0.1:11434/api/generate", "http://example.test/api", "http://[::2]/"):
            with self.assertRaisesRegex(ValueError, "endpoint"):
                validate_local_ollama_url(endpoint)
        for endpoint in ("http://localhost:11434/api/generate", "http://127.0.0.1/", "http://[::1]/"):
            validate_local_ollama_url(endpoint)



class TuiLayoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_layout_is_2x2_at_100x30_and_has_small_terminal_fallback(self) -> None:
        producer = type("Producer", (), {"state": "idle", "error": None})()
        runtime = SliceRuntime(PCMBlockRing(2, 4), producer, source_label="external file a1b2c3d4e5f6")
        app = GlitchTui(runtime)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            self.assertFalse(app.query_one("#too-small").display)
            self.assertTrue(app.query_one("#panels").display)
            self.assertIn("external file a1b2c3d4e5f6", str(app.query_one("#source").content))
        compact = GlitchTui(runtime)
        async with compact.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            compact.action_toggle_gemma_summary()
            await pilot.pause()
            self.assertTrue(compact.query_one("#too-small").display)
            self.assertFalse(compact.query_one("#panels").display)
            self.assertIn("SUMMARY", str(compact.query_one("#gemma-status").content))

    async def test_gemma_detail_shows_waiting_and_disabled_states(self) -> None:
        events = tuple(DSPEvent(f"event-{index}", 1, "CLOSED", "detected", "click", 0, 1, 1, 1.0,
            ("click",), (f"ev-{index}",), "poc-d2-v2", 0, True) for index in range(4))
        annotations = tuple(GemmaAnnotation("v", item.event_id, 1, "coherent", "click", "high", (f"ev-{index}",),
            f"explanation {index}", True, 0, "gemma4:e4b", (), "v2", 10) for index, item in enumerate(events))
        queue = GemmaQueueStats(16, 0, 4, 0, 4, 0, 0, 0, 0, "idle")
        snap = SimpleNamespace(gemma_queue=queue, gemma_annotations=annotations, dsp_events=events)
        panel = GlitchTui._gemma_detail(snap, events[-1])
        self.assertIn("DSP verdict: detected/click", panel)
        self.assertIn("Gemma: coherent / high", panel)
        self.assertIn("Event: event-0", panel)
        self.assertIn("Event: event-3", panel)
        self.assertIn("Evidence:\n  - ev-3", panel)
        self.assertIn("completed 4/4", GlitchTui._gemma_status(snap))
        waiting = GlitchTui._gemma_detail(SimpleNamespace(gemma_queue=GemmaQueueStats(16, 0, 1, 0, 0, 0, 0, 0, 1, "draining"),
            gemma_annotations=(), dsp_events=events), events[-1])
        self.assertIn("do not quit yet", waiting)
        disabled = GlitchTui._gemma_detail(SimpleNamespace(gemma_queue=None, gemma_annotations=(), dsp_events=events), events[-1])
        self.assertIn("disabled", disabled)

    async def test_gemma_panel_wraps_and_scrolls_without_hiding_status(self) -> None:
        runtime = SliceRuntime(PCMBlockRing(2, 4), type("Producer", (), {"state": "idle", "error": None})())
        app = GlitchTui(runtime)
        long_text = " ".join(["readable-explanation"] * 80)
        async with app.run_test(size=(100, 30)) as pilot:
            app._refresh_timer.stop()  # Keep this rendering test independent from runtime snapshots.
            app.refresh_snapshot()
            detail = app.query_one("#gemma-detail")
            detail.update(Text(long_text, overflow="fold", no_wrap=False))
            await pilot.pause()
            rendered = detail.render()
            wrapped = rendered.wrap(detail.size.width)
            self.assertGreater(len(wrapped), 1)
            self.assertTrue(all(line.cell_length <= detail.size.width for line in wrapped))
            scroll = app.query_one("#gemma-scroll")
            self.assertEqual(scroll.scroll_y, 0)
            app.action_scroll_gemma_down()
            await pilot.pause()
            self.assertGreater(scroll.scroll_y, 0)
            app.action_scroll_gemma_up()
            await pilot.pause()
            self.assertIn("[/] scroll · mouse wheel", str(app.query_one("#gemma-status").content))
        self.assertIn(("[", "scroll_gemma_up", "Gemma up"), app.BINDINGS)
        self.assertIn(("]", "scroll_gemma_down", "Gemma down"), app.BINDINGS)

    async def test_gemma_summary_has_four_canonical_rows_and_toggle(self) -> None:
        events = (
            DSPEvent("poc:e0:00004", 4, "CLOSED", "detected", "clipping", 400, 480, 480, 4.0,
                ("flat_top",), ("flat-4",), "poc-d2-v2", 0, True),
            DSPEvent("poc:e0:00001", 1, "CLOSED", "detected", "click", 100, 120, 120, 1.0,
                ("click",), ("click-1",), "poc-d2-v2", 0, True),
            DSPEvent("poc:e0:00003", 3, "CLOSED", "uncertain", "stutter", 300, 360, 360, 3.0,
                ("block_repeat",), ("repeat-3",), "poc-d2-v2", 0, True),
            DSPEvent("poc:e0:00002", 2, "CLOSED", "detected", "dropout", 200, 260, 260, 2.0,
                ("dropout",), ("dropout-2",), "poc-d2-v2", 0, True),
        )
        annotations = (
            GemmaAnnotation("v", "poc:e0:00003", 3, "coherent", "stutter", "high", (), "ok", True,
                0, "gemma4:e4b", (), "v2", 10),
            GemmaAnnotation("v", "poc:e0:00004", 4, "error", None, None, (), "", True,
                0, "gemma4:e4b", (), "v2", 10, "model unavailable"),
            GemmaAnnotation("v", "poc:e0:00001", 99, "coherent", "click", "high", (), "wrong revision", True,
                0, "gemma4:e4b", (), "v2", 10),
        )
        pending = SimpleNamespace(
            gemma_queue=GemmaQueueStats(16, 1, 4, 0, 2, 0, 1, 0, 2, "analyzing"),
            gemma_annotations=annotations,
            dsp_events=events,
        )
        summary = GlitchTui._gemma_summary(pending)
        self.assertIn("Evento  Tipo DSP", summary)
        self.assertIn("Score raw", summary)
        self.assertIn("Gemma annotation  Confidence", summary)
        self.assertEqual([line.split()[0] for line in summary.splitlines()[2:]], ["00001", "00002", "00003", "00004"])
        self.assertIn("stutter (block_repeat) [uncertain]", summary)
        self.assertIn("clipping (flat_top) [detected]", summary)
        self.assertIn("queued", summary)  # Revision 99 does not correlate to event 00001.
        self.assertIn("coherent", summary)
        self.assertIn("error", summary)
        disabled = GlitchTui._gemma_summary(SimpleNamespace(gemma_queue=None, gemma_annotations=(), dsp_events=events))
        self.assertEqual(disabled.count("disabled"), 4)

        runtime = SliceRuntime(PCMBlockRing(2, 4), type("Producer", (), {"state": "idle", "error": None})())
        runtime._events.extend(events)  # Test-only immutable snapshot setup.
        app = GlitchTui(runtime)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            self.assertIn("DETAIL", str(app.query_one("#gemma-status").content))
            app.action_toggle_gemma_summary()
            await pilot.pause()
            self.assertTrue(app._show_gemma_summary)
            self.assertIn("SUMMARY", str(app.query_one("#gemma-status").content))
            self.assertIn("Evento  Tipo DSP", str(app.query_one("#gemma-detail").content))
            app.action_toggle_gemma_summary()
            await pilot.pause()
            self.assertFalse(app._show_gemma_summary)
            self.assertIn("DETAIL", str(app.query_one("#gemma-status").content))
            self.assertIn("Gemma annotations are disabled.", str(app.query_one("#gemma-detail").content))
        self.assertIn(("ctrl+s", "toggle_gemma_summary", "Summary / detail"), app.BINDINGS)

    async def test_d_binding_starts_explicit_nonblocking_drain(self) -> None:
        runtime = SliceRuntime(PCMBlockRing(2, 4), type("Producer", (), {"state": "idle", "error": None})())
        runtime.gemma_worker = object()  # type: ignore[assignment]
        app = GlitchTui(runtime)
        with patch.object(runtime, "gemma_drain_budget", return_value=73.0), patch.object(app, "_drain_then_exit") as drain:
            app.action_drain_and_quit()
        self.assertTrue(app._draining)
        self.assertEqual(app._drain_budget_s, 73.0)
        drain.assert_called_once()
        self.assertIn(("q", "cancel_and_quit", "Cancel pending"), app.BINDINGS)
        self.assertIn(("d", "drain_and_quit", "Drain Gemma + quit"), app.BINDINGS)

    async def test_textual_runtime_uses_prestarted_spawn_executor_for_annotation(self) -> None:
        producer = type("Producer", (), {"state": "eof", "error": None})()
        runtime = SliceRuntime(PCMBlockRing(2, 4), producer)
        runtime.gemma_worker = GemmaWorker(TuiSpawnClient(), runtime.add_annotation)  # type: ignore[arg-type]
        runtime.start()
        event = DSPEvent("tui:e0:00001", 1, "CLOSED", "uncertain", "loop", 0, 480, 480, 1.0,
            ("loop",), ("tui:e0:loop:0:480",), "poc-d2-v2", 0, True)
        evidence = (("tui:e0:loop:0:480", {"stream_id": "observed", "start_frame": 0, "end_frame": 480,
            "sample_rate_hz": 48_000, "window_ms": 1000, "features": {"correlation_lag_500ms": .999},
            "profile_id": "poc-d2-v2"}),)
        try:
            async with GlitchTui(runtime).run_test(size=(100, 30)) as pilot:
                self.assertTrue(runtime.gemma_worker.submit(event, evidence))
                deadline = asyncio.get_running_loop().time() + 3
                while not runtime.snapshot().gemma_annotations and asyncio.get_running_loop().time() < deadline:
                    await pilot.pause()
                    await asyncio.sleep(.01)
                annotation = runtime.snapshot().gemma_annotations[0]
                self.assertEqual((annotation.annotation_status, annotation.explanation), ("coherent", "TUI spawn mock"))
        finally:
            runtime.stop()


if __name__ == "__main__":
    unittest.main()
