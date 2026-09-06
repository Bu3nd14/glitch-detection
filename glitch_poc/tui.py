from __future__ import annotations

from typing import ClassVar

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Grid, ScrollableContainer, Vertical
from textual.timer import Timer
from textual.widgets import Footer, Header, Static

from .contracts import DSPEvent, GemmaAnnotation
from .runtime import RuntimeSnapshot, SliceRuntime


def wave(values: tuple[float, ...]) -> str:
    glyphs = "▁▂▃▄▅▆▇█"
    return "".join(glyphs[min(7, max(0, int((value + 1) * 3.5)))] for value in values)


class GlitchTui(App[None]):
    CSS = """
    Grid { grid-size: 2 2; grid-gutter: 1; height: 1fr; }
    .panel { border: round $accent; padding: 0 1; }
    #gemma { padding: 0 1; }
    #gemma-status { height: 3; border-bottom: solid $accent; }
    #gemma-scroll { height: 1fr; }
    #gemma-detail { width: 1fr; height: auto; text-wrap: wrap; }
    #gemma-detail.summary { min-width: 88; text-wrap: nowrap; }
    #too-small { display: none; height: 1fr; content-align: center middle; }
    """
    BINDINGS: ClassVar = [("q", "cancel_and_quit", "Cancel pending"),
                             ("d", "drain_and_quit", "Drain Gemma + quit"),
                             ("[", "scroll_gemma_up", "Gemma up"),
                             ("]", "scroll_gemma_down", "Gemma down"),
                             ("ctrl+s", "toggle_gemma_summary", "Summary / detail")]

    def __init__(self, runtime: SliceRuntime, *, gemma_drain_timeout_s: float | None = None) -> None:
        super().__init__()
        self.runtime = runtime
        self.gemma_drain_timeout_s = gemma_drain_timeout_s
        self._draining = False
        self._show_gemma_summary = False
        self._drain_budget_s: float | None = None
        self._refresh_timer: Timer | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("Terminal too small: this POC needs at least 100x30.", id="too-small")
        with Grid(id="panels"):
            yield Static(classes="panel", id="source")
            yield Static(classes="panel", id="health")
            yield Static(classes="panel", id="dsp")
            with Vertical(classes="panel", id="gemma"):
                yield Static(id="gemma-status")
                with ScrollableContainer(id="gemma-scroll"):
                    yield Static(id="gemma-detail")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_timer = self.set_interval(1 / 12, self.refresh_snapshot)

    def on_unmount(self) -> None:
        if self._refresh_timer is not None:
            self._refresh_timer.stop()

    def action_cancel_and_quit(self) -> None:
        """Normal q is deliberately immediate; pending annotations receive audit cancellation outcomes."""
        if not self._draining:
            self.exit()

    def action_drain_and_quit(self) -> None:
        """Explicitly opt into a bounded post-EOF drain without blocking Textual's event loop."""
        if self._draining or self.runtime.gemma_worker is None:
            return
        self._draining = True
        self._drain_budget_s = self.runtime.gemma_drain_budget(self.gemma_drain_timeout_s)
        self._drain_then_exit()

    def action_scroll_gemma_up(self) -> None:
        self.query_one("#gemma-scroll", ScrollableContainer).scroll_relative(y=-5, animate=False)

    def action_scroll_gemma_down(self) -> None:
        self.query_one("#gemma-scroll", ScrollableContainer).scroll_relative(y=5, animate=False)

    def action_toggle_gemma_summary(self) -> None:
        """Switch the Gemma pane only; this never changes runtime or audit state."""
        self._show_gemma_summary = not self._show_gemma_summary
        self.query_one("#gemma-scroll", ScrollableContainer).scroll_home(animate=False)
        self.refresh_snapshot()

    @work(thread=True)
    def _drain_then_exit(self) -> None:
        self.runtime.stop(graceful_gemma_drain=True, gemma_drain_timeout_s=self.gemma_drain_timeout_s)
        self.call_from_thread(self.exit)

    def on_resize(self) -> None:
        compact = self.size.width < 100 or self.size.height < 30
        self.query_one("#too-small", Static).display = compact
        self.query_one("#panels", Grid).display = not compact

    def refresh_snapshot(self) -> None:
        # Textual may deliver a final queued timer tick after screen teardown in tests.
        if not self.query("#source"):
            return
        snap = self.runtime.snapshot()
        seconds = snap.position_frames / 48_000
        self.query_one("#source", Static).update(
            f"OBSERVED PCM — {getattr(self.runtime, 'source_label', 'source unavailable')}\n{wave(snap.waveform)}\n"
            f"position {seconds:6.2f}s  RMS {snap.rms:.3f}  peak {snap.peak:.3f}")
        self.query_one("#health", Static).update(
            f"RUNTIME HEALTH\nFFmpeg: {snap.ffmpeg_state} {snap.ffmpeg_error or ''}\n"
            f"ring {snap.ring.fill}/{snap.ring.capacity}  dropped {snap.ring.dropped_blocks}  underruns {snap.underruns}\n"
            f"audit {snap.audit_log.state if snap.audit_log else 'disabled'} "
            f"drop {snap.audit_log.dropped if snap.audit_log else 0} err {snap.audit_log.errors if snap.audit_log else 0}")
        latest = snap.dsp_events[-1] if snap.dsp_events else None
        if latest is None:
            detail = "state: clean (no emitted candidate)"
        else:
            detail = (f"{latest.status} {latest.glitch_type} score {latest.score:.2f}\n"
                      f"frames {latest.start_frame}-{latest.end_frame}\n"
                      f"evidence {', '.join(latest.evidence_ids)}")
        self.query_one("#dsp", Static).update(f"DSP DETECTION\n{detail}")
        gemma_detail = self.query_one("#gemma-detail", Static)
        gemma_detail.set_class(self._show_gemma_summary, "summary")
        self.query_one("#gemma-status", Static).update(
            Text(self._gemma_status(snap, summary=self._show_gemma_summary, drain_budget_s=self._drain_budget_s)))
        if self._show_gemma_summary:
            gemma_detail.update(Text(self._gemma_summary(snap), overflow="crop", no_wrap=True))
        else:
            gemma_detail.update(Text(self._gemma_detail(snap, latest), overflow="fold", no_wrap=False))

    @staticmethod
    def _gemma_status(snap: RuntimeSnapshot, *, summary: bool = False, drain_budget_s: float | None = None) -> str:
        mode = "SUMMARY" if summary else "DETAIL"
        if snap.gemma_queue is None:
            return (f"GEMMA {mode} — disabled\nEnable with --gemma-annotations.\n"
                    "[/] scroll · mouse wheel · Ctrl+S Summary/Detail")
        queue = snap.gemma_queue
        drain = "" if drain_budget_s is None else f"  drain {drain_budget_s:.1f}s"
        return (f"GEMMA {mode} {queue.worker_state}  completed {queue.completed}/{queue.submitted}{drain}\n"
                f"queue {queue.queued}  pending {queue.pending}  dropped {queue.dropped_backpressure}\n"
                "[/] scroll · mouse wheel · Ctrl+S Summary/Detail")

    @staticmethod
    def _gemma_detail(snap: RuntimeSnapshot, latest: DSPEvent | None) -> str:
        """Structured annotation detail list; the fixed status area remains visible while it scrolls."""
        if snap.gemma_queue is None:
            return "Gemma annotations are disabled."
        queue = snap.gemma_queue
        if not snap.gemma_annotations:
            if queue.worker_state in {"analyzing", "draining"} or queue.pending:
                return f"{queue.worker_state} {queue.completed}/{queue.submitted}; do not quit yet."
            return "Waiting for CLOSED DSP events. q cancels pending; D drains then quits."
        event_by_id = {event.event_id: event for event in reversed(snap.dsp_events)}
        lines: list[str] = []
        for item in reversed(snap.gemma_annotations):
            event = event_by_id.get(item.event_id)
            verdict = "unknown" if event is None else f"{event.status}/{event.glitch_type}"
            evidence_lines = [f"  - {evidence_id}" for evidence_id in item.supporting_evidence_ids] or ["  - none"]
            lines.extend([
                "────────────────────",
                f"Event: {item.event_id}",
                f"DSP verdict: {verdict}",
                f"Gemma: {item.annotation_status} / {item.confidence or '-'}",
                "Evidence:",
                *evidence_lines,
                f"{'Error' if item.error else 'Explanation'}:",
                item.error or item.explanation or "none",
            ])
        return "\n".join(lines)

    @staticmethod
    def _gemma_summary(snap: RuntimeSnapshot) -> str:
        """One immutable display row per final DSP event, optionally correlated to Gemma."""
        events = sorted(
            (event for event in snap.dsp_events
             if event.lifecycle == "CLOSED" and event.status in {"detected", "uncertain"}),
            key=lambda event: (event.start_frame, event.event_id),
        )
        annotations = {
            (item.event_id, item.event_revision): item for item in snap.gemma_annotations
        }
        header = ("Evento  Tipo DSP                          Score raw  Frames             "
                  "Gemma annotation  Confidence")
        separator = "──────  ───────────────────────────────  ─────────  ─────────────────  ────────────────  ──────────"
        if not events:
            return f"{header}\n{separator}\nNo canonical CLOSED DSP events."
        lines = [header, separator]
        for event in events:
            annotation = annotations.get((event.event_id, event.revision))
            annotation_status, confidence = GlitchTui._summary_annotation_state(snap, annotation)
            lines.append(
                f"{GlitchTui._summary_event_id(event.event_id):<6}  "
                f"{GlitchTui._summary_dsp_type(event):<33}  "
                f"{event.raw_score:>9.2f}  "
                f"{event.start_frame}–{event.end_frame:<15}  "
                f"{annotation_status:<16}  {confidence}"
            )
        return "\n".join(lines)

    @staticmethod
    def _summary_event_id(event_id: str) -> str:
        suffix = event_id.rsplit(":", 1)[-1]
        return f"{int(suffix):05d}" if suffix.isdecimal() else suffix

    @staticmethod
    def _summary_dsp_type(event: DSPEvent) -> str:
        detector = ", ".join(event.detector_ids)
        label = event.glitch_type if not detector or detector == event.glitch_type else f"{event.glitch_type} ({detector})"
        return f"{label} [{event.status}]"

    @staticmethod
    def _summary_annotation_state(snap: RuntimeSnapshot, annotation: GemmaAnnotation | None) -> tuple[str, str]:
        if annotation is not None:
            return annotation.annotation_status, annotation.confidence or ""
        if snap.gemma_queue is None:
            return "disabled", ""
        queue = snap.gemma_queue
        if queue.queued or queue.pending or queue.submitted > queue.completed:
            return "queued", ""
        return "", ""
