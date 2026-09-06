from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from threading import Event, Thread
from typing import IO

import numpy as np

from .contracts import PCMBlock
from .ring import PCMBlockRing


def ffmpeg_command(path: str, *, realtime: bool = True) -> list[str]:
    """One decoder process emits normalized f32le; -re avoids filling/dropping the FIFO."""
    return ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", *( ["-re"] if realtime else []),
            "-i", path, "-vn", "-ar", "48000", "-ac", "2", "-f", "f32le", "pipe:1"]


def pcm_blocks(source: IO[bytes], *, stream_id: str, block_frames: int = 1024) -> Iterator[PCMBlock]:
    bytes_per_block = block_frames * 2 * 4
    pending = b""
    sequence = start_frame = 0
    while True:
        data = source.read(bytes_per_block)
        if not data:
            break
        pending += data
        complete = len(pending) // 8 * 8
        if not complete:
            continue
        frames = np.frombuffer(pending[:complete], dtype="<f4").reshape(-1, 2)
        pending = pending[complete:]
        while len(frames):
            chunk, frames = frames[:block_frames], frames[block_frames:]
            # Own the buffer before the pipe's bytes object is released.
            samples = np.array(chunk, dtype=np.float32, copy=True)
            yield PCMBlock.create(stream_id, sequence, start_frame, samples)
            sequence += 1
            start_frame += len(samples)
    if pending:
        raise ValueError(f"truncated PCM stream: {len(pending)} byte(s) do not form a stereo f32 frame")


class FFmpegProducer:
    def __init__(self, path: str, ring: PCMBlockRing, *, stream_id: str, block_frames: int = 1024) -> None:
        self.path, self.ring, self.stream_id, self.block_frames = path, ring, stream_id, block_frames
        self.state = "idle"
        self.error: str | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._stop = Event()
        self._thread: Thread | None = None

    def start(self) -> bool:
        if shutil.which("ffmpeg") is None:
            self.state, self.error = "unavailable", "ffmpeg not found on PATH"
            return False
        self._thread = Thread(target=self._run, name="ffmpeg-pcm-producer", daemon=True)
        self._thread.start()
        return True

    def _run(self) -> None:
        try:
            self.state = "running"
            self._process = subprocess.Popen(
                ffmpeg_command(self.path), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )
            assert self._process.stdout is not None
            for block in pcm_blocks(self._process.stdout, stream_id=self.stream_id, block_frames=self.block_frames):
                if self._stop.is_set():
                    break
                self.ring.push(block)
            code = self._process.wait()
            self.state = "eof" if code == 0 and not self._stop.is_set() else "stopped"
            if code and not self._stop.is_set():
                self.error = f"ffmpeg exited with code {code} (stderr disabled for bounded ingest)"
                self.state = "error"
        except (OSError, ValueError) as error:
            self.state, self.error = "error", str(error)

    def stop(self) -> None:
        self._stop.set()
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
        if self._thread is not None:
            self._thread.join(timeout=1)
