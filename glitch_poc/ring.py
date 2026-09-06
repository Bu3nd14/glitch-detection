from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

import numpy as np

from .contracts import PCMBlock


@dataclass(frozen=True)
class RingStats:
    capacity: int
    fill: int
    dropped_blocks: int
    dropped_frames: int


class PCMBlockRing:
    """Bounded FIFO with preallocated PCM storage. Producer never waits for UI."""

    def __init__(self, capacity: int, block_frames: int, channels: int = 2) -> None:
        if capacity < 1 or block_frames < 1:
            raise ValueError("capacity and block_frames must be positive")
        self._storage = np.empty((capacity, block_frames, channels), dtype=np.float32)
        self._metadata: list[PCMBlock | None] = [None] * capacity
        self._capacity, self._block_frames = capacity, block_frames
        self._read = self._write = self._fill = 0
        self._dropped_blocks = self._dropped_frames = 0
        self._lock = Lock()

    def push(self, block: PCMBlock) -> None:
        if block.frame_count > self._block_frames:
            raise ValueError("block exceeds preallocated block size")
        with self._lock:
            discontinuity = block.discontinuity
            missing = block.missing_frames
            if self._fill == self._capacity:
                old = self._metadata[self._read]
                assert old is not None
                self._dropped_blocks += 1
                self._dropped_frames += old.frame_count
                self._read = (self._read + 1) % self._capacity
                self._fill -= 1
                discontinuity, missing = True, missing + old.frame_count
            slot = self._storage[self._write]
            slot[:block.frame_count] = block.samples
            stored = PCMBlock.create(block.stream_id, block.sequence, block.start_frame,
                slot[:block.frame_count], sample_rate_hz=block.sample_rate_hz,
                profile_id=block.profile_id, discontinuity=discontinuity, missing_frames=missing)
            self._metadata[self._write] = stored
            self._write = (self._write + 1) % self._capacity
            self._fill += 1

    def pop(self) -> PCMBlock | None:
        with self._lock:
            if not self._fill:
                return None
            result = self._metadata[self._read]
            self._metadata[self._read] = None
            self._read = (self._read + 1) % self._capacity
            self._fill -= 1
            return result

    def stats(self) -> RingStats:
        with self._lock:
            return RingStats(self._capacity, self._fill, self._dropped_blocks, self._dropped_frames)
