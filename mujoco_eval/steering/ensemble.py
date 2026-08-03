"""Temporal ensembling of overlapping action chunks (the ACT mechanism).

A chunk policy predicts H rows and we execute a block of them, so each timestep is decided by
exactly one chunk -- the oldest row of a block is acted on H steps after the observation that
produced it. Ensembling instead replans every step and averages every prediction that covers the
current timestep, weighted toward the freshest.

This is orthogonal to the block size: it trades proxy calls (one per step rather than one per
block) for a decision that is an average over several observations.
"""
from __future__ import annotations

import collections

import numpy as np


class ChunkEnsemble:
    """Exponentially weighted average of the chunks that cover each timestep.

    weight(age) = exp(-decay * age), age 0 being the chunk predicted this step. decay=0 averages
    them equally; large decay reduces to executing the newest chunk's first row.
    """

    def __init__(self, decay=0.01, keep=None):
        self.decay = float(decay)
        self._chunks = collections.deque(maxlen=keep)

    def push(self, step, chunk):
        self._chunks.append((int(step), np.asarray(chunk, dtype=np.float32)))

    def action(self, step):
        """The ensembled action for `step`, or None if no chunk covers it."""
        rows = []
        for start, chunk in self._chunks:          # deque is in push order, oldest first
            i = step - start
            if 0 <= i < len(chunk):
                rows.append(chunk[i])
        if not rows:
            return None
        age = np.arange(len(rows))[::-1].astype(np.float32)   # 0 = freshest
        w = np.exp(-self.decay * age)
        w /= w.sum()
        return (np.stack(rows) * w[:, None]).sum(axis=0)

    def __len__(self):
        return len(self._chunks)
