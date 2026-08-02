"""Print duration, format and level for each WAV given.

The quick "did that generate sensibly?" check. Peak near 0 dBFS suggests
clipping; an RMS down near -40 dBFS usually means the model produced near
silence, which is what a bad prompt or a broken weight load looks like.

    python3 wavchk.py out/*.wav
"""

from __future__ import annotations

import os
import sys
import wave

import numpy as np


def dbfs(x: float) -> float:
    """Amplitude (0..1) as dBFS, floored so silence cannot produce -inf."""
    return 20 * np.log10(max(x, 1e-9))


def describe(path: str) -> str:
    """One line of duration, sample rate, channels, peak and RMS."""
    with wave.open(path) as w:
        frames = w.getnframes()
        rate = w.getframerate()
        channels = w.getnchannels()
        width = w.getsampwidth()
        raw = w.readframes(frames)

    # These tools assume the 16-bit PCM that sa-audio writes. Reading any
    # other width as int16 silently produces meaningless numbers, so refuse
    # rather than report a confident wrong answer.
    if width != 2:
        return f"{os.path.basename(path):22s} unsupported: {width * 8}-bit (expected 16-bit PCM)"

    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768
    peak = float(np.abs(a).max()) if a.size else 0.0
    rms = float(np.sqrt((a**2).mean())) if a.size else 0.0

    return (
        f"{os.path.basename(path):22s} {frames / rate:5.1f}s {rate}Hz {channels}ch  "
        f"peak {dbfs(peak):6.1f}dBFS  rms {dbfs(rms):6.1f}dBFS"
    )


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    for path in sys.argv[1:]:
        if not os.path.exists(path):
            print(f"{os.path.basename(path)}: MISSING")
            continue

        print(describe(path))

    return 0


if __name__ == "__main__":
    sys.exit(main())
