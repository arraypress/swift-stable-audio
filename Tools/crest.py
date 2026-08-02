"""Rank WAVs by crest factor, to spot over-compressed or lifeless output.

Crest factor is peak minus RMS in dB — how much dynamic range a clip has
above its average level. Useful when comparing takes: a drum loop with
transients intact sits high, while something the model has smeared into a
wash of noise sits low. It says nothing about whether a clip sounds *good*,
only how peaky it is.

Rough bearings for 16-bit material: percussive content 15 dB and up,
sustained pads and drones nearer 6-10 dB, and anything under about 4 dB is
either heavily limited or barely modulating at all.

Takes a single glob, quoted so the shell does not expand it first:

    python3 crest.py 'out/*.wav'
"""

from __future__ import annotations

import glob
import os
import sys
import wave

import numpy as np


def dbfs(x: float) -> float:
    """Amplitude (0..1) as dBFS, floored so silence cannot produce -inf."""
    return 20 * np.log10(max(x, 1e-9))


def measure(path: str) -> tuple[float, float, float] | None:
    """Return `(crest_db, peak_db, seconds)`, or None if not 16-bit PCM."""
    with wave.open(path) as w:
        frames = w.getnframes()
        rate = w.getframerate()
        width = w.getsampwidth()
        raw = w.readframes(frames)

    if width != 2:
        return None

    # Channels are left interleaved: peak and RMS over the whole file are
    # what matter here, and neither depends on de-interleaving.
    a = np.frombuffer(raw, dtype=np.int16).astype(float) / 32768

    if a.size == 0:
        return None

    peak = dbfs(float(np.abs(a).max()))
    rms = dbfs(float(np.sqrt((a**2).mean())))

    return peak - rms, peak, frames / rate


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    paths = sorted(glob.glob(os.path.expanduser(sys.argv[1])))

    if not paths:
        print(f"no files matched {sys.argv[1]!r} (quote the glob so the shell keeps it intact)")
        return 1

    rows = []

    for path in paths:
        result = measure(path)

        if result is None:
            print(f"  skipped (not 16-bit PCM): {os.path.basename(path)}")
            continue

        crest, peak, seconds = result
        rows.append((crest, peak, seconds, os.path.basename(path)))

    print(" crest   peak   dur   file")

    for crest, peak, seconds, name in sorted(rows, reverse=True):
        print(f" {crest:5.1f}  {peak:5.1f}  {seconds:4.1f}s  {name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
