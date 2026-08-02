"""Estimate the tempo of each WAV given, for checking bar-locked loops.

The model is not tempo-locked, so `sa-audio`'s `loopBPM` mode crops output to
a whole number of bars at the BPM you asked for. This is how you check what
actually came out: ask for 120 and measure 121, and the crop is close enough
to be inaudible; measure 93, and the model ignored you.

Method is onset strength by spectral flux, then autocorrelation of that
envelope. The lag with the strongest correlation in the 50-200 BPM range is
taken as the beat period.

Confidence is the autocorrelation peak relative to zero-lag energy. **Do not
read it as "is there a beat here".** It measures how self-similar the onset
envelope is at the chosen lag, and a smoothly sustained sound is extremely
self-similar — a two-tone pad with no rhythm whatsoever measures 143.6 BPM at
0.890 confidence, higher than a real 120 BPM drum pattern scores at 0.843.
It is only meaningful for comparing takes that are already known to be
rhythmic.

Two further caveats. Octave errors are inherent to autocorrelation — half or
double the true tempo correlates nearly as well, so 140 BPM is often reported
as 70. And this always returns a number: there is no "no tempo found" answer,
so decide whether the material has a pulse before reading the figure.

    python3 tempo.py out/*.wav
"""

from __future__ import annotations

import os
import sys
import wave

import numpy as np

N_FFT = 1024
HOP = 256
BPM_MIN = 50
BPM_MAX = 200


def tempo(path: str) -> tuple[float | None, float]:
    """Return `(bpm, confidence)`, or `(None, 0.0)` if it cannot be estimated."""
    with wave.open(path) as w:
        rate = w.getframerate()
        channels = w.getnchannels()
        width = w.getsampwidth()
        raw = w.readframes(w.getnframes())

    if width != 2:
        return None, 0.0

    a = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768
    a = a.reshape(-1, channels).mean(1)

    frames = 1 + (len(a) - N_FFT) // HOP

    if frames < 2:
        return None, 0.0

    window = np.hanning(N_FFT)
    spectra = np.empty((frames, N_FFT // 2 + 1))

    for i in range(frames):
        spectra[i] = np.abs(np.fft.rfft(a[i * HOP : i * HOP + N_FFT] * window))

    # Spectral flux: per-frame sum of positive bin-to-bin change. Energy
    # appearing is an onset; energy decaying is not, hence positive only.
    flux = np.maximum(0, np.diff(spectra, axis=0)).sum(1)
    flux = (flux - flux.mean()) / (flux.std() + 1e-9)

    autocorr = np.correlate(flux, flux, "full")[len(flux) - 1 :]

    fps = rate / HOP
    lo, hi = int(fps * 60 / BPM_MAX), int(fps * 60 / BPM_MIN)
    window_ac = autocorr[lo:hi]

    if len(window_ac) == 0:
        return None, 0.0

    lag = lo + int(np.argmax(window_ac))
    bpm = 60 * fps / lag
    confidence = float(window_ac.max() / (autocorr[0] + 1e-9))

    return bpm, confidence


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    for path in sys.argv[1:]:
        name = os.path.basename(path)

        if not os.path.exists(path):
            print(f"{name:34s} MISSING")
            continue

        bpm, confidence = tempo(path)

        if bpm is None:
            print(f"{name:34s} no estimate (too short, or not 16-bit PCM)")
            continue

        print(f"{name:34s} est {bpm:6.1f} BPM   confidence {confidence:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
