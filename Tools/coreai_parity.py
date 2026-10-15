# Copyright (c) 2026 David Sherlock
#
# Use of this source code is governed by an MIT license that can be found in
# the LICENSE file or at https://opensource.org/licenses/MIT
"""The Python side of the Core AI parity check: CoreAIEngine against the export's own example.py.

    coreai_parity.py dump <export-dir> <fixtures-dir> <prompt> <seconds> <seed> <steps> [cfg] [negative]
    sa-coreai parity <export-dir> <fixtures-dir>
    coreai_parity.py compare <fixtures-dir>

`dump` runs example.py (put a copy beside this file; every export ships one) on a
prompt and seed and writes each stage: token ids, the conditioner's outputs, the
noise numpy's generator drew, one transformer step, the final latent, the audio.
`sa-coreai parity` feeds the same noise to the Swift engine and writes its stages;
`compare` prints cosine / SNR / max difference per stage.

Needs the Core AI runtime for Python (`pip install coreai-torch tokenizers numpy`
in a virtualenv), unlike the other tools here, which need only numpy.
"""
import asyncio, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import example as ex


def save(path, array):
    np.ascontiguousarray(np.asarray(array, dtype=np.float32)).tofile(path)


async def dump(model_dir, out, prompt, seconds, seed, steps, cfg, negative):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    frames = ex.frames_for(seconds)
    duration = frames * ex.DOWNSAMPLE / ex.SR
    t0 = time.time(); model = await ex.Model.load(model_dir); load = time.time() - t0
    enc = model._tokenizer.encode(prompt)
    save(out / "ids.f32", enc.ids); save(out / "mask.f32", enc.attention_mask)
    cross, mask, glob = await model.condition(prompt, duration)
    save(out / "cross.f32", cross.numpy()); save(out / "global.f32", glob.numpy())
    rng = np.random.default_rng(seed)
    noises = [rng.standard_normal((1, ex.LATENT_DIM, frames)).astype(np.float32) for _ in range(steps + 1)]
    save(out / "noise.f32", np.concatenate([n.ravel() for n in noises]))
    t = ex.schedule(steps)
    local = ex.f32(np.zeros((1, ex.LATENT_DIM + 1, frames)))
    v0 = await model.velocity(noises[0], t[0], (cross, mask, glob), local)
    save(out / "v0.f32", v0)
    # The sampler loop exactly as example.generate runs it, to keep the final latent.
    uncond = await model.condition(negative, duration) if (negative is not None and cfg != 1.0) else None
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((1, ex.LATENT_DIM, frames)).astype(np.float32)
    for i in range(steps):
        v = await model.velocity(x, t[i], (cross, mask, glob), local)
        if uncond is not None:
            vu = await model.velocity(x, t[i], uncond, local)
            v = ex.guide(x, t[i], v, vu, cfg, 0.0)
        x = ex.step(x, v, t[i], t[i + 1], rng, "pingpong")
    save(out / "latent.f32", x)
    # And the canonical path end to end.
    t0 = time.time()
    audio = await ex.generate(model_dir, prompt, seconds, seed=seed, steps=steps, negative=negative, cfg_scale=cfg)
    took = time.time() - t0
    save(out / "audio.f32", audio[0])
    json.dump({"prompt": prompt, "frames": frames, "steps": steps, "seed": seed, "cfg_scale": cfg,
               "negative": negative, "python_load_s": load, "python_generate_s": took},
              open(out / "meta.json", "w"), indent=1)
    print(f"frames {frames}, load {load:.1f}s, generate {took:.2f}s")


def metrics(a, b):
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    n = min(len(a), len(b))
    if len(a) != len(b): print(f"   (length {len(a)} vs {len(b)})")
    a, b = a[:n], b[:n]
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    err = np.sum((a - b) ** 2)
    snr = float(10 * np.log10(np.sum(a ** 2) / err)) if err > 0 else float("inf")
    return cos, snr, float(np.max(np.abs(a - b)))


def compare(out):
    out = Path(out)
    r = lambda n: np.fromfile(out / n, dtype=np.float32)
    rows = [("token ids", "ids.f32", "swift_ids.f32"), ("attention mask", "mask.f32", "swift_mask.f32"),
            ("conditioner cross_attn_cond", "cross.f32", "swift_cross.f32"),
            ("conditioner global_cond", "global.f32", "swift_global.f32"),
            ("one DiT step (v at t0)", "v0.f32", "swift_v0.f32"),
            ("decoder (Python's final latent)", "audio.f32", "swift_decoded.f32"),
            ("final audio", "audio.f32", "swift_audio.f32")]
    result = {}
    for name, p, s in rows:
        cos, snr, maxdiff = metrics(r(p), r(s))
        result[name] = {"cosine": cos, "snr_db": snr, "max_abs_diff": maxdiff}
        print(f"{name:34s} cosine {cos:.9f}  SNR {snr:7.1f} dB  max|Δ| {maxdiff:.3g}")
    json.dump(result, open(out / "parity.json", "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        a = sys.argv[2:]
        asyncio.run(dump(a[0], a[1], a[2], float(a[3]), int(a[4]), int(a[5]),
                         float(a[6]) if len(a) > 6 else 1.0, a[7] if len(a) > 7 else None))
    else:
        compare(sys.argv[2])
