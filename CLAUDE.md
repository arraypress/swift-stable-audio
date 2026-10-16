# CLAUDE.md — swift-stable-audio

On-device text-to-audio (SFX + music) for macOS/iOS. Module: **StableAudioKit**. One API + a model registry + engines behind a protocol (same shape as swift-llm / swift-tts): **Core AI** (`CoreAIEngine`, the published `.aimodel` exports — what sfx runs since 0.3.0), **MLX** (`MLXEngine`, also iPhone, locally converted weights) and cloud.

## Build & test
```bash
swift build                     # all products
swift test                      # 31 offline unit tests (core only — no GPU/MLX/Core AI)
swift build -c release --product sa-coreai   # Core AI: info / generate / parity
swift build --product sa-audio  # typed prompt → WAV
```

## The MLX metallib (superseded 2026-09-03)
A plain `swift build` now emits `mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib` beside each binary and MLX loads it from there — so binaries run headless with nothing copied, and a release ships the binary plus that bundle. The old workaround below (a Python `mlx.metallib` beside the binary) still works but is no longer needed:
```bash
cp <path-to-mlx-python>/site-packages/mlx/lib/mlx.metallib \
   "$(swift build --show-bin-path)/mlx.metallib"
```
The bin path is **not** `.build/arm64-apple-macosx/debug` on the current
toolchain — it is `.build/out/Products/Debug`. Ask for it rather than writing
it down, or the copy silently lands nowhere and the binary dies looking for
Metal.
Xcode app builds (the StableAudioTester companion) bundle the metallib automatically — nothing to do there.

## ⚠️ Memory footprint — this crashed the machine once

Measured **peak memory footprint** for one *medium* clip on a 36 GB Mac:

| Clip length | Peak |
|---|---|
| 5 s | 9.3 GB |
| 10 s | 10.7 GB |
| 15 s | 13.6 GB |
| 20 s | 14.6 GB |
| 30 s | 17.2 GB |

Roughly **7.7 GB fixed + 0.32 GB per second of audio**.

The **small** models are about half that — measured on the same machine with
`sa3-small-sfx`:

| Clip length | Peak |
|---|---|
| 5 s | 4.6 GB |
| 15 s | 6.3 GB |

Roughly **3.8 GB fixed + 0.17 GB per second**. A 120 s small clip therefore
extrapolates to ~24 GB, which is still most of a 36 GB machine — the small
models are cheaper, not cheap. The fixed base is the 8-bit weights dequantised to a full fp16 copy at load; the growth is the medium decoder's O(seq²) band attention. A 30 s clip generates in ~6 s.

- **`ps` RSS is useless here — it reads ~1.2 GB for a 17.2 GB job**, because MLX's unified-memory GPU buffers don't count toward RSS. Always measure with `/usr/bin/time -l` and read "peak memory footprint".
- **Run generation strictly sequentially.** Two 30 s clips in parallel ≈ 35 GB ≈ the whole machine. Never run a weight conversion concurrently with generation. ~60 s clips extrapolate to ~27 GB — don't attempt alongside anything.

## ⭐ SA3 prompting rules (learned the hard way)

1. **Never use words describing quietness.** "faint", "quietly", "soft hum" produce files at −40 to −59 LUFS that are unusable. Rewriting one as "**loud** flyback whine, **strong** electrical hum" moved it **33 dB**. Generate loud and turn it down in the DAW.
2. **Avoid motion/modulation words** ("evolving", "slow filter sweep", "shifting") — they push the model off its stationary comfort zone into unfocused mush. Prefer static timbral words: "sustained", "slow attack", "chord", "dreamy".
3. **Negation is unreliable** — "no drums" may add drums. (ACE-Step respects negation; SA3 does not.)
4. **SA3 is a texture engine.** Best at continuous stationary ambience, industrial/sci-fi texture, sustained tonal beds, wordless vocal timbre, struck-and-decaying resonance. Worst at tempo (there is *no* tempo conditioning — only text + duration), musical structure, lyrics, and change-over-time. Short designed one-shots (clicks, foley) are weaker but usable.

## Core AI engine (1.2.0)
- Exports: `arraypress/stable-audio-3-{small-sfx,small-music,medium}-coreai` on HF (public, the user's). Functions are `condition`, `denoise` (dynamic batch + length), `decode_N`/`encode_N` for N in 256…4096 (medium 256…2048) — the README's `conditioner`/`dit`/`decoder_N` names are WRONG, example.py's are right (checked with `sa-coreai info`). `cross_attn_mask` is float32.
- **Parity: bit-identical to example.py** (all stages max|Δ| 0 with the same numpy noise); with CFG 3 + negative 82.5 dB (example.py's `guide` sees a float32 latent on step 0). Keep the host maths in float64 or parity goes.
- Window ≥ 256 frames (23.8 s), +3 s margin, conditioner told the WINDOW's seconds (training used 23.78 for everything; telling it 2 s puts the loss ×10⁴ — `LLM Training/Code/stable-audio/lora/prepare_captioned.py`). 258 frames needs the 512 rung — cost, not a bug.
- Core AI compiles the WHOLE asset (every rung, encoders included) on first load per executable: small-sfx/music 14–22 min and 7.0 GB; **medium 5 h 3 min and 29.5 GB** (one core; an attempt was killed at 2 h 26 min with nothing cached — run it detached). Cache folder name = the `main.mlirb` SHA-256 of `~/Library/Caches/coreai-cache/<build>/<program>/<main.hash hex>`. Every new program name (test runner, probe, python) pays it again — delete stale entries.
- Memory flat ~3 GB for small at 2–120 s (MLX: 4.5→9.6 GB). First clip in a process 4–7 s (warm-up, mostly the first decode), later clips 1.0 s.
- No CFG without a negative in example.py; here cfg > 1 without one guides against zeros (the reference's null embed).

## Module map
- `StableAudioKit/` — `StableAudioModel` (registry), `GenerationOptions`, `GeneratedAudio` (WAV + `trimmingEnd`/`faded`/`seamlessLoop`), `RemoteEngine`, `AudioGenEngine` protocol, `SATokenizer`, `Support/` (`PingPongSampler`, `GaussianNoise`, `ClipPlan` — shared by both engines).
- `StableAudioKitCoreAI/` — `CoreAIEngine` + `Tensors` (NDArray in/out, any strides).
- `StableAudioKitMLX/` — `MLXEngine` + ported nets: `SADiT`, `SADecoder`, `SAT5Gemma`, `SAPipeline` (ping-pong sampler), `MLXOps` (shared primitives). `SADiT`/`SADecoder` **infer dims from the weights** → one codebase runs small + medium.
- `StableAudioKitGemma/` — `GemmaTokenizer` (swift-transformers `AutoTokenizer`). App injects it via the `SATokenizer` protocol (in the core since 1.2.0; `StableAudioKitMLX` re-exports the core so old imports still compile).

## `sa-audio` CLI
```
sa-audio <tokDir> <weights> <out.wav> <prompt> [seconds] [loop|bpm|none] [seed]
```
- Arg 6: a **number** = bar-locked musical loop at that BPM; literal **`loop`** = texture loop (long equal-power crossfade, for SFX/ambience); **`none`**/`-` = raw. You need `none` to pass a seed without looping.
- Arg 7 = seed. **Generation is fully deterministic** — same prompt + seed → byte-identical file. Varying the seed is the *only* way to get different takes of one prompt.
- **No seed = a random seed** (since 1.1.0): `MLXEngine` draws one per generation from `GenerationOptions.seedRange` and reports it as `GeneratedAudio.seed`. Up to 1.0.0 a nil seed silently meant 0.

## Weight conversion
Use **`Tools/convert_stream.py`** (in this repo — see `Tools/README.md`):
```bash
python3 Tools/convert_stream.py <hf-snapshot-dir> <out.safetensors> [--dry-run] [--source=REPO@REVISION]
```
`--source` writes `source_repository`/`source_revision` into the header's `__metadata__`; `StableAudioWeights.source(of:)` reads them back and sfx records them in recipes. Small-sfx converts too (2026-10-15: 688 tensors, 3.18 GB fp32, from `stabilityai/stable-audio-3-small-sfx@05ca385`).
Was `~/StableAudioModels/convert_stream.py`, which was unversioned. That copy still exists and is identical; this one is now canonical.
**numpy + stdlib only — no torch, no stable-audio-tools, no venv.** Streams tensor-by-tensor off disk, so peak RAM is the largest single tensor (786 MB, the T5 embed table) rather than the old `get_pretrained_model()` route's ~2× model size (~20 GB — this is what crashed the Mac). Medium converts in <20 s → 880 tensors / 8.64 GB fp32. Handles bf16→f32 (T5Gemma is BF16) and the weight-norm mapping conv (`g·v/‖v‖`).

**Gotcha:** the six front-of-decoder tensors need the `dec.` prefix (`dec.map_weight`, `map_bias`, `new_tokens`, `lin_w`, `lin_b`, `running_std`) — `MLXEngine` routes via `sub("dec.")` and won't see them bare. `num_w`/`num_b`/`padding_embedding` stay unprefixed.

## Behaviours worth knowing
- **End-of-clip tone:** the model marks its final latent frame. `MLXEngine` generates `seconds + 3s` of frames and crops back to `seconds` so the artifact is discarded (mirrors the reference pipeline). Do NOT "fix" this by trimming/fading the output — that eats real content.
- **Loops:** `options.loop` → equal-power texture crossfade; `options.loopBPM` → crop to a whole number of bars at that tempo (downbeat-aligned) + 12 ms de-click. The model isn't tempo-locked, so bar-cropping is a safety net, not a guarantee.
- **Quantization:** 8-bit (`quant.bits` metadata) is near-lossless (0.98 correlation vs fp32, confirmed by ear) and dequantizes to fp16 at load. **4-bit is audibly degraded** for this diffusion model (0.15 correlation) — the 8 sampling steps compound weight error. Don't ship it.
- **The 8-bit quantizer is still missing.** The shipped `sa3-*.safetensors` are 8-bit (`quant.bits = 8`, `quant.group = 64`), but `convert_stream.py` only emits fp32 and `MLXEngine` only *de*quantizes. `Tools/quantize.py` was recovered from a session transcript on 2026-08-02, but it is the **4-bit** experiment: it hardcodes `BITS = 4` and never writes the `quant.bits`/`quant.group` tensors the engine needs. Getting back to 8-bit means `BITS = 8`, emitting those two tensors, and A/B-ing against an fp32 reference. **Until that is done and verified, the existing weight files remain irreplaceable.**
- Frames `T = round(seconds · 44100 / 4096)`, rounded even.

## Port validation (reference numbers)
Every net was implemented in MLX-Python, matched against PyTorch golden fixtures, then transcribed 1:1 to MLX-Swift and re-checked on GPU via `sa-validate`. Max abs diff vs PyTorch:

| Net | MLX-Python | MLX-Swift (GPU) |
|---|---|---|
| Decoder (small) | 2.7e-5 | 2.7e-5 |
| Decoder (medium) | — | 1.44e-5 |
| DiT (20-layer) | 1.14e-4 | 9.9e-4 |
| Sampler | 9.5e-5 | — |
| T5Gemma | 1.9e-3 | 9.7e-2 |
| End-to-end audio | 8.7e-4 | — |

T5Gemma's larger residual is **expected, not a bug**: the HF model is bf16, so validate against a float32 re-run (`enc.float()`). The Swift figure matches the MLX-Python port exactly.

Architecture was reverse-engineered from `stable-audio-tools` (the authoritative reference). Shared primitives proven across nets: differential attention, DynamicTanh, partial RoPE, SwiGLU, weight-norm Conv1d, chunked midpoint-shift, patch-unfold. Medium differs from small: decoder 1536/24 heads/12 layers, `Sin(x)=sin(πx)` in the last 8 blocks, no midpoint-shift, sliding-window attention (band ±17); DiT uses differential attention and embed 1536/depth 24.

## ⭐ Quality-floor decision
The user judges **small (0.6 B) not good enough — medium (2 B) is the minimum acceptable quality.** `large` is closed (Stability API/enterprise only; weights unobtainable). Consequence: **the flagship is a Mac-class product, not iPhone.** The iPhone tier is demoted to a "quick sketch" on small; medium-on-iPhone is only *maybe* viable for short clips and only after true in-memory 8-bit (unproven — not a promise). The audience is producers on Macs anyway.

## Rejected / evaluated
- **ACE-Step 1.5 — evaluated and rejected.** Installed, ran, deleted. It genuinely has what SA3 lacks: real `bpm`/`keyscale`/`timesignature` conditioning (asked 90 → measured 90.0), 10–600 s durations, repaint/edit, stem separation, documented LoRA/SFT with working MPS paths, MIT licence, fully native MLX. **But the audio was judged "terrible… not even music".** *Caveat for any re-evaluation:* that verdict was on the **turbo model at 8 steps with `thinking=False`** — the cheapest possible config. Base/XL at 32–100 steps with CoT were never tested. It also writes full songs, not loops, and stem extraction needs the base model.
- **MOSS-SoundEffect-v2** — ~80 s/step on CPU, MPS blocked by complex-RoPE. Impractical.
- **Landscape re-scouted 2026-07-24:** SA3 (released ~May 2026) is current; nothing newer to migrate to. The "459 M vs 0.6 B" confusion is DiT-only vs full-stack count — same models. Vocals gap → cloud (`RemoteEngine`) is the pragmatic path.

## ⚠️ Never put irreplaceable artifacts in the scratchpad
Session scratchpads live under `/private/tmp/claude-501/<session-id>/` and are **wiped on reboot**. A machine crash on 2026-07-25 destroyed the MLX-Python venv, the `mlx_*.py` reference ports, the `dump_*`/`export_*` fixture scripts, and the fixtures directory. They're recoverable only by extracting Write-tool contents from session transcripts. Durable assets that survived: `~/StableAudioModels/` (quantized weights + `tokenizer/`), this git repo, and the built binaries with their colocated `mlx.metallib`. **The four Python tools are now in `Tools/` and versioned** — they were the remaining unbacked-up work.

## Local assets (not in the repo)
- **2026-10-16: `~/Developer/StableAudioModels` (the fp32 small-sfx conversion + tokenizer) was DELETED** once sfx 0.3.0 ran from the Core AI exports; nothing else used it. MLX runs need a fresh `Tools/convert_stream.py` conversion. ~~Quantized weights + tokenizer: `~/StableAudioModels/`.~~ (that folder had already gone) **Not reproducible** — see the quantizer note above. `tokenizer/` alone is replaceable: identical files ship in the HF snapshot under `t5gemma-b-b-ul2/`.
- HF source snapshot (conversion input): `~/.cache/huggingface/hub/models--stabilityai--stable-audio-3-medium` (9.7 GB). Only *medium* is cached — the `sa3-small-music` / `sa3-small-sfx` sources are gone, so those two would need re-downloading.
- ~~fp32 reference for 8-bit A/B: `~/Documents/stable/weights/sa3_medium_fp32.safetensors`~~ — **gone**, verified 2026-08-02. Regenerate with `Tools/convert_stream.py` if an A/B is needed again (8.64 GB).
- Companion app: `../../swift-stable-audio-tester` (xcodegen; `xcodegen generate && xcodebuild ...`).

## Open direction
LoRA / fine-tune of **medium** on **Freshly Squeezed Samples** (owned, already-tagged, fully-cleared library) → merge LoRA into base weights → existing `convert_stream.py` → MLX engine, giving an on-device "house sound" generator with no new engine. Tags collapse the captioning cost; owned data removes copyright risk; bar-aligned BPM-labelled loops should improve loopability. **TODO:** confirm `stable-audio-tools` LoRA/fine-tune supports the SA3 (T5Gemma) family at the *medium* size, and estimate 2 B fine-tune box/cost.

## Conventions
- Commit only when asked; branch off `main` first if needed.
- Weights/fixtures/venv stay OUT of git.
