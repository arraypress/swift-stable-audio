# StableAudioKit

One Swift API for **text-to-audio** — sound effects and music — across tiers: **on-device** (Stable Audio 3, via Apple's Core AI or MLX) or **cloud** (ElevenLabs / Stability). Program against one protocol; the model and the tier become a swappable choice, not a rewrite.

For macOS and iOS. Same shape as [LLMKit](https://github.com/arraypress/swift-llm) and the TTS library — a model registry + engines behind one protocol.

## Engines

| Engine | Tier | What it is |
|---|---|---|
| **`CoreAIEngine`** | on-device | Stable Audio 3 through **Apple's Core AI** (macOS 27 / iOS 27), running the published `.aimodel` exports ([small-sfx](https://huggingface.co/arraypress/stable-audio-3-small-sfx-coreai), [small-music](https://huggingface.co/arraypress/stable-audio-3-small-music-coreai), [medium](https://huggingface.co/arraypress/stable-audio-3-medium-coreai)). Bit-identical to the exports' own `example.py`. Steps, guidance and negative prompts. |
| **`MLXEngine`** | on-device | Stable Audio 3 (small-sfx / small-music / medium) via MLX — Mac + iPhone. Runs the full ported pipeline: T5Gemma → DiT ×8 ping-pong → decoder. Needs locally converted weights. |
| **`RemoteEngine`** | cloud | any HTTP text-to-audio endpoint — **ElevenLabs Sound Effects/Music**, Stability API, or your own proxy |

## Products

| Product | Contents |
|---|---|
| `StableAudioKit` | Core: model registry, `GenerationOptions`, `GeneratedAudio` (WAV + post-processing), `RemoteEngine`, the engine protocol, `SATokenizer`, and the host-side sampler maths (`PingPongSampler`, `GaussianNoise`, `ClipPlan`) |
| `StableAudioKitCoreAI` | `CoreAIEngine` — the `.aimodel` exports through Core AI. No MLX. Its types are `@available(macOS 27, iOS 27)`; the package floor stays at 26 |
| `StableAudioKitMLX` | `MLXEngine` + the ported nets (`SADiT`, `SADecoder`, `SAT5Gemma`, `SAPipeline`) |
| `StableAudioKitGemma` | `GemmaTokenizer` (the T5Gemma tokenizer, via swift-transformers). Depends on the core only since 1.2.0, so Core AI users pull no MLX |

## Model registry

| Model | Kind | Max | On-device | Notes |
|---|---|---|---|---|
| `.sa3SmallSFX` ⭐ | SFX | 120s | ✅ Mac + iPhone | dedicated SFX, T5Gemma, ~1s/clip. ~3.5 GB fp16 / ~0.9 GB 8-bit |
| `.sa3SmallMusic` | music | 120s | ✅ Mac + iPhone | music sibling, identical architecture |
| `.sa3Medium` | general | 380s | Mac | top open quality (2B). ~2.4 GB converted; heavier decoder |
| `.legacyOpenSmall` | general | 12s | — | older architecture, cloud/reference only |
| `.legacyOpen1_0` | general | 47s | — | legacy full-diffusion, cloud/reference only |

`.all` = `[.sa3SmallSFX, .sa3SmallMusic, .sa3Medium]`.

## Usage

**On-device (Core AI, macOS 27 / iOS 27):**
```swift
import StableAudioKit
import StableAudioKitCoreAI
import StableAudioKitGemma

// An export folder from huggingface.co/arraypress/stable-audio-3-small-sfx-coreai:
// small-sfx.aimodel/ and tokenizer/.
let tokenizer = try await GemmaTokenizer(modelFolder: folder.appending(path: "tokenizer"))
let engine = try await CoreAIEngine(contentsOf: folder.appending(path: "small-sfx.aimodel"),
                                    model: .sa3SmallSFX, tokenizer: tokenizer)

let audio = try await engine.generate(
    prompt: "loud crackling campfire with crickets",
    options: GenerationOptions(seconds: 8, seed: 7))
try audio.wavData.write(to: url)

// Guidance and a negative prompt (each step then runs the transformer twice):
GenerationOptions(seconds: 8, cfgScale: 3, negativePrompt: "music, melody")
```

The first load of an asset in a program compiles it for the device and caches
the result (`~/Library/Caches/coreai-cache/<OS build>/<program>/`) — measured
below; later loads take seconds.

**On-device (MLX):**
```swift
import StableAudioKit
import StableAudioKitMLX
import StableAudioKitGemma

// Tokenizer files = the t5gemma-b-b-ul2 tokenizer.json/config placed in a folder.
let tokenizer = try await GemmaTokenizer(modelFolder: tokenizerDir)

let engine = MLXEngine(.sa3SmallSFX, tokenizer: tokenizer)
try engine.prepare(weightsFile: weightsURL)   // converted clean-keyed safetensors (fp16 or 8-bit)

let audio = try await engine.generate(
    prompt: "crackling campfire at night with crickets",
    options: GenerationOptions(seconds: 7))
try audio.wavData.write(to: url)              // lossless WAV, no extra deps
```

**Seeds:** the same prompt, options and seed give byte-identical audio. With no
seed the engine draws a random one (in `GenerationOptions.seedRange`,
0–4294967295) for each generation and reports it on the result, so any take
can be made again:

```swift
let audio = try await engine.generate(prompt: "loud metallic clang", options: .init(seconds: 3))
print(audio.seed!)   // e.g. 2918401177 — pass it back as GenerationOptions(seed:) to repeat it
```

Up to 1.0.0, no seed meant seed 0, so every unseeded run made the same sound.

**Which weights:** generation is deterministic for one weights file, so a
caller that saves settings to replay later should pin the file too.
`StableAudioWeights.sha256(of:)` digests it (an 8-bit and an fp32 conversion
of one revision make different audio), and `StableAudioWeights.source(of:)`
reads the upstream repository and revision `Tools/convert_stream.py
--source=REPO@REVISION` records in the safetensors header.

**Loops:**
```swift
// Texture loop (rain, wind, fire, drones) — equal-power tail→head crossfade
GenerationOptions(seconds: 8, loop: true)

// Musical loop — cropped to a whole number of bars at the given tempo,
// so the loop point lands on a downbeat (de-click crossfade only)
GenerationOptions(seconds: 6, loop: true, loopBPM: 138)
```

**Cloud (ElevenLabs):**
```swift
let gen = StableAudio(RemoteEngine(.elevenLabsSoundEffects(apiKey: key)))
let mp3 = try await (gen.engine as! RemoteEngine)
    .generateFile(prompt: "thunder rolling over a city", options: .init(seconds: 8))
try mp3.write(to: url)
```

## How the Core AI path works

The exports hold the networks as named functions — `condition`, `denoise` (any
length), and a ladder of fixed-length `decode_N` / `encode_N` — and everything
between the calls runs on the host, transcribed from the exports' own
`example.py`: the log-SNR schedule, the ping-pong sampler (required: Euler
clips), vanilla guidance blended on the host, the shortest decoder that holds
the latent (zero-padded, then trimmed). Clips are generated in a window of at
least 256 latent frames (23.8 s) — shorter windows give gross high-frequency
junk — with 3 s of margin past what is kept, and the conditioner is told the
window's length, as in training. Noise is a seeded xoshiro256** + Box–Muller
stream (`GaussianNoise`), so a seed repeats on every Mac; `render(…noise:)`
takes any `LatentNoise` for checks like the one below.

**Parity with `example.py`** (small-sfx, "loud crackling campfire with
crickets", 24 s window = 258 frames, seed 7, 8 steps, the same numpy noise fed
to both; `Tools/coreai_parity.py` + `sa-coreai parity`):

| Stage | Cosine | SNR | Max abs diff |
|---|---|---|---|
| Token ids, mask | 1 | identical | 0 |
| Conditioner (`cross_attn_cond`, `global_cond`) | 1.000000000 | identical | 0 |
| One DiT step (v at t₀) | 1.000000000 | identical | 0 |
| Decoder (on Python's final latent) | 1.000000000 | identical | 0 |
| **Final audio** | **1.000000000** | **identical** | **0** |
| Final audio, `--cfg 3` + negative prompt | 0.999999997 | 82.5 dB | 3.6e-5 |

Bit-identical without guidance: the host arithmetic runs in float64 between
network calls, as numpy does. With guidance the export's `guide` sees a float32
latent on the first step where this sees float64 — 82.5 dB apart.

**Measured on an M3 Max (36 GB), small-sfx**, one clip per process unless
noted:

| | Core AI (fp32 export) | MLX (fp32 conversion) |
|---|---|---|
| First load in a program | 15–20 min (compiles every function) | none |
| Compiled copy on disk | 6.5 GB per program | none |
| Later loads | 3–9 s | ~2 s |
| 2 s clip (23.8 s window) | 4–7 s for the first clip in a process, 1.0 s for each one after | 2.1 s (5 s window) |
| 30 s / 60 s / 120 s clip | 5.6 s / 6.2 s / 9.7 s | 1.0 s / 2.6 s / — |
| Peak memory, 2–120 s | **2.9–3.3 GB** (4.7 GB for a batch of three) | 4.5 GB (2 s) → 9.6 GB (60 s) |

Core AI's memory is flat because the decoder runs at fixed lengths and the
weights stay float32 on the GPU without a second copy; MLX grows with the
decoder's attention. The first clip in each process pays a few seconds of
warm-up (mostly the first decode); a batch amortises it.

## How the MLX path works

The full Stable Audio 3 pipeline was reverse-engineered from `stable-audio-tools`, reimplemented in MLX-Python and numerically matched to PyTorch (decoder 2.7e-5, DiT 1.1e-4, end-to-end audio 8.7e-4), then transcribed 1:1 to MLX-Swift and **runtime-validated on the Metal GPU**. One parameterised codebase (`SADiT`/`SADecoder` infer dims from the weights) runs small **and** medium.

- **End-of-clip artifact:** the model puts a tone on its final latent frame, so generation renders a few seconds of margin and crops back — the artifact lands in the discarded tail, not the audible clip.
- **iPhone:** 8-bit quantization (`quant.bits` metadata) is near-lossless and dequantizes to fp16 at load. 4-bit is too aggressive for this diffusion model.
- **Weights** are converted to a single clean-keyed safetensors with `Tools/convert_stream.py` (numpy only; `--source=REPO@REVISION` records where they came from); `MLXEngine` splits by prefix (`dit.`/`dec.`/`t5.`) and builds the pipeline. fp32 output loads as-is; 8-bit files (`quant.bits`) dequantize at load.

## Executables

`sa-coreai` (Core AI: `info`, `generate`, `parity`), `sa-audio`, `sa-generate` (text→WAV), `sa-validate` (net-vs-fixture GPU check), `sa-tokenize`. See `Package.swift`.

> **Note:** a plain `swift build` puts `mlx-swift_Cmlx.bundle` (holding MLX's `default.metallib`) beside each binary, and MLX loads it from there — ship the binary together with that bundle. No Xcode and no Python `mlx.metallib` needed (measured 2026-09-03, and again 2026-10-15 with `sfx`).

## Status
- ✅ **Core + `RemoteEngine`** — built, 31 offline unit tests (`swift test`; no GPU, no weights).
- ✅ **`CoreAIEngine`** (1.2.0) — the published Core AI exports, bit-identical to their `example.py`; what `sfx` 0.3.0 runs.
- ✅ **`MLXEngine`** — all three SA3 models generate end-to-end, runtime-validated on GPU; loops, margin-crop, and 8-bit quantization landed.
- ✅ **`GemmaTokenizer`** — byte-identical token ids to the reference.
- 🔜 **Audio in** — continuation and inpainting through the exports' `encode_N` functions (`example.py --init-audio`); `render` already takes the `local_add_cond` it needs.
- 🔜 **MLX weight download** (today: a local converted safetensors). The Core AI exports are on Hugging Face and `sfx` fetches them.

## Requirements
- macOS 26+ / iOS 26+ · Swift 6.2+ (`CoreAIEngine`: macOS 27 / iOS 27)
- `CoreAIEngine`: an export folder from Hugging Face (`.aimodel` + `tokenizer/`; Stability AI Community License + Gemma Terms).
- `MLXEngine`: converted Stable Audio weights + the T5Gemma tokenizer files (Stability Community License + Gemma Terms).
- `RemoteEngine`: a provider API key.

## Testing
```bash
swift test
```

## License
MIT — see LICENSE. (Model weights carry their own licenses.)

## Author
Created by David Sherlock ([ArrayPress](https://github.com/arraypress)) in 2026.
