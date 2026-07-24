# StableAudioKit

One Swift API for **text-to-audio** — sound effects and music — across tiers: **on-device** (Stable Audio 3, via MLX) or **cloud** (ElevenLabs / Stability). Program against one protocol; the model and the tier become a swappable choice, not a rewrite.

For macOS and iOS. Same shape as [LLMKit](https://github.com/arraypress/swift-llm) and the TTS library — a model registry + engines behind one protocol.

## Engines

| Engine | Tier | What it is |
|---|---|---|
| **`MLXEngine`** | on-device | Stable Audio 3 (small-sfx / small-music / medium) via MLX — Mac + iPhone. Runs the full ported pipeline: T5Gemma → DiT ×8 ping-pong → decoder. |
| **`RemoteEngine`** | cloud | any HTTP text-to-audio endpoint — **ElevenLabs Sound Effects/Music**, Stability API, or your own proxy |

## Products

| Product | Contents |
|---|---|
| `StableAudioKit` | Core: model registry, `GenerationOptions`, `GeneratedAudio` (WAV + post-processing), `RemoteEngine`, the engine protocol |
| `StableAudioKitMLX` | `MLXEngine` + the ported nets (`SADiT`, `SADecoder`, `SAT5Gemma`, `SAPipeline`) |
| `StableAudioKitGemma` | `GemmaTokenizer` (the T5Gemma tokenizer, via swift-transformers) |

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

## How the on-device path works

The full Stable Audio 3 pipeline was reverse-engineered from `stable-audio-tools`, reimplemented in MLX-Python and numerically matched to PyTorch (decoder 2.7e-5, DiT 1.1e-4, end-to-end audio 8.7e-4), then transcribed 1:1 to MLX-Swift and **runtime-validated on the Metal GPU**. One parameterised codebase (`SADiT`/`SADecoder` infer dims from the weights) runs small **and** medium.

- **End-of-clip artifact:** the model puts a tone on its final latent frame, so generation renders a few seconds of margin and crops back — the artifact lands in the discarded tail, not the audible clip.
- **iPhone:** 8-bit quantization (`quant.bits` metadata) is near-lossless and dequantizes to fp16 at load. 4-bit is too aggressive for this diffusion model.
- **Weights** are converted to a single clean-keyed safetensors via `scratchpad/.../convert_weights.py`; `MLXEngine` splits by prefix (`dit.`/`dec.`/`t5.`) and builds the pipeline.

## Executables

`sa-tts`, `sa-generate` (text→WAV), `sa-validate` (net-vs-fixture GPU check), `sa-tokenize`. See `Package.swift`.

> **Note:** MLX-Swift needs a Metal library that only ships in Xcode **app** builds. To run the executables headless, colocate `mlx.metallib` next to the binary (see `CLAUDE.md`). The companion **StableAudioTester** Mac app runs the engine with the metallib bundled automatically.

## Status
- ✅ **Core + `RemoteEngine`** — built, 14 offline unit tests.
- ✅ **`MLXEngine`** — all three SA3 models generate end-to-end, runtime-validated on GPU; loops, margin-crop, and 8-bit quantization landed.
- ✅ **`GemmaTokenizer`** — byte-identical token ids to the reference.
- 🔜 **HF weight download in `prepare()`** (today: load a local converted safetensors) and app-side weight delivery (Background Assets).

## Requirements
- macOS 26+ / iOS 26+ · Swift 6.2+
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
