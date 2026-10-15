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

## How the on-device path works

The full Stable Audio 3 pipeline was reverse-engineered from `stable-audio-tools`, reimplemented in MLX-Python and numerically matched to PyTorch (decoder 2.7e-5, DiT 1.1e-4, end-to-end audio 8.7e-4), then transcribed 1:1 to MLX-Swift and **runtime-validated on the Metal GPU**. One parameterised codebase (`SADiT`/`SADecoder` infer dims from the weights) runs small **and** medium.

- **End-of-clip artifact:** the model puts a tone on its final latent frame, so generation renders a few seconds of margin and crops back — the artifact lands in the discarded tail, not the audible clip.
- **iPhone:** 8-bit quantization (`quant.bits` metadata) is near-lossless and dequantizes to fp16 at load. 4-bit is too aggressive for this diffusion model.
- **Weights** are converted to a single clean-keyed safetensors with `Tools/convert_stream.py` (numpy only; `--source=REPO@REVISION` records where they came from); `MLXEngine` splits by prefix (`dit.`/`dec.`/`t5.`) and builds the pipeline. fp32 output loads as-is; 8-bit files (`quant.bits`) dequantize at load.

## Executables

`sa-tts`, `sa-generate` (text→WAV), `sa-validate` (net-vs-fixture GPU check), `sa-tokenize`. See `Package.swift`.

> **Note:** a plain `swift build` puts `mlx-swift_Cmlx.bundle` (holding MLX's `default.metallib`) beside each binary, and MLX loads it from there — ship the binary together with that bundle. No Xcode and no Python `mlx.metallib` needed (measured 2026-09-03, and again 2026-10-15 with `sfx`).

## Status
- ✅ **Core + `RemoteEngine`** — built, 21 offline unit tests (`swift test`; no GPU, no weights).
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
