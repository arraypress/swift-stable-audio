# StableAudioKit

One Swift API for **text-to-audio** — sound effects and music — across tiers: **on-device** (Stable Audio, via MLX) or **cloud** (ElevenLabs / Stability). Program against one protocol; the model (and the tier: on-device / most-capable) becomes a swappable choice, not a rewrite.

For macOS and iOS. Same shape as [LLMKit](https://github.com/arraypress/swift-llm) and the TTS library — a model registry + engines behind one protocol.

## Engines

| Engine | Tier | What it is |
|---|---|---|
| **`RemoteEngine`** | cloud | any HTTP text-to-audio endpoint — **ElevenLabs Sound Effects**, Stability API, or your own proxy |
| **`MLXEngine`** *(next)* | on-device, downloaded | Stable Audio 3 / Open Small via MLX (Mac + iPhone) |

## Model registry

Pick a tier per app, like `MLXModel` in LLMKit:

| Model | Kind | Max | On-device | Notes |
|---|---|---|---|---|
| `.sa3SmallSFX` ⭐ | SFX | 120s | ✅ | dedicated SFX, T5Gemma, ~1s/clip |
| `.sa3SmallMusic` | music | 120s | ✅ | music sibling |
| `.openSmall` | general | 12s | ✅ | tiny (~1GB), proven LiteRT export too |
| `.sa3Medium` | general | 120s | ✗ (Mac) | top open quality (~10GB) |
| `.open1_0` | general | 47s | ✗ (Mac) | legacy, full diffusion (~3 min/clip) |

## Usage

**Cloud (ElevenLabs Sound Effects):**
```swift
import StableAudioKit

let gen = StableAudio(RemoteEngine(.elevenLabsSoundEffects(apiKey: key)))
let mp3 = try await (gen.engine as! RemoteEngine)
    .generateFile(prompt: "thunder rolling over a city", options: .init(seconds: 8))
try mp3.write(to: url)     // ElevenLabs returns MP3; decode with AVFoundation to play
```

**On-device (MLX — planned):**
```swift
let gen = StableAudio(MLXEngine(.sa3SmallSFX))
try await gen.prepare()                       // downloads + loads on first use
let audio = try await gen.generate("crackling campfire at night", seconds: 7)
try audio.wavData.write(to: url)              // lossless WAV, no extra deps
```

## Status
- ✅ **Core + `RemoteEngine` — built and unit-tested (14 tests).** Model registry, `GenerationOptions` resolution, `GeneratedAudio` WAV round-trip, and the cloud request shaping are covered by pure offline tests.
- 🔜 **`MLXEngine`** — on-device Stable Audio via MLX. See [`MLX_ENGINE_PLAN.md`](MLX_ENGINE_PLAN.md). The models, sampler settings, and pipeline are validated in Python; the work is the MLX-Swift inference.

## Requirements
- macOS 26+ / iOS 26+ · Swift 6.2+
- `RemoteEngine` needs a provider API key. `MLXEngine` (when landed) downloads gated Stability weights (accept the license on Hugging Face).

## Testing
```bash
swift test
```

## License
MIT — see LICENSE.

## Author
Created by David Sherlock ([ArrayPress](https://github.com/arraypress)) in 2026.
