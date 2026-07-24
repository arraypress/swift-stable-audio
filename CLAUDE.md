# CLAUDE.md — swift-stable-audio

On-device text-to-audio (SFX + music) for macOS/iOS. Module: **StableAudioKit**. MLX-first, one API + a model registry + engines behind a protocol (same shape as swift-llm / swift-tts).

## Build & test
```bash
swift build                     # all products
swift test                      # 14 offline unit tests (core only — no GPU/MLX)
swift build --product sa-tts    # typed prompt → WAV
```

## The MLX metallib gotcha (READ THIS FIRST)
MLX-Swift can't run via `swift build`/`swift test` here — the Metal library only bundles in Xcode **app** builds. To run any executable (`sa-tts`, `sa-generate`, `sa-validate`) headless, colocate the metallib next to the binary:
```bash
cp <mlx-python>/.venv/lib/python3.12/site-packages/mlx/lib/mlx.metallib \
   .build/arm64-apple-macosx/debug/mlx.metallib
```
The MLX-Python venv lives at `scratchpad/moss/stableaudio/.venv` (path recorded in the project memory). Xcode app builds (the StableAudioTester companion) bundle the metallib automatically — nothing to do there.

## Module map
- `StableAudioKit/` — `StableAudioModel` (registry), `GenerationOptions`, `GeneratedAudio` (WAV + `trimmingEnd`/`faded`/`seamlessLoop`), `RemoteEngine`, `AudioGenEngine` protocol.
- `StableAudioKitMLX/` — `MLXEngine` + ported nets: `SADiT`, `SADecoder`, `SAT5Gemma`, `SAPipeline` (ping-pong sampler), `MLXOps` (shared primitives). `SADiT`/`SADecoder` **infer dims from the weights** → one codebase runs small + medium.
- `StableAudioKitGemma/` — `GemmaTokenizer` (swift-transformers `AutoTokenizer`). App injects it via the `SATokenizer` protocol.

## MLX port workflow (how the nets were built/validated)
Reverse-engineered from `stable-audio-tools` (the authoritative arch ref). Each net was implemented in **MLX-Python**, matched to PyTorch against golden fixtures, then transcribed 1:1 to MLX-Swift and runtime-checked on GPU via `sa-validate`. Reference impls + fixtures: `scratchpad/moss/stableaudio/mlx_{t5,dit,decoder,sampler,pipeline}.py`, `fixtures/`. Weight conversion: `convert_weights.py <hf-repo> <out.safetensors>` → clean-keyed single file; `MLXEngine` splits by prefix (`dit.`/`dec.`/`t5.`).

## Behaviours worth knowing
- **End-of-clip tone:** the model marks its final latent frame. `MLXEngine` generates `seconds + 3s` of frames and crops back to `seconds` so the artifact is discarded (mirrors the reference pipeline). Do NOT "fix" this by trimming/fading the output — that eats real content.
- **Loops:** `options.loop` → equal-power texture crossfade; `options.loopBPM` → crop to a whole number of bars at that tempo (downbeat-aligned) + 12ms de-click. The model isn't tempo-locked, so bar-cropping is the safety net, not a guarantee.
- **Quantization:** 8-bit (`quant.bits` metadata) is near-lossless and dequantizes to fp16 at load; 4-bit is audibly degraded for this diffusion model — don't ship it.

## Local assets (not in the repo)
- Converted weights + tokenizer: `~/StableAudioModels/`.
- Companion app: `../../swift-stable-audio-tester` (xcodegen; `xcodegen generate && xcodebuild ...`).

## Conventions
- Commit only when asked; branch off `main` first if needed.
- Weights/fixtures/venv stay OUT of git (they're large and live in scratchpad / `~/StableAudioModels`).
