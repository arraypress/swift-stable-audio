# MLXEngine — on-device plan

The on-device engine for `StableAudioKit`, backed by **MLX** (Mac + iPhone GPU). This is the real work behind the library; everything here is **validated in Python** — what remains is the MLX-Swift inference.

## Status
- ✅ **`StableAudioKitMLX` target set up** — depends on `mlx-swift` 0.31.6 (MLX/MLXNN/MLXRandom); compiles.
- ✅ **`MLXEngine`** conforms to `AudioGenEngine`, loads `.safetensors` shards into `[String: MLXArray]` via `prepare(weightsDirectory:)`.
- 🔜 **Hugging Face download** in `prepare()` (gated → token).
- 🔜 **Forward passes** — Conditioner (T5Gemma), DiT (20-layer differential-attn), Autoencoder decoder, ping-pong sampler. Port from `stable_audio_tools` (PyTorch) using the MLX weight map below. Validate each stage numerically against the Python run.

### Weight namespaces (from `mlx-community/stable-audio-3-small-sfx`, 682 tensors)
- `model.model.transformer.layers.{0..19}.` — DiT: `self_attn.{to_qkv,to_out,q_norm,k_norm}`, `cross_attn.{to_q,to_kv,to_out,q_norm,k_norm,cross_attend_norm}`, `ff.ff.*`, `pre_norm`, `ff_norm`, `to_local_embed`
- `pretransform.model.{encoder,decoder}.layers.*.transformers.*.` — autoencoder (transformer-based)
- config: `latent_dim 256`, `transformer_depths [6]`, `dim_heads 64`, `dyt true`, `differential true`, `variable_stride`, `chunk_size 32`, `model_type stable_audio_3`

## Why MLX (not LiteRT)
- One runtime across the ecosystem (`swift-tts`, `swift-llm` already use MLX).
- Stable Audio 3 already has community MLX **weight** ports (`mlx-community/stable-audio-3-small-sfx`, `aufklarer/Stable-Audio-3-DiT-*-MLX-*bit`).
- The diffusion sampling loop is dynamic — natural in MLX, awkward in a static LiteRT graph.
- LiteRT stays a proven **fallback** (Arm/Stability official CPU path; Open Small export verified — 4 `.tflite`, loads on XNNPACK).

> Caveat: the MLX ports are **converted weights**, not a ready Swift runtime. `mlx-audio-swift` is speech-only, so we implement the inference ourselves.

## Pipeline (verified against `stable-audio-3-small-sfx`)
Three sub-models + a sampling loop (the same decomposition the LiteRT export produced):

```
prompt ──▶ [T5Gemma conditioner]  ──▶ cross_attn_cond [1,65,768] + global_cond [1,768]
                                          │
             x = noise [1,64,256] ──▶ ┌── loop ×8 (ping-pong) ───────────────┐
                                      │  denoised = [DiT](x, t, conds)        │
                                      │  x = pingpong_step(x, denoised, t)    │
                                      └───────────────────────────────────────┘
                                          │
                                   [autoencoder decoder] ──▶ stereo audio [2, N]
```

### Verified I/O signatures (from the LiteRT export of Open Small; SA3 is analogous)
- **conditioners**: `tokens[1,128] i64 + mask[1,128] i64 + seconds[1] f32` → `cross_attn[1,65,768] + global[1,768]`
- **DiT** (int8-friendly): `x[1,64,256] + t[1] + cross_attn[1,65,768] + global[1,768]` → `denoised[1,64,256]`
- **decoder**: `latent[1,64,256]` → `audio[2, 524288]`  (Open Small = 12s; SA3 latent is longer, up to 120s)

### Sampler (distilled models)
`steps = 8`, `cfg_scale = 1.0`, **ping-pong** sampler. Reference: Arm's `audiogen.cpp` (LiteRT) and stable-audio-tools' `inference/sampling.py` (`sample`/pingpong).

## Work breakdown
1. **Package target** `StableAudioKitMLX` depending on `mlx-swift` (MLX, MLXNN, MLXRandom). Conforms `MLXEngine: AudioGenEngine`.
2. **Weights + download** — resolve the MLX port from the model's `repoID`; download via Hugging Face (gated → token). Map to MLX arrays.
3. **T5Gemma tokenizer + conditioner** in MLX (or reuse a tokenizer lib) → `cross_attn_cond`, `global_cond`.
4. **DiT forward** in MLX — transformer blocks + rotary (real-valued; the `float64→float32` fixes from the Python run apply). Quantized (4/8-bit) weights supported natively.
5. **Ping-pong sampling loop** — 8 steps, cfg 1.0.
6. **Autoencoder decoder** in MLX → interleave to `GeneratedAudio`.
7. **`prepare()`** = download + load; **`generate()`** = tokenize → conditioners → loop → decode.

## Reference material (in the prototype)
- Python generators: `scratchpad/moss/stableaudio/gen_sao.py` (Open Small), `gen_sa3.py` (SA3 small-sfx), `gen_big.py` (Open 1.0).
- LiteRT export + I/O dump: `scratchpad/moss/ML-examples/kleidiai-examples/audiogen/` (`export_sao.py`, `app/audiogen.cpp`).
- Measured on M3 Pro / MPS: SA3 small-sfx & Open Small ≈ **1s/clip**; Open 1.0 ≈ 3 min/clip.
