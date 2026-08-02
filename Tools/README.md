# Tools

Python helpers for preparing weights and inspecting generated audio. Nothing
here is built or shipped — SwiftPM only scans `Sources/` and `Tests/`, so this
directory is invisible to the package.

**numpy and the standard library only.** No torch, no stable-audio-tools, no
virtualenv. That is deliberate: the environment these replaced was destroyed
by a machine crash on 2026-07-25, and anything needing a venv is a thing that
can rot again.

```bash
python3 Tools/<script>.py            # every script prints its own usage
```

## `convert_stream.py`

Converts a HuggingFace snapshot of `stabilityai/stable-audio-3-medium` into
the MLX-format safetensors that `MLXEngine` loads.

```bash
python3 Tools/convert_stream.py <snapshot_dir> <out.safetensors> [--dry-run]
```

Streams tensor-by-tensor off disk, so peak memory is one tensor — 786 MB, the
T5 embedding table — rather than roughly twice the model size. The
`get_pretrained_model()` route it replaced needed about 20 GB and is what
crashed the Mac. Medium converts in under 20 seconds: 880 tensors, 8.64 GB
fp32.

`--dry-run` reports tensor count, output size and peak tensor without writing,
which is the cheap way to confirm a snapshot is intact.

The output key names are a contract with `MLXEngine` — it looks tensors up by
exact string, so a rename here is a silent load failure there. Two traps are
documented inline: the six front-of-decoder tensors need a `dec.` prefix, and
`num_w` / `num_b` / `padding_embedding` must **not** have one.

## Audio inspection

All three read 16-bit PCM WAVs and refuse anything else rather than reporting
confident nonsense.

| Script | Question it answers |
|---|---|
| `wavchk.py` | Did that generate sensibly? Duration, rate, channels, peak, RMS. |
| `crest.py` | How peaky is it? Ranks a glob by crest factor (peak minus RMS). |
| `tempo.py` | What tempo actually came out? For checking `loopBPM` crops. |

```bash
python3 Tools/wavchk.py out/*.wav
python3 Tools/crest.py 'out/*.wav'    # quote the glob — one argument
python3 Tools/tempo.py out/*.wav
```

`tempo.py`'s confidence figure is **not** a "is there a beat" score — see its
docstring. A beatless pad measures higher than a real drum pattern.

## Missing: the quantizer

The `sa3-*.safetensors` weights in `~/StableAudioModels/` are 8-bit quantized
(`quant.bits = 8`, `quant.group = 64`), but `convert_stream.py` only emits
fp32. **The tool that quantized them no longer exists** — not here, not in
`Sources/`, not on disk. `MLXEngine` dequantizes at load and has no quantize
path.

So those weight files cannot currently be regenerated from the checkpoint, and
should be treated as irreplaceable until a quantizer is written. It was most
likely lost with the other scratchpad scripts in the 2026-07-25 crash; if so,
it is recoverable from the session transcripts of that work.
