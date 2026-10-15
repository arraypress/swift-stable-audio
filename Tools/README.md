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

Converts a HuggingFace snapshot of any Stable Audio 3 model (`stabilityai/stable-audio-3-medium`, `-small-sfx`, `-small-music`) into
the MLX-format safetensors that `MLXEngine` loads.

```bash
python3 Tools/convert_stream.py <snapshot_dir> <out.safetensors> [--dry-run] [--source=REPO@REVISION]
```

Streams tensor-by-tensor off disk, so peak memory is one tensor — 786 MB, the
T5 embedding table — rather than roughly twice the model size. The
`get_pretrained_model()` route it replaced needed about 20 GB and is what
crashed the Mac. Medium converts in under 20 seconds: 880 tensors, 8.64 GB
fp32.

`--dry-run` reports tensor count, output size and peak tensor without writing,
which is the cheap way to confirm a snapshot is intact.

`--source=REPO@REVISION` records where the snapshot came from in the output's
`__metadata__` (`source_repository`, `source_revision`);
`StableAudioWeights.source(of:)` reads it back, and `sfx` writes it into every
recipe. Any SA3 snapshot converts — small-sfx is 688 tensors, 3.18 GB fp32:

```bash
python3 Tools/convert_stream.py <snapshot> sa3-small-sfx.safetensors \
    --source=stabilityai/stable-audio-3-small-sfx@05ca3857881c94a2b84026b1111c537d78f5f9f0
```

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

## `quantize.py` — recovered, and not the one that made the weights

Recovered from a session transcript on 2026-08-02; the original lived in a
scratchpad and was wiped. **It produces 4-bit, and the shipped weights are
8-bit.** CLAUDE.md records 4-bit as audibly degraded for this model, and the
script omits the `quant.bits` / `quant.group` tensors `MLXEngine` requires, so
its output will not load correctly as-is. It also needs `mlx` and
`safetensors`, which the other tools here deliberately do not.

Read its docstring before running it. Reconstituting the 8-bit production
quantizer is three small changes plus an A/B against an fp32 reference — until
that is done, the `sa3-*.safetensors` files cannot be regenerated and should
be treated as irreplaceable.
