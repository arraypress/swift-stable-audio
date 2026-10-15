"""Convert a Stable Audio 3 checkpoint into MLX-format safetensors.

Reads the checkpoint tensor-by-tensor straight off disk rather than
instantiating the torch model. Peak memory is therefore one tensor — 786 MB,
the T5 embedding table — instead of roughly twice the model size. The
`get_pretrained_model()` route this replaced needed about 20 GB and crashed
the machine.

Dependencies are numpy and the standard library. No torch, no
stable-audio-tools, no virtualenv.

    python3 convert_stream.py <snapshot_dir> <out.safetensors> [--dry-run]
                              [--source=stabilityai/stable-audio-3-small-sfx@<revision>]

`snapshot_dir` is a HuggingFace snapshot of any Stable Audio 3 model,
containing `model.safetensors` and `t5gemma-b-b-ul2/model.safetensors`.
`--dry-run` reports the tensor count, output size and peak tensor without
writing anything, which is the cheap way to confirm a checkpoint is intact.
`--source=REPO@REVISION` records where the checkpoint came from in the
file's `__metadata__` (`source_repository`, `source_revision`), so a tool
can name the exact upstream revision its audio came from — sfx writes it
into every recipe.

Medium converts in under 20 seconds: 880 tensors, 8.64 GB fp32.

The output feeds `MLXEngine` in StableAudioKitMLX. Its key names are a
contract with that engine — see `build_entries()` before changing any of them.
"""

from __future__ import annotations

import json
import os
import struct
import sys
from typing import Callable

import numpy as np

# --------------------------------------------------------------- safetensors

# A safetensors file is: u64 header length, then that many bytes of JSON
# describing every tensor, then the raw tensor data concatenated.

Producer = Callable[[], np.ndarray]
Entry = tuple[str, tuple[int, ...], Producer]


def _read_header(path: str) -> tuple[dict, int]:
    """Return the parsed JSON header and the byte offset where data starts."""
    with open(path, "rb") as f:
        length = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(length)), 8 + length


class Reader:
    """Random access to one safetensors file, one tensor at a time.

    Nothing is held open between reads and no tensor is cached, so memory
    stays flat however many times the file is queried.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.hdr, self.base = _read_header(path)

    def __contains__(self, key: str) -> bool:
        return key in self.hdr

    def shape(self, key: str) -> tuple[int, ...]:
        return tuple(self.hdr[key]["shape"])

    def read(self, key: str) -> np.ndarray:
        """Read one tensor and widen it to float32.

        Every dtype in these checkpoints becomes f32 on the way out, because
        that is what the MLX engine loads. T5Gemma ships as BF16, which numpy
        has no native type for — a bf16 value is the top 16 bits of the f32
        with the same value, so shifting left by 16 into a u32 and
        reinterpreting is exact rather than an approximation.
        """
        entry = self.hdr[key]
        start, end = entry["data_offsets"]

        with open(self.path, "rb") as f:
            f.seek(self.base + start)
            buf = f.read(end - start)

        dtype = entry["dtype"]

        if dtype == "F32":
            a = np.frombuffer(buf, dtype="<f4")
        elif dtype == "BF16":
            a = (np.frombuffer(buf, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
        elif dtype == "F16":
            a = np.frombuffer(buf, dtype="<f2").astype(np.float32)
        else:
            raise ValueError(f"unhandled dtype {dtype} for {key}")

        return a.reshape(entry["shape"])


def write_stream(out_path: str, entries: list[Entry], metadata: dict[str, str] | None = None) -> int:
    """Write tensors to a safetensors file one at a time.

    The header only needs each tensor's name, dtype and shape, all of which
    are known up front — so it can be written before any tensor is produced.
    That is what allows each one to be read, converted, flushed and freed in
    turn, rather than assembling the whole model in memory first.

    Args:
        out_path: Destination file.
        entries: Ordered `(name, shape, producer)`, where `producer()` returns
            the float32 array for that tensor.
        metadata: Strings for the header's `__metadata__` — where the weights
            came from, which `StableAudioWeights.source(of:)` reads back.

    Returns:
        Total bytes of tensor data written, excluding the header.
    """
    offset, meta = 0, {}
    if metadata:
        meta["__metadata__"] = dict(metadata)

    for name, shape, _ in entries:
        nbytes = int(np.prod(shape)) * 4
        meta[name] = {
            "dtype": "F32",
            "shape": list(shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes

    blob = json.dumps(meta, separators=(",", ":")).encode()

    # The spec requires the data section to start 8-byte aligned.
    blob += b" " * ((-len(blob)) % 8)

    with open(out_path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)

        for i, (name, shape, produce) in enumerate(entries):
            a = np.ascontiguousarray(produce(), dtype=np.float32)
            assert a.shape == tuple(shape), f"{name}: {a.shape} != {shape}"
            f.write(a.tobytes())
            del a

            if (i + 1) % 100 == 0:
                print(f"    ... {i + 1}/{len(entries)} tensors", flush=True)

    return offset


# ---------------------------------------------------------------- key mapping


def build_entries(main: Reader, t5: Reader) -> tuple[list[Entry], int, int, int]:
    """Map checkpoint keys onto the names MLXEngine expects.

    Key-for-key equivalent to the validated `convert_weights.py`. The output
    names are a contract with `MLXEngine`: it looks tensors up by these exact
    strings, so a rename here is a silent load failure there.

    Layer counts are derived from the checkpoint rather than hardcoded, so a
    different model size converts without edits.

    Returns:
        `(entries, decoder_blocks, dit_layers, t5_layers)`.
    """
    ent: list[Entry] = []

    def put(out: str, src: Reader, key: str) -> None:
        """Queue one tensor, copied across unchanged."""
        # `s=src, k=key` binds the current values — a bare closure over the
        # loop variables would make every producer read the last key.
        ent.append((out, src.shape(key), lambda s=src, k=key: s.read(k)))

    def lin(out: str, src: Reader, base: str) -> None:
        """Queue a linear layer's weight, and its bias when it has one."""
        put(out + ".w", src, base + ".weight")
        if base + ".bias" in src:
            put(out + ".b", src, base + ".bias")

    # ---- Decoder: weight-normed mapping conv --------------------------------

    D = "pretransform.model.decoder"
    R = f"{D}.layers.3"

    def map_weight() -> np.ndarray:
        """Collapse PyTorch weight normalisation into a plain kernel.

        Weight norm stores a direction `v` and a magnitude `g` separately and
        recombines them as `g * v / ||v||` at forward time. MLX has no
        equivalent, so it is folded once here.
        """
        g = main.read(f"{R}.mapping.weight_g")
        v = main.read(f"{R}.mapping.weight_v")
        return g * v / np.linalg.norm(v, axis=(1, 2), keepdims=True)

    # These six front-of-decoder tensors need the `dec.` prefix even though
    # they sit outside any numbered block: MLXEngine routes the decoder dict
    # through sub("dec.") and will not find them bare. By contrast `num_w`,
    # `num_b` and `padding_embedding` at the bottom of this function stay
    # unprefixed. Getting this wrong loads a model that runs and produces
    # silence.
    ent.append(("dec.map_weight", main.shape(f"{R}.mapping.weight_v"), map_weight))
    put("dec.map_bias", main, f"{R}.mapping.bias")
    put("dec.new_tokens", main, f"{R}.new_tokens")
    put("dec.lin_w", main, f"{D}.layers.1.weight")
    put("dec.lin_b", main, f"{D}.layers.1.bias")
    put("dec.running_std", main, "pretransform.model.bottleneck.running_std")

    n_dec = len({k.split(".")[6] for k in main.hdr if k.startswith(f"{R}.transformers.")})

    for i in range(n_dec):
        b, p = f"{R}.transformers.{i}", f"dec.L{i}."

        for tag, sub in (("prenorm", "pre_norm"), ("ffnorm", "ff_norm")):
            for w in ("alpha", "gamma", "beta"):
                put(p + f"{tag}_{w}", main, f"{b}.{sub}.{w}")

        put(p + "to_qkv", main, f"{b}.self_attn.to_qkv.weight")
        put(p + "to_out", main, f"{b}.self_attn.to_out.weight")

        for qk in ("q", "k"):
            for w in ("alpha", "gamma", "beta"):
                put(p + f"{qk}_{w}", main, f"{b}.self_attn.{qk}_norm.{w}")

        put(p + "glu_proj_w", main, f"{b}.ff.ff.0.proj.weight")
        put(p + "glu_proj_b", main, f"{b}.ff.ff.0.proj.bias")
        put(p + "ff_out_w", main, f"{b}.ff.ff.2.weight")
        put(p + "ff_out_b", main, f"{b}.ff.ff.2.bias")

    # ---- DiT ----------------------------------------------------------------

    M, T = "model.model", "model.model.transformer"

    lin("dit.t_embed0", main, f"{M}.to_timestep_embed.0")
    lin("dit.t_embed2", main, f"{M}.to_timestep_embed.2")
    lin("dit.cond_embed0", main, f"{M}.to_cond_embed.0")
    lin("dit.cond_embed2", main, f"{M}.to_cond_embed.2")
    lin("dit.glob_embed0", main, f"{M}.to_global_embed.0")
    lin("dit.glob_embed2", main, f"{M}.to_global_embed.2")
    lin("dit.gce0", main, f"{T}.global_cond_embedder.0")
    lin("dit.gce2", main, f"{T}.global_cond_embedder.2")
    put("dit.memory_tokens", main, f"{T}.memory_tokens")
    lin("dit.project_in", main, f"{T}.project_in")
    lin("dit.project_out", main, f"{T}.project_out")
    put("dit.pre_conv", main, f"{M}.preprocess_conv.weight")
    put("dit.post_conv", main, f"{M}.postprocess_conv.weight")

    n_dit = len({k.split(".")[4] for k in main.hdr if k.startswith(f"{T}.layers.")})

    for i in range(n_dit):
        b, p = f"{T}.layers.{i}", f"dit.L{i}."

        put(p + "ssg", main, f"{b}.to_scale_shift_gate")
        put(p + "prenorm_g", main, f"{b}.pre_norm.gamma")
        lin(p + "sa_qkv", main, f"{b}.self_attn.to_qkv")
        lin(p + "sa_out", main, f"{b}.self_attn.to_out")
        put(p + "sa_qn", main, f"{b}.self_attn.q_norm.gamma")
        put(p + "sa_kn", main, f"{b}.self_attn.k_norm.gamma")
        put(p + "xnorm_g", main, f"{b}.cross_attend_norm.gamma")
        lin(p + "ca_q", main, f"{b}.cross_attn.to_q")
        lin(p + "ca_kv", main, f"{b}.cross_attn.to_kv")
        lin(p + "ca_out", main, f"{b}.cross_attn.to_out")
        put(p + "ca_qn", main, f"{b}.cross_attn.q_norm.gamma")
        put(p + "ca_kn", main, f"{b}.cross_attn.k_norm.gamma")
        put(p + "ffnorm_g", main, f"{b}.ff_norm.gamma")
        lin(p + "glu", main, f"{b}.ff.ff.0.proj")
        lin(p + "ff_out", main, f"{b}.ff.ff.2")
        lin(p + "loc0", main, f"{b}.to_local_embed.0")
        lin(p + "loc2", main, f"{b}.to_local_embed.2")

    # ---- T5Gemma text encoder (separate file, BF16 -> F32) ------------------

    put("t5.embed_tokens", t5, "model.encoder.embed_tokens.weight")
    put("t5.final_norm", t5, "model.encoder.norm.weight")

    n_t5 = len({k.split(".")[3] for k in t5.hdr if k.startswith("model.encoder.layers.")})

    for i in range(n_t5):
        b, p = f"model.encoder.layers.{i}", f"t5.L{i}."

        for o, s in (("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj"), ("o", "o_proj")):
            put(p + o, t5, f"{b}.self_attn.{s}.weight")

        put(p + "pre_attn", t5, f"{b}.pre_self_attn_layernorm.weight")
        put(p + "post_attn", t5, f"{b}.post_self_attn_layernorm.weight")

        for o in ("gate", "up", "down"):
            put(p + o, t5, f"{b}.mlp.{o}_proj.weight")

        put(p + "pre_ff", t5, f"{b}.pre_feedforward_layernorm.weight")
        put(p + "post_ff", t5, f"{b}.post_feedforward_layernorm.weight")

    # ---- NumberConditioner and prompt padding -------------------------------

    # Deliberately unprefixed — see the note on the `dec.` tensors above.
    C = "conditioner.conditioners"
    put("num_w", main, f"{C}.seconds_total.embedder.embedding.1.weight")
    put("num_b", main, f"{C}.seconds_total.embedder.embedding.1.bias")
    put("padding_embedding", main, f"{C}.prompt.padding_embedding")

    return ent, n_dec, n_dit, n_t5


def main_cli() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry = "--dry-run" in sys.argv
    metadata: dict[str, str] = {}
    for flag in sys.argv[1:]:
        if flag.startswith("--source="):
            repo, _, revision = flag[len("--source="):].partition("@")
            if not repo or not revision:
                print("error: --source is REPO@REVISION", file=sys.stderr)
                return 2
            metadata = {"source_repository": repo, "source_revision": revision}

    if len(args) != 2:
        print(__doc__)
        return 2

    snap, out = args

    main_path = os.path.realpath(os.path.join(snap, "model.safetensors"))
    t5_path = os.path.realpath(os.path.join(snap, "t5gemma-b-b-ul2", "model.safetensors"))

    for path in (main_path, t5_path):
        if not os.path.isfile(path):
            print(f"error: not found: {path}", file=sys.stderr)
            print(f"is {snap!r} a Stable Audio 3 snapshot directory?", file=sys.stderr)
            return 1

    ent, n_dec, n_dit, n_t5 = build_entries(Reader(main_path), Reader(t5_path))

    total = sum(int(np.prod(s)) * 4 for _, s, _ in ent)
    peak = max(int(np.prod(s)) * 4 for _, s, _ in ent)

    print(f"decoder blocks={n_dec}  DiT layers={n_dit}  T5 layers={n_t5}")
    print(f"{len(ent)} tensors -> {total / 1e9:.2f} GB   peak tensor {peak / 1e6:.0f} MB")

    if dry:
        print("dry run: no write")
        return 0

    print(f"writing {out} ...", flush=True)
    written = write_stream(out, ent, metadata)
    print(f"done: {out} ({written / 1e9:.2f} GB, {len(ent)} tensors)")

    return 0


if __name__ == "__main__":
    sys.exit(main_cli())
