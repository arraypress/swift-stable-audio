"""Quantize a converted SA3 safetensors file.

RECOVERED 2026-08-02 from the session transcript of the work that produced it.
The original lived only in a `/private/tmp/.../scratchpad/` directory and was
wiped, which is the failure mode CLAUDE.md warns about. Its logic below is
unmodified.

READ THIS BEFORE USING IT — it is not the script that made the shipped
weights:

  * This produces **4-bit**. The shipped `sa3-*.safetensors` are **8-bit**
    (`quant.bits = 8`, `quant.group = 64`, confirmed by reading their
    headers). CLAUDE.md records 4-bit as audibly degraded for this diffusion
    model — 0.15 correlation against fp32, because the 8 sampling steps
    compound weight error. **Do not ship 4-bit output.**
  * It does not write the `quant.bits` / `quant.group` tensors that
    `MLXEngine` reads to decide whether a file needs dequantizing. Without
    them the engine treats the file as unquantized and loads garbage.
  * It has never been run in this form here. `mlx` and `safetensors` are not
    currently installed — unlike the other tools in this directory, this one
    needs them.

To reconstitute the 8-bit production quantizer, three changes are needed:
set `BITS = 8`, add `quant.bits` and `quant.group` as int32 tensors in the
output dict, and verify a generated clip against an fp32 reference. That
fp32 reference no longer exists either, but `convert_stream.py` regenerates
it in under 20 seconds.

Until that is done and checked, treat the existing weight files as
irreplaceable.

    python3 quantize.py <converted.safetensors>

Produces:
    <out>_4bit.safetensors  true 4-bit packed (weight/scales/biases)
    <out>_dq.safetensors    dequantized fp32, for measuring the quality the
                            quantized weights will produce using unchanged nets
"""

import sys

import numpy as np
import mlx.core as mx
from safetensors.numpy import load_file, save_file

src = sys.argv[1]
base = src.replace(".safetensors", "")
GS, BITS = 64, 4
w = load_file(src)
q, dq = {}, {}
n_q = 0
orig = 0
comp = 0

for k, v in w.items():
    a = mx.array(v)
    # quantize 2D matrices whose input dim is group-aligned (the big linears + embed)
    if a.ndim == 2 and a.shape[-1] % GS == 0 and a.shape[0] >= 64:
        wq, sc, bi = mx.quantize(a, group_size=GS, bits=BITS)
        mx.eval(wq, sc, bi)
        q[k] = np.array(wq)
        q[k + ".scales"] = np.array(sc)
        q[k + ".biases"] = np.array(bi)
        dqv = mx.dequantize(wq, sc, bi, group_size=GS, bits=BITS)
        mx.eval(dqv)
        dq[k] = np.array(dqv).astype(np.float32)
        n_q += 1
        orig += v.nbytes
        comp += q[k].nbytes + sc.nbytes + bi.nbytes
    else:
        q[k] = v.astype(np.float16)
        dq[k] = v.astype(np.float32)
        orig += v.nbytes
        comp += q[k].nbytes

save_file(q, base + "_4bit.safetensors")
save_file(dq, base + "_dq.safetensors")
print(f"quantized {n_q} matrices to 4-bit")
print(f"size: {orig/1e9:.2f} GB (fp32) -> {comp/1e9:.2f} GB (4-bit)  [{orig/comp:.1f}x smaller]")
