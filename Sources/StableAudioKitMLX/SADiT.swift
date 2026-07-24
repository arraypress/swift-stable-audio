//
//  SADiT.swift
//  StableAudioKitMLX
//
//  The Stable Audio 3 Diffusion Transformer (20 layers), transcribed 1:1 from
//  the validated MLX-Python port (`mlx_dit.py`, matched to PyTorch at 1.14e-4).
//  AdaLN self-attention (RMS-qk + RoPE) + cross-attention to text + per-frame
//  local conditioning + timestep Fourier features + memory tokens.
//

import Foundation
import MLX

struct SADiT {

    let w: [String: MLXArray]
    private func W(_ k: String) -> MLXArray { w[k]! }

    private func lin(_ x: MLXArray, _ p: String, bias: Bool) -> MLXArray {
        SAOps.linear(x, W(p + ".w"), bias: bias ? W(p + ".b") : nil)
    }
    /// Linear - SiLU - Linear.
    private func mlp(_ x: MLXArray, _ p0: String, _ p2: String, bias: Bool) -> MLXArray {
        lin(SAOps.silu(lin(x, p0, bias: bias)), p2, bias: bias)
    }

    // MARK: - Attention

    private func selfAttn(_ x: MLXArray, _ p: String, _ freqs: MLXArray) -> MLXArray {
        let B = x.dim(0), n = x.dim(1), H = 16, D = 64
        let s = split(SAOps.linear(x, W(p + "sa_qkv.w")), parts: 3, axis: -1)
        var q = SAOps.toHeads(s[0], B, n, H, D)
        var k = SAOps.toHeads(s[1], B, n, H, D)
        let v = SAOps.toHeads(s[2], B, n, H, D)
        q = SAOps.rmsNorm(q, W(p + "sa_qn"), eps: 1e-6)
        k = SAOps.rmsNorm(k, W(p + "sa_kn"), eps: 1e-6)
        q = SAOps.applyRopeHalf(q, freqs); k = SAOps.applyRopeHalf(k, freqs)
        return SAOps.linear(SAOps.mergeHeads(SAOps.sdpa(q, k, v), B, n, H, D), W(p + "sa_out.w"))
    }

    private func crossAttn(_ x: MLXArray, _ ctx: MLXArray, _ p: String) -> MLXArray {
        let B = x.dim(0), n = x.dim(1), M = ctx.dim(1), H = 16, D = 64
        var q = SAOps.toHeads(SAOps.linear(x, W(p + "ca_q.w")), B, n, H, D)
        let kv = split(SAOps.linear(ctx, W(p + "ca_kv.w")), parts: 2, axis: -1)
        var k = SAOps.toHeads(kv[0], B, M, H, D)
        let v = SAOps.toHeads(kv[1], B, M, H, D)
        q = SAOps.rmsNorm(q, W(p + "ca_qn"), eps: 1e-6)
        k = SAOps.rmsNorm(k, W(p + "ca_kn"), eps: 1e-6)
        return SAOps.linear(SAOps.mergeHeads(SAOps.sdpa(q, k, v), B, n, H, D), W(p + "ca_out.w"))
    }

    private func feedForward(_ x: MLXArray, _ p: String) -> MLXArray {
        let s = split(lin(x, p + "glu", bias: true), parts: 2, axis: -1)
        return lin(s[0] * SAOps.silu(s[1]), p + "ff_out", bias: true)
    }

    // MARK: - Block (AdaLN + cross-attn + local cond)

    private func block(_ x0: MLXArray, _ p: String, _ ctx: MLXArray, _ freqs: MLXArray, _ gc: MLXArray, _ local: MLXArray, _ nMem: Int) -> MLXArray {
        let ss = split((W(p + "ssg") + gc).expandedDimensions(axis: 1), parts: 6, axis: -1)
        var x = x0
        var h = SAOps.rmsNorm(x, W(p + "prenorm_g"), eps: 1e-5) * (1 + ss[0]) + ss[1]
        h = selfAttn(h, p, freqs) * sigmoid(1 - ss[2])
        x = h + x
        x = x + crossAttn(SAOps.rmsNorm(x, W(p + "xnorm_g"), eps: 1e-5), ctx, p)
        var le = mlp(local, p + "loc0", p + "loc2", bias: true)          // (B,T,1024)
        le = concatenated([MLXArray.zeros([le.dim(0), nMem, 1024]), le], axis: 1)
        x = x + le
        let res = x
        h = SAOps.rmsNorm(x, W(p + "ffnorm_g"), eps: 1e-5) * (1 + ss[3]) + ss[4]
        h = feedForward(h, p) * sigmoid(1 - ss[5])
        return h + res
    }

    // MARK: - Helpers

    private func conv1x1(_ x: MLXArray, _ weight: MLXArray) -> MLXArray {   // (Cout,Cin,1)
        let w2 = weight.reshaped([weight.dim(0), weight.dim(1)])
        let y = matmul(x.transposed(axes: [0, 2, 1]), w2.T)
        return y.transposed(axes: [0, 2, 1])
    }

    // MARK: - Forward

    /// `x`: (1,256,T), `t`: (1,), conds -> velocity (1,256,T).
    func forward(_ x: MLXArray, _ t: MLXArray, _ ctxRaw: MLXArray, _ globalRaw: MLXArray, _ localRaw: MLXArray) -> MLXArray {
        let B = x.dim(0), nMem = 64
        let ctx = mlp(ctxRaw, "cond_embed0", "cond_embed2", bias: false)
        var glob = mlp(globalRaw, "glob_embed0", "glob_embed2", bias: false)
        glob = glob + mlp(SAOps.expoFeatures(t), "t_embed0", "t_embed2", bias: true)
        let gc = mlp(glob, "gce0", "gce2", bias: true)
        let local = localRaw.transposed(axes: [0, 2, 1])                 // (B,T,257)

        var h = conv1x1(x, W("pre_conv")) + x                           // (B,256,T)
        h = SAOps.linear(h.transposed(axes: [0, 2, 1]), W("project_in.w"))  // (B,T,1024)
        let mem = broadcast(W("memory_tokens").expandedDimensions(axis: 0), to: [B, nMem, 1024])
        h = concatenated([mem, h], axis: 1)                             // (B,T+64,1024)
        let freqs = SAOps.ropeFreqs(h.dim(1), dim: 32)
        for i in 0 ..< 20 { h = block(h, "L\(i).", ctx, freqs, gc, local, nMem) }
        h = split(h, indices: [nMem], axis: 1)[1]                       // drop memory
        h = SAOps.linear(h, W("project_out.w")).transposed(axes: [0, 2, 1]) // (B,256,T)
        return conv1x1(h, W("post_conv")) + h
    }
}
