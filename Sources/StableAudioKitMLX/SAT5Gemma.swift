//
//  SAT5Gemma.swift
//  StableAudioKitMLX
//
//  The T5Gemma text encoder (12-layer bidirectional Gemma2-style), transcribed
//  1:1 from the validated MLX-Python port (`mlx_t5.py`, matched to a float32
//  PyTorch reference at 1.9e-3). Sandwich RMSNorms (1+w), full-dim RoPE, GeGLU,
//  attention logit softcapping (50), padding mask.
//

import Foundation
import MLX

struct SAT5Gemma {

    let w: [String: MLXArray]
    private func W(_ k: String) -> MLXArray { w[k]! }
    private static let H = 12, D = 64

    private func rms(_ x: MLXArray, _ g: MLXArray) -> MLXArray {
        SAOps.rmsNorm(x, g, eps: 1e-6, onePlus: true)
    }

    private func attn(_ x: MLXArray, _ p: String, _ freqs: MLXArray, _ keyMask: MLXArray) -> MLXArray {
        let B = x.dim(0), n = x.dim(1)
        let q = SAOps.applyRopeFull(SAOps.toHeads(SAOps.linear(x, W(p + "q")), B, n, Self.H, Self.D), freqs)
        let k = SAOps.applyRopeFull(SAOps.toHeads(SAOps.linear(x, W(p + "k")), B, n, Self.H, Self.D), freqs)
        let v = SAOps.toHeads(SAOps.linear(x, W(p + "v")), B, n, Self.H, Self.D)
        var scores = matmul(q * (1.0 / 8.0), k.swappedAxes(-1, -2))       // (B,H,n,n)
        scores = tanh(scores / 50.0) * 50.0                              // attn logit softcap
        let masked = (keyMask .== 0).reshaped([B, 1, 1, n])
        scores = MLX.where(masked, MLXArray(Float(-1e9)), scores)
        let o = SAOps.mergeHeads(matmul(softmax(scores, axis: -1), v), B, n, Self.H, Self.D)
        return SAOps.linear(o, W(p + "o"))
    }

    private func mlp(_ x: MLXArray, _ p: String) -> MLXArray {
        SAOps.linear(SAOps.geluTanh(SAOps.linear(x, W(p + "gate"))) * SAOps.linear(x, W(p + "up")), W(p + "down"))
    }

    private func layer(_ x0: MLXArray, _ p: String, _ freqs: MLXArray, _ keyMask: MLXArray) -> MLXArray {
        var x = x0 + rms(attn(rms(x0, W(p + "pre_attn")), p, freqs, keyMask), W(p + "post_attn"))
        x = x + rms(mlp(rms(x, W(p + "pre_ff")), p), W(p + "post_ff"))
        return x
    }

    /// `ids`: (1, n) token ids, `mask`: (1, n) 1/0. -> (1, n, 768) embeddings.
    func encode(_ ids: MLXArray, _ mask: MLXArray) -> MLXArray {
        let n = ids.dim(1)
        var x = W("embed_tokens")[ids] * Float(768.0).squareRoot()
        let freqs = SAOps.ropeFreqs(n, dim: 64)
        for i in 0 ..< 12 { x = layer(x, "L\(i).", freqs, mask) }
        return rms(x, W("final_norm"))
    }
}
