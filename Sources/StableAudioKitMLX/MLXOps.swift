//
//  MLXOps.swift
//  StableAudioKitMLX
//
//  Shared neural primitives for the Stable Audio port, transcribed 1:1 from the
//  validated MLX-Python reference (differential attention, DynamicTanh, RMSNorm,
//  partial/full RoPE, SwiGLU, SDPA). Numerically matched to PyTorch offline.
//

import Foundation
import MLX

enum SAOps {

    /// `x @ w.T (+ bias)` — `w` is stored (out, in).
    static func linear(_ x: MLXArray, _ w: MLXArray, bias: MLXArray? = nil) -> MLXArray {
        var y = matmul(x, w.T)
        if let b = bias { y = y + b }
        return y
    }

    /// RMSNorm (fp32 internally). `onePlus` uses Gemma's `(1 + weight)` scale.
    static func rmsNorm(_ x: MLXArray, _ g: MLXArray, eps: Float = 1e-5, onePlus: Bool = false) -> MLXArray {
        let x32 = x.asType(.float32)
        let n = x32 * rsqrt(mean(x32 * x32, axis: -1, keepDims: true) + eps)
        let scale = onePlus ? (1.0 + g) : g
        return (n * scale).asType(x.dtype)
    }

    /// DynamicTanh: `tanh(alpha * x) * gamma + beta`.
    static func dynamicTanh(_ x: MLXArray, _ a: MLXArray, _ g: MLXArray, _ b: MLXArray) -> MLXArray {
        tanh(a * x) * g + b
    }

    /// SiLU / swish.
    static func silu(_ x: MLXArray) -> MLXArray { x * sigmoid(x) }

    /// GELU (tanh approximation), as in `gelu_pytorch_tanh`.
    static func geluTanh(_ x: MLXArray) -> MLXArray {
        0.5 * x * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    }

    /// Rotate the two halves of the last dim: `[-x2, x1]`.
    static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let p = split(x, parts: 2, axis: -1)
        return concatenated([-p[1], p[0]], axis: -1)
    }

    /// RoPE frequency table of shape `(n, dim)` (dim = 2 × half-freqs).
    static func ropeFreqs(_ n: Int, dim: Int, base: Float = 10000) -> MLXArray {
        let idx = MLXArray(stride(from: 0, to: dim, by: 2).map { Float($0) })
        let inv = 1.0 / pow(MLXArray(base), idx / Float(dim))          // (dim/2,)
        let t = MLXArray((0 ..< n).map { Float($0) })                  // (n,)
        let fr = t.reshaped([n, 1]) * inv.reshaped([1, -1])           // (n, dim/2)
        return concatenated([fr, fr], axis: -1)                       // (n, dim)
    }

    /// Apply RoPE where `freqs` covers the first half of the head dim
    /// (decoder/DiT: rot = head_dim/2, so split the head in two).
    static func applyRopeHalf(_ t: MLXArray, _ freqs: MLXArray) -> MLXArray {
        let p = split(t, parts: 2, axis: -1)                          // rot = second half kept
        let rotated = p[0] * cos(freqs) + rotateHalf(p[0]) * sin(freqs)
        return concatenated([rotated, p[1]], axis: -1)
    }

    /// Apply RoPE across the full head dim (T5Gemma).
    static func applyRopeFull(_ t: MLXArray, _ freqs: MLXArray) -> MLXArray {
        t * cos(freqs) + rotateHalf(t) * sin(freqs)
    }

    /// Scaled dot-product attention over `(b, h, n, d)`.
    static func sdpa(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray) -> MLXArray {
        let scale = 1.0 / Float(q.dim(-1)).squareRoot()
        let att = softmax(matmul(q * scale, k.swappedAxes(-1, -2)), axis: -1)
        return matmul(att, v)
    }

    /// Reshape `(B, n, H*D)` → heads `(B, H, n, D)`.
    static func toHeads(_ t: MLXArray, _ B: Int, _ n: Int, _ H: Int, _ D: Int) -> MLXArray {
        t.reshaped([B, n, H, D]).transposed(axes: [0, 2, 1, 3])
    }

    /// Merge heads `(B, H, n, D)` → `(B, n, H*D)`.
    static func mergeHeads(_ t: MLXArray, _ B: Int, _ n: Int, _ H: Int, _ D: Int) -> MLXArray {
        t.transposed(axes: [0, 2, 1, 3]).reshaped([B, n, H * D])
    }
}
