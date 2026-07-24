//
//  SADecoder.swift
//  StableAudioKitMLX
//
//  The Stable Audio autoencoder decoder (latent -> stereo audio), transcribed
//  from the validated MLX-Python port and parameterized to cover both the SA3
//  `small` and `medium` autoencoders (they share the SAMEDecoder /
//  TransformerResamplingBlock structure — differential attention + DyT +
//  partial RoPE + SwiGLU — differing only in dims, chunking, and a Sin FF
//  activation in medium's later blocks).
//
//    small : dim 768,  12 heads, 6 layers,  chunk 32,  midpoint-shift, k3 map
//    medium: dim 1536, 24 heads, 12 layers, chunk 128, no shift,       k1 map,
//            Sin activation in the last `sinusoidalBlocks` FF blocks
//

import Foundation
import MLX

public struct SADecoder {

    let w: [String: MLXArray]
    private let dim: Int             // transformer dim (768 / 1536)
    private let numHeads: Int        // dim / 64
    private let depth: Int           // resampling-block transformer layers
    private let mapKernel: Int       // mapping conv kernel (3 / 1)
    private let chunkSize: Int
    private let midpointShift: Bool
    private let sinusoidalBlocks: Int
    private let stride = 16
    private var subChunk: Int { stride + 1 }        // 17
    private var newTokens: Int { stride }           // 16
    private var effChunk: Int { chunkSize + chunkSize / stride }

    /// Infers dims from the weights; chunking/Sin config keyed off `dim`.
    public init(w: [String: MLXArray]) {
        self.w = w
        let d = w["lin_w"]!.dim(0)
        self.dim = d
        self.numHeads = d / 64
        self.mapKernel = w["map_weight"]!.dim(2)
        var n = 0; while w["L\(n).to_qkv"] != nil { n += 1 }
        self.depth = n
        if d == 768 { self.chunkSize = 32; self.midpointShift = true; self.sinusoidalBlocks = 0; self.window = nil }
        else { self.chunkSize = 128; self.midpointShift = false; self.sinusoidalBlocks = 8; self.window = 17 }
    }
    private let window: Int?           // sliding-window half-width (medium) or nil
    private func W(_ k: String) -> MLXArray { w[k]! }

    // MARK: - Transformer block (differential attention, DyT, RoPE, SwiGLU)

    /// SDPA with an optional additive mask (sliding window).
    private func sdpaM(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, _ mask: MLXArray?) -> MLXArray {
        var s = matmul(q * (1.0 / 8.0), k.swappedAxes(-1, -2))
        if let m = mask { s = s + m }
        return matmul(softmax(s, axis: -1), v)
    }

    /// Band mask `(1,1,n,n)`: 0 where |i-j| <= window, else -inf.
    private func bandMask(_ n: Int) -> MLXArray {
        let idx = MLXArray(0 ..< n)
        let diff = idx.reshaped([n, 1]) - idx.reshaped([1, n])
        let outside = abs(diff) .> window!
        return MLX.where(outside, MLXArray(Float(-1e9)), MLXArray(Float(0))).reshaped([1, 1, n, n])
    }

    private func attention(_ x: MLXArray, _ p: String, _ freqs: MLXArray, _ mask: MLXArray?) -> MLXArray {
        let B = x.dim(0), n = x.dim(1), H = numHeads, D = 64
        let s = split(SAOps.linear(x, W(p + "to_qkv")), parts: 5, axis: -1)
        var q  = SAOps.toHeads(s[0], B, n, H, D)
        var k  = SAOps.toHeads(s[1], B, n, H, D)
        let v  = SAOps.toHeads(s[2], B, n, H, D)
        var qd = SAOps.toHeads(s[3], B, n, H, D)
        var kd = SAOps.toHeads(s[4], B, n, H, D)
        q  = SAOps.dynamicTanh(q,  W(p + "q_alpha"), W(p + "q_gamma"), W(p + "q_beta"))
        qd = SAOps.dynamicTanh(qd, W(p + "q_alpha"), W(p + "q_gamma"), W(p + "q_beta"))
        k  = SAOps.dynamicTanh(k,  W(p + "k_alpha"), W(p + "k_gamma"), W(p + "k_beta"))
        kd = SAOps.dynamicTanh(kd, W(p + "k_alpha"), W(p + "k_gamma"), W(p + "k_beta"))
        q = SAOps.applyRopeHalf(q, freqs); qd = SAOps.applyRopeHalf(qd, freqs)
        k = SAOps.applyRopeHalf(k, freqs); kd = SAOps.applyRopeHalf(kd, freqs)
        let o = SAOps.mergeHeads(sdpaM(q, k, v, mask) - sdpaM(qd, kd, v, mask), B, n, H, D)
        return SAOps.linear(o, W(p + "to_out"))
    }

    private func feedForward(_ x: MLXArray, _ p: String, sinusoidal: Bool) -> MLXArray {
        let h = SAOps.linear(x, W(p + "glu_proj_w"), bias: W(p + "glu_proj_b"))
        let g = split(h, parts: 2, axis: -1)
        let act = sinusoidal ? sin(Float.pi * g[1]) : SAOps.silu(g[1])   // Sin(x)=sin(pi x)
        return SAOps.linear(g[0] * act, W(p + "ff_out_w"), bias: W(p + "ff_out_b"))
    }

    private func block(_ x0: MLXArray, _ i: Int, _ freqs: MLXArray, _ mask: MLXArray?) -> MLXArray {
        let p = "L\(i)."
        let sinusoidal = (depth - i) < sinusoidalBlocks
        var x = x0 + attention(SAOps.dynamicTanh(x0, W(p + "prenorm_alpha"), W(p + "prenorm_gamma"), W(p + "prenorm_beta")), p, freqs, mask)
        x = x + feedForward(SAOps.dynamicTanh(x, W(p + "ffnorm_alpha"), W(p + "ffnorm_gamma"), W(p + "ffnorm_beta")), p, sinusoidal: sinusoidal)
        return x
    }

    // MARK: - Weight-norm conv (k=3 or k=1) + patch unfold

    private func conv1dSame(_ x: MLXArray, _ weight: MLXArray, _ bias: MLXArray) -> MLXArray {
        let xn = x.transposed(axes: [0, 2, 1])          // (1,L,Cin)
        let wn = weight.transposed(axes: [0, 2, 1])     // (Cout,K,Cin)
        let y = conv1d(xn, wn, padding: (mapKernel - 1) / 2) + bias
        return y.transposed(axes: [0, 2, 1])            // (1,Cout,L)
    }

    /// Zero-pad axis-1 up to a multiple of `m`.
    private func padTo(_ x: MLXArray, _ m: Int) -> MLXArray {
        let n = x.dim(1), pad = (m - n % m) % m
        if pad == 0 { return x }
        return concatenated([x, MLXArray.zeros([x.dim(0), pad, x.dim(2)])], axis: 1)
    }

    // MARK: - Resampling block: (1, dim, T) -> (1, 512, T*16)

    private func resampling(_ x0: MLXArray, _ T0: Int) -> MLXArray {
        var x = x0.transposed(axes: [0, 2, 1])          // (1,T,dim)
        x = padTo(x, chunkSize / stride)                // decoder pad (no-op when aligned)
        let T = x.dim(1)
        x = x.reshaped([T, 1, dim])
        x = concatenated([x, broadcast(W("new_tokens"), to: [T, newTokens, dim])], axis: 1) // (T,17,dim)
        let seq = T * subChunk
        x = x.reshaped([1, seq, dim])

        if midpointShift {
            // small: chunk into effChunk, midpoint-shift, full attention per chunk
            let freqs = SAOps.ropeFreqs(effChunk, dim: 32)
            let shift = effChunk / 2
            var xa = x.reshaped([seq / effChunk, effChunk, dim])
            for i in 0 ..< (depth / 2) { xa = block(xa, i, freqs, nil) }
            x = xa.reshaped([1, seq, dim])
            let head = split(x, indices: [shift], axis: 1)[0]
            let tail = split(x, indices: [seq - shift], axis: 1)[1]
            x = concatenated([head, x, tail], axis: 1)
            let seq2 = seq + effChunk
            var xb = x.reshaped([seq2 / effChunk, effChunk, dim])
            for i in (depth / 2) ..< depth { xb = block(xb, i, freqs, nil) }
            x = xb.reshaped([1, seq2, dim])
            x = split(x, indices: [shift, seq2 - shift], axis: 1)[1]
        } else {
            // medium: no chunking; full sequence with sliding-window band mask
            let freqs = SAOps.ropeFreqs(seq, dim: 32)
            let mask = bandMask(seq)
            for i in 0 ..< depth { x = block(x, i, freqs, mask) }
        }

        x = x.reshaped([T, subChunk, dim])
        x = split(x, indices: [subChunk - newTokens], axis: 1)[1]     // keep last 16 per 17
        x = x.reshaped([1, T, newTokens, dim])
        x = x.transposed(axes: [0, 3, 1, 2]).reshaped([1, dim, T * newTokens])
        return conv1dSame(x, W("map_weight"), W("map_bias"))          // (1,512,T*16)
    }

    // MARK: - Full decode

    /// `latent`: (1, 256, T) -> audio (1, 2, T*4096).
    public func decode(_ latent: MLXArray) -> MLXArray {
        let T = latent.dim(2)
        let z = latent * W("running_std")
        var x = SAOps.linear(z.transposed(axes: [0, 2, 1]), W("lin_w"), bias: W("lin_b"))
        x = x.transposed(axes: [0, 2, 1])               // (1,dim,T)
        x = resampling(x, T)                            // (1,512,T*16)
        let L = x.dim(2)
        x = x.reshaped([1, 2, 256, L])                  // patch unfold h=256
        return x.transposed(axes: [0, 1, 3, 2]).reshaped([1, 2, L * 256])
    }
}
