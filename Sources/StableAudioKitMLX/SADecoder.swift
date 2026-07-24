//
//  SADecoder.swift
//  StableAudioKitMLX
//
//  The Stable Audio autoencoder decoder (latent -> stereo audio), transcribed
//  1:1 from the validated MLX-Python port (`mlx_decoder.py`, matched to PyTorch
//  at 2.7e-5). A `TransformerResamplingBlock`: differential attention + DyT +
//  partial RoPE + SwiGLU, chunked with midpoint-shift, then a weight-norm conv
//  and a patch-unfold to stereo.
//

import Foundation
import MLX

public struct SADecoder {

    let w: [String: MLXArray]
    public init(w: [String: MLXArray]) { self.w = w }
    private func W(_ k: String) -> MLXArray { w[k]! }

    // MARK: - Transformer block (differential attention, DyT, RoPE, SwiGLU)

    private func attention(_ x: MLXArray, _ p: String, _ freqs: MLXArray) -> MLXArray {
        let B = x.dim(0), n = x.dim(1), H = 12, D = 64
        let qkv = SAOps.linear(x, W(p + "to_qkv"))
        let s = split(qkv, parts: 5, axis: -1)                     // q,k,v,qd,kd
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
        let o = SAOps.mergeHeads(SAOps.sdpa(q, k, v) - SAOps.sdpa(qd, kd, v), B, n, H, D)
        return SAOps.linear(o, W(p + "to_out"))
    }

    private func feedForward(_ x: MLXArray, _ p: String) -> MLXArray {
        let h = SAOps.linear(x, W(p + "glu_proj_w"), bias: W(p + "glu_proj_b"))
        let g = split(h, parts: 2, axis: -1)
        return SAOps.linear(g[0] * SAOps.silu(g[1]), W(p + "ff_out_w"), bias: W(p + "ff_out_b"))
    }

    private func block(_ x0: MLXArray, _ p: String, _ freqs: MLXArray) -> MLXArray {
        var x = x0 + attention(SAOps.dynamicTanh(x0, W(p + "prenorm_alpha"), W(p + "prenorm_gamma"), W(p + "prenorm_beta")), p, freqs)
        x = x + feedForward(SAOps.dynamicTanh(x, W(p + "ffnorm_alpha"), W(p + "ffnorm_gamma"), W(p + "ffnorm_beta")), p)
        return x
    }

    // MARK: - Weight-norm 1x1/3 conv + patch unfold

    private func conv1dSame(_ x: MLXArray, _ weight: MLXArray, _ bias: MLXArray) -> MLXArray {
        let xn = x.transposed(axes: [0, 2, 1])          // (1,L,Cin)
        let wn = weight.transposed(axes: [0, 2, 1])     // (Cout,3,Cin) mlx layout
        let y = conv1d(xn, wn, padding: 1) + bias       // (1,L,Cout)
        return y.transposed(axes: [0, 2, 1])            // (1,Cout,L)
    }

    // MARK: - Resampling block (latent frames T -> audio-rate features)

    /// `x`: (1, 768, T) -> (1, 512, T*16). `T` must be even.
    private func resampling(_ x0: MLXArray, _ T: Int) -> MLXArray {
        let ecs = 34, shift = 17, subChunk = 17
        var x = x0.transposed(axes: [0, 2, 1])          // (1,T,768)
        x = x.reshaped([T, 1, 768])
        let nt = broadcast(W("new_tokens"), to: [T, 16, 768])
        x = concatenated([x, nt], axis: 1)              // (T,17,768)
        let seq = T * subChunk                           // (1, T*17, 768)
        x = x.reshaped([1, seq, 768])
        let freqs = SAOps.ropeFreqs(ecs, dim: 32)

        // first half: aligned chunks, layers 0..2
        var xa = x.reshaped([seq / ecs, ecs, 768])
        for i in 0 ..< 3 { xa = block(xa, "L\(i).", freqs) }
        x = xa.reshaped([1, seq, 768])

        // second half: shifted chunks, layers 3..5
        let head = split(x, indices: [shift], axis: 1)[0]
        let tail = split(x, indices: [seq - shift], axis: 1)[1]
        x = concatenated([head, x, tail], axis: 1)      // (1, seq+34, 768)
        let seq2 = seq + ecs
        var xb = x.reshaped([seq2 / ecs, ecs, 768])
        for i in 3 ..< 6 { xb = block(xb, "L\(i).", freqs) }
        x = xb.reshaped([1, seq2, 768])
        x = split(x, indices: [shift, seq2 - shift], axis: 1)[1]   // drop shift pad -> (1,seq,768)

        // keep the 16 new-token outputs per 17-subchunk
        x = x.reshaped([T, subChunk, 768])
        x = split(x, indices: [1], axis: 1)[1]          // (T,16,768)
        x = x.reshaped([1, T, 16, 768])
        x = x.transposed(axes: [0, 3, 1, 2]).reshaped([1, 768, T * 16])  // 'b d (n c)'
        return conv1dSame(x, W("map_weight"), W("map_bias"))            // (1,512,T*16)
    }

    // MARK: - Full decode

    /// `latent`: (1, 256, T) -> audio (1, 2, T*4096).
    public func decode(_ latent: MLXArray) -> MLXArray {
        let T = latent.dim(2)
        let z = latent * W("running_std")
        var x = SAOps.linear(z.transposed(axes: [0, 2, 1]), W("lin_w"), bias: W("lin_b")) // (1,T,768)
        x = x.transposed(axes: [0, 2, 1])                                                 // (1,768,T)
        x = resampling(x, T)                                                              // (1,512,T*16)
        let L = x.dim(2)
        // patch unfold "b (c h) l -> b c (l h)", h = 256
        x = x.reshaped([1, 2, 256, L])
        return x.transposed(axes: [0, 1, 3, 2]).reshaped([1, 2, L * 256])
    }
}
