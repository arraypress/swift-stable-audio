//
//  SAPipeline.swift
//  StableAudioKitMLX
//
//  Composes the ported nets into text -> audio: T5Gemma conds -> ping-pong
//  flow sampler over the DiT -> autoencoder decoder. Transcribed from the
//  validated MLX-Python pipeline (end-to-end matched to PyTorch at 8.7e-4).
//

import Foundation
import MLX
import MLXRandom

struct SAPipeline {

    let t5: SAT5Gemma
    let dit: SADiT
    let decoder: SADecoder
    let paddingEmbedding: MLXArray          // (768,)

    /// The distilled model's fixed 8-step ping-pong schedule (sigmas, len steps+1).
    static let sigmas8: [Float] = [1.0, 0.9943756, 0.9844802, 0.95791227,
                                   0.8909032, 0.7455466, 0.51249737, 0.273885, 0.0]

    /// Generate stereo audio (1, 2, frames*4096).
    /// - Parameters:
    ///   - ids/mask: tokenized prompt (1, L).
    ///   - secondsEmbedding: NumberConditioner output (1, 1, 768).
    ///   - frames: latent length T (even).
    func generate(ids: MLXArray, mask: MLXArray, secondsEmbedding: MLXArray,
                  frames T: Int, seed: UInt64 = 0) -> MLXArray {
        MLXRandom.seed(seed)

        // Text conditioning
        var prompt = t5.encode(ids, mask)                                  // (1,L,768)
        let m = (mask .!= 0).reshaped([mask.dim(0), mask.dim(1), 1])
        prompt = MLX.where(m, prompt, paddingEmbedding)                    // apply padding embedding
        let ctx = concatenated([prompt, secondsEmbedding], axis: 1)        // (1,L+1,768)
        let glob = secondsEmbedding.reshaped([secondsEmbedding.dim(0), 768])
        let local = MLXArray.zeros([1, 257, T])                           // pure gen (no inpaint)

        // Ping-pong flow sampling
        let s = Self.sigmas8
        var x = MLXRandom.normal([1, 256, T])
        for i in 0 ..< (s.count - 1) {
            let denoised = x - s[i] * dit.forward(x, MLXArray([s[i]]), ctx, glob, local)
            x = (1 - s[i + 1]) * denoised + s[i + 1] * MLXRandom.normal([1, 256, T])
            eval(x)
        }

        return decoder.decode(x)                                          // (1,2,T*4096)
    }
}
