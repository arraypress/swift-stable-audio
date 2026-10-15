//
//  GaussianNoise.swift
//  StableAudioKit
//
//  Seeded standard-normal noise for the samplers that run on the host:
//  xoshiro256** seeded through SplitMix64, turned normal by Box–Muller. The
//  same seed gives the same values on every machine and every run, which is
//  what makes a generation repeatable from its seed.
//

import Foundation

/// A source of the noise a sampler draws: the start latent and each step's.
public protocol LatentNoise {
    /// The next `count` standard-normal values.
    mutating func next(count: Int) -> [Float]
}

/// Deterministic standard-normal noise from a seed.
public struct GaussianNoise: LatentNoise, Sendable {

    private var state: (UInt64, UInt64, UInt64, UInt64)
    private var spare: Double?

    public init(seed: UInt64) {
        var mix = seed
        func splitMix() -> UInt64 {
            mix &+= 0x9E37_79B9_7F4A_7C15
            var z = mix
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
        state = (splitMix(), splitMix(), splitMix(), splitMix())
    }

    /// The next 64 random bits (xoshiro256**).
    private mutating func nextBits() -> UInt64 {
        let result = ((state.1 &* 5) << 7 | (state.1 &* 5) >> 57) &* 9
        let t = state.1 << 17
        state.2 ^= state.0
        state.3 ^= state.1
        state.1 ^= state.2
        state.0 ^= state.3
        state.2 ^= t
        state.3 = state.3 << 45 | state.3 >> 19
        return result
    }

    /// A uniform double in (0, 1]: never zero, so its logarithm is finite.
    private mutating func unit() -> Double {
        (Double(nextBits() >> 11) + 1) / 9_007_199_254_740_992.0
    }

    /// One standard-normal value.
    public mutating func nextValue() -> Double {
        if let spare {
            self.spare = nil
            return spare
        }
        let radius = (-2 * log(unit())).squareRoot()
        let angle = 2 * Double.pi * unit()
        spare = radius * sin(angle)
        return radius * cos(angle)
    }

    public mutating func next(count: Int) -> [Float] {
        (0..<count).map { _ in Float(nextValue()) }
    }
}
