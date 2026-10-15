//
//  PingPongSampler.swift
//  StableAudioKit
//
//  The host-side arithmetic of Stable Audio 3 generation, as pure functions:
//  the timestep schedule, one ping-pong step, classifier-free guidance and
//  the latent window a clip is generated in. The Core AI engine runs the
//  networks; everything between their calls is here, where it can be
//  tested without a model. Transcribed from the export's `example.py`, which
//  works in float64 between network calls — so does this.
//

import Foundation

/// The sampler Stable Audio 3's distilled models were trained for.
///
/// Three settings are not optional for this family, and getting them wrong
/// gives audibly broken output rather than subtly worse output: ping-pong
/// sampling (Euler overshoots full scale by +7.8 dBFS), a window of at least
/// ``minimumFrames`` latent frames (shorter puts 16–27 % of the energy above
/// 10 kHz), and 8 steps at guidance 1.0.
public enum PingPongSampler {

    /// Audio samples per second.
    public static let sampleRate = 44_100

    /// Audio samples per latent frame.
    public static let samplesPerFrame = 4_096

    /// Channels of the latent the transformer works on.
    public static let latentChannels = 256

    /// The models' `distribution_shift_options.min_length`: 256 frames, 23.8 s.
    public static let minimumFrames = 256

    /// The log-SNR at the start (t = 1) and end (t = 0) of the schedule. The
    /// models' shift has rate 0, so these two fully determine it at any length.
    public static let logSNRStart = -6.2
    public static let logSNREnd = 2.0

    /// The timesteps, `steps + 1` values from `sigmaMax` down to 0, evenly
    /// spaced in log-SNR with the ends pinned.
    ///
    /// At 8 steps this is the distilled schedule the MLX engine hard-codes.
    public static func schedule(steps: Int, sigmaMax: Double = 1.0) -> [Double] {
        guard steps > 0 else { return [sigmaMax] }
        // numpy's linspace: start + i·step, with the last value exactly 0.
        let delta = (0 - sigmaMax) / Double(steps)
        return (0...steps).map { index in
            if index == 0 { return sigmaMax }
            let t = index == steps ? 0 : Double(index) * delta + sigmaMax
            if t <= 0 { return 0 }
            if t >= 1 { return 1 }
            let logSNR = logSNREnd - t * (logSNREnd - logSNRStart)
            return 1.0 / (1.0 + exp(logSNR))                // sigmoid(-logSNR)
        }
    }

    /// One ping-pong step: the denoised estimate, re-noised to the next
    /// timestep with fresh noise.
    ///
    /// - Parameters:
    ///   - x: The latent, updated in place.
    ///   - velocity: The transformer's output at `now`.
    ///   - noise: Fresh standard-normal noise, as many values as `x`.
    public static func step(_ x: inout [Double], velocity: [Float], now: Double, next: Double, noise: [Float]) {
        precondition(velocity.count == x.count && noise.count == x.count, "latent, velocity and noise differ in size")
        for i in x.indices {
            let denoised = x[i] - now * Double(velocity[i])
            x[i] = (1 - next) * denoised + next * Double(noise[i])
        }
    }

    /// Vanilla classifier-free guidance on the velocities:
    /// `v = v_cond + (scale − 1)(v_cond − v_uncond)`, computed through the
    /// denoised estimates the way the export's `guide` does (identical at
    /// any `t` above zero).
    public static func guided(x: [Double], t: Double, conditional: [Float], unconditional: [Float],
                              scale: Double) -> [Float] {
        precondition(conditional.count == x.count && unconditional.count == x.count, "sizes differ")
        guard t > 0 else { return conditional }
        return x.indices.map { i in
            let cond = x[i] - t * Double(conditional[i])
            let uncond = x[i] - t * Double(unconditional[i])
            let guidedDenoised = cond + (scale - 1) * (cond - uncond)
            return Float((x[i] - guidedDenoised) / t)
        }
    }

    /// The latent frames a clip of `seconds` is generated in: `seconds` plus
    /// `margin`, at least ``minimumFrames``. The margin keeps the tone the
    /// model puts on its final frame out of what is kept; the floor keeps the
    /// models inside what they were trained on.
    public static func windowFrames(seconds: Double, margin: Double = 3) -> Int {
        let frames = ((seconds + margin) * Double(sampleRate) / Double(samplesPerFrame)).rounded(.toNearestOrEven)
        return max(Int(frames), minimumFrames)
    }

    /// The seconds a window of `frames` holds — what the conditioner is told.
    /// The models were trained on full windows; telling one that a hit lasts
    /// 2 s puts its loss three orders of magnitude off (measured in training).
    public static func seconds(ofFrames frames: Int) -> Double {
        Double(frames * samplesPerFrame) / Double(sampleRate)
    }

    /// The shortest fixed-length codec in `lengths` that holds `frames`.
    public static func rung(for frames: Int, among lengths: [Int]) -> Int? {
        lengths.sorted().first { $0 >= frames }
    }
}
