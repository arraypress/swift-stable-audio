//
//  GenerationOptions.swift
//  StableAudioKit
//
//  Per-request knobs. Anything left `nil` falls back to the model's own
//  defaults via `resolved(for:)`, so callers usually just set `seconds`.
//

import Foundation

/// Options for a single generation request.
public struct GenerationOptions: Sendable, Equatable {

    /// Desired clip length (seconds). Capped to the model's `maxSeconds`.
    public var seconds: Double

    /// Diffusion steps. `nil` → the model's `defaultSteps`.
    public var steps: Int?

    /// Classifier-free-guidance scale. `nil` → the model's `defaultCFG`.
    public var cfgScale: Double?

    /// Fixed seed for reproducibility. `nil` → random.
    public var seed: Int?

    /// Optional negative prompt (things to steer away from).
    public var negativePrompt: String?

    /// Produce a seamless loop (crossfades the tail into the head). Best for
    /// steady content — rain, wind, fire, drones, pads.
    public var loop: Bool

    /// Tempo for a *musical* loop. When set together with `loop`, the clip is
    /// cropped to an exact whole number of bars at this BPM so the loop point
    /// lands on a downbeat, and only a short de-click crossfade is applied (a
    /// long blend would smear the beat). `nil` → texture loop (long crossfade).
    public var loopBPM: Double?

    public init(
        seconds: Double = 10,
        steps: Int? = nil,
        cfgScale: Double? = nil,
        seed: Int? = nil,
        negativePrompt: String? = nil,
        loop: Bool = false,
        loopBPM: Double? = nil
    ) {
        self.seconds = seconds
        self.steps = steps
        self.cfgScale = cfgScale
        self.seed = seed
        self.negativePrompt = negativePrompt
        self.loop = loop
        self.loopBPM = loopBPM
    }

    /// Resolve the effective parameters against a model's defaults and limits.
    public func resolved(for model: StableAudioModel)
        -> (steps: Int, cfgScale: Double, seconds: Double)
    {
        (
            steps: steps ?? model.defaultSteps,
            cfgScale: cfgScale ?? model.defaultCFG,
            seconds: min(max(seconds, 0), model.maxSeconds)
        )
    }
}
