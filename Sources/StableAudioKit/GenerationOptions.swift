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

    public init(
        seconds: Double = 10,
        steps: Int? = nil,
        cfgScale: Double? = nil,
        seed: Int? = nil,
        negativePrompt: String? = nil,
        loop: Bool = false
    ) {
        self.seconds = seconds
        self.steps = steps
        self.cfgScale = cfgScale
        self.seed = seed
        self.negativePrompt = negativePrompt
        self.loop = loop
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
