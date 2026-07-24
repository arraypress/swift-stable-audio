//
//  StableAudio.swift
//  StableAudioKit
//
//  The friendly entry point: wrap any engine and generate with one call.
//

import Foundation

/// A text-to-audio generator backed by a swappable ``AudioGenEngine``.
///
/// ```swift
/// let gen = StableAudio(RemoteEngine(.elevenLabsSoundEffects(apiKey: key)))
/// let audio = try await gen.generate("thunder rolling over a city", seconds: 8)
/// try audio.wavData.write(to: url)
/// ```
public struct StableAudio: Sendable {

    /// The backend doing the work.
    public let engine: any AudioGenEngine

    public init(_ engine: any AudioGenEngine) {
        self.engine = engine
    }

    /// Whether the engine is ready to generate.
    public var isReady: Bool { engine.isReady }

    /// Download / load the model if needed.
    public func prepare() async throws {
        try await engine.prepare()
    }

    /// Generate audio from a text prompt.
    @discardableResult
    public func generate(
        _ prompt: String,
        seconds: Double = 10,
        steps: Int? = nil,
        cfgScale: Double? = nil,
        seed: Int? = nil,
        negativePrompt: String? = nil
    ) async throws -> GeneratedAudio {
        try await engine.generate(
            prompt: prompt,
            options: GenerationOptions(
                seconds: seconds, steps: steps, cfgScale: cfgScale,
                seed: seed, negativePrompt: negativePrompt
            )
        )
    }
}
