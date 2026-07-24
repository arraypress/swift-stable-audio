//
//  AudioGenEngine.swift
//  StableAudioKit
//
//  The one protocol every backend implements — on-device (MLX / LiteRT) or
//  cloud (RemoteEngine). The model/tier becomes a swappable choice, not a
//  rewrite, exactly like LLMKit's `LLMEngine`.
//

import Foundation

/// A text-to-audio backend.
public protocol AudioGenEngine: Sendable {

    /// Whether the engine is loaded and ready to `generate`.
    var isReady: Bool { get }

    /// Download / load whatever the engine needs (no-op for cloud engines).
    func prepare() async throws

    /// Generate audio for `prompt` under `options`.
    func generate(prompt: String, options: GenerationOptions) async throws -> GeneratedAudio
}

public extension AudioGenEngine {
    /// Cloud engines need no preparation; on-device engines override this.
    func prepare() async throws {}
}
