//
//  MLXEngine.swift
//  StableAudioKitMLX
//
//  On-device Stable Audio via MLX (Mac + iPhone GPU). Loads the MLX-converted
//  weights, then runs the diffusion pipeline fully on-device:
//
//      prompt ─▶ [Conditioner] ─▶ conds ─▶ [DiT ×N ping-pong] ─▶ [Decoder] ─▶ audio
//
//  A `final class` (not an actor) because MLX graph objects are non-Sendable;
//  `@unchecked Sendable` documents prepare-then-generate use.
//
//  Architecture reference: `stable_audio_tools` (PyTorch) + the MLX weight map
//  of `mlx-community/stable-audio-3-small-sfx`. See MLX_ENGINE_PLAN.md.
//

import Foundation
import MLX
import StableAudioKit

/// On-device text-to-audio via an MLX Stable Audio model.
public final class MLXEngine: AudioGenEngine, @unchecked Sendable {

    /// The model this engine runs.
    public let model: StableAudioModel

    /// Loaded weights, keyed by tensor name (e.g. `model.model.transformer…`).
    private var weights: [String: MLXArray]?

    /// Create an engine for one of the curated ``StableAudioModel`` presets.
    public init(_ model: StableAudioModel) {
        self.model = model
    }

    public var isReady: Bool { weights != nil }

    // MARK: - Preparation

    /// Download (if needed) and load the model. **Download is not yet wired** —
    /// use ``prepare(weightsDirectory:)`` with a local MLX checkpoint for now.
    public func prepare() async throws {
        throw AudioGenError.unavailable(
            "MLXEngine.prepare(): Hugging Face download not wired yet — " +
            "call prepare(weightsDirectory:) with a local MLX checkpoint."
        )
    }

    /// Load MLX-format weights from a local directory of `.safetensors` shards.
    public func prepare(weightsDirectory: URL) throws {
        let shards = try FileManager.default
            .contentsOfDirectory(at: weightsDirectory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }
        guard !shards.isEmpty else {
            throw AudioGenError.unavailable("no .safetensors found in \(weightsDirectory.path)")
        }
        var all: [String: MLXArray] = [:]
        for shard in shards {
            for (key, value) in try loadArrays(url: shard) { all[key] = value }
        }
        self.weights = all
    }

    // MARK: - Generation

    public func generate(prompt: String, options: GenerationOptions) async throws -> GeneratedAudio {
        guard let weights else { throw AudioGenError.notReady }
        let (steps, cfg, seconds) = options.resolved(for: model)
        _ = (weights, steps, cfg, seconds)   // wired in the forward-pass work

        // Pipeline (see PingPongSampler + the DiT/Conditioner/Decoder ports):
        //   1. cond = Conditioner(weights).encode(prompt, seconds)
        //   2. latent = PingPongSampler(steps, cfg).sample(dit, cond, sampleSize)
        //   3. audio  = Decoder(weights).decode(latent)
        throw AudioGenError.unavailable("MLX forward pass not yet implemented (see MLX_ENGINE_PLAN.md)")
    }
}
