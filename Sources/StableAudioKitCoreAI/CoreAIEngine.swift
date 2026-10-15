//
//  CoreAIEngine.swift
//  StableAudioKitCoreAI
//
//  Stable Audio 3 through Apple's Core AI: the `.aimodel` exports published
//  as arraypress/stable-audio-3-{small-sfx,small-music,medium}-coreai. One
//  asset holds the networks as named functions; this runs them and does the
//  rest on the host, transcribed from the export's own `example.py`:
//
//      prompt ─▶ condition ─▶ denoise × steps (ping-pong, optional CFG) ─▶ decode_N ─▶ audio
//
//  The transformer takes any length; the decoder comes as a ladder of fixed
//  lengths, and the shortest that fits is used. Everything is float32.
//  Core AI specialises an asset for the device on its first load and caches
//  the result per program (~/Library/Caches/coreai-cache), so the first load
//  of a model is slow and every later one is quick.
//

import CoreAI
import Foundation
import StableAudioKit

/// On-device Stable Audio 3 through Core AI (macOS 27 / iOS 27).
@available(macOS 27.0, iOS 27.0, *)
public final class CoreAIEngine: AudioGenEngine, Sendable {

    /// The model the asset holds — its defaults and limits.
    public let model: StableAudioModel

    /// The fixed lengths, in latent frames, the asset can decode.
    public let decoderLengths: [Int]

    /// Seconds of margin generated past what is kept, so the tone the model
    /// puts on its last frame is cut away.
    public static let margin = 3.0

    private let tokenizer: any SATokenizer
    private let asset: AIModel
    private let conditioner: InferenceFunction
    private let denoiser: InferenceFunction
    private let decoderPrefix: String

    /// The conditioner's fixed prompt length.
    public static let promptLength = 256

    /// Loads (and on its first load in this program, compiles) an asset.
    ///
    /// - Parameters:
    ///   - url: The `.aimodel`.
    ///   - model: Which Stable Audio 3 model it is.
    ///   - tokenizer: The T5Gemma tokenizer from the export's `tokenizer/` folder.
    ///   - options: Core AI's specialisation; the GPU is preferred.
    public init(contentsOf url: URL, model: StableAudioModel, tokenizer: any SATokenizer,
                options: SpecializationOptions = SpecializationOptions(preferredComputeUnitKind: .gpu)) async throws {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw AudioGenError.unavailable("no Core AI asset at \(url.path)")
        }
        let asset: AIModel
        do { asset = try await AIModel(contentsOf: url, options: options) } catch {
            throw AudioGenError.unavailable("\(url.lastPathComponent) did not load: \(error)")
        }
        let names = asset.functionNames
        func load(_ candidates: [String]) throws -> InferenceFunction {
            for name in candidates where names.contains(name) {
                if let function = try asset.loadFunction(named: name) { return function }
            }
            throw AudioGenError.unavailable("\(url.lastPathComponent) has no \(candidates.joined(separator: "/")) function "
                                            + "(it has \(names.sorted().joined(separator: ", ")))")
        }
        let ladder = Self.ladder(names)
        guard !ladder.lengths.isEmpty else {
            throw AudioGenError.unavailable("\(url.lastPathComponent) has no decode_N functions")
        }
        self.model = model
        self.tokenizer = tokenizer
        self.asset = asset
        self.conditioner = try load(["condition", "conditioner"])
        self.denoiser = try load(["denoise", "dit"])
        self.decoderLengths = ladder.lengths
        self.decoderPrefix = ladder.prefix
    }

    /// The decoder functions in `names`: their prefix and lengths.
    static func ladder(_ names: [String]) -> (prefix: String, lengths: [Int]) {
        for prefix in ["decode_", "decoder_"] {
            let lengths = names.compactMap { $0.hasPrefix(prefix) ? Int($0.dropFirst(prefix.count)) : nil }.sorted()
            if !lengths.isEmpty { return (prefix, lengths) }
        }
        return ("decode_", [])
    }

    public var isReady: Bool { true }

    /// The longest clip this asset makes: the model's own limit, or what the
    /// longest decoder holds less the margin, whichever is shorter.
    public var maxSeconds: Double {
        min(model.maxSeconds, PingPongSampler.seconds(ofFrames: decoderLengths.last ?? 0) - Self.margin)
    }

    // MARK: - Generation

    public func generate(prompt: String, options: GenerationOptions) async throws -> GeneratedAudio {
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw AudioGenError.invalidInput("empty prompt")
        }
        let plan = ClipPlan(options: options, maxSeconds: maxSeconds, sampleRate: PingPongSampler.sampleRate)
        let frames = min(PingPongSampler.windowFrames(seconds: plan.keptSeconds(), margin: Self.margin),
                         decoderLengths.last ?? 0)
        let steps = options.steps ?? model.defaultSteps
        let scale = options.cfgScale ?? model.defaultCFG
        guard steps > 0 else { throw AudioGenError.invalidInput("steps must be at least 1") }

        // No seed asked for is a random one, reported back on the audio.
        let seed = options.resolvedSeed()
        var noise = GaussianNoise(seed: UInt64(max(0, seed)))
        let audio = try await render(prompt: prompt, negativePrompt: options.negativePrompt, frames: frames,
                                     steps: steps, cfgScale: scale, noise: &noise)
        return plan.finish(GeneratedAudio(samples: Self.interleaved(audio, channels: 2),
                                          sampleRate: PingPongSampler.sampleRate, channelCount: 2, seed: seed))
    }

    /// One whole generation at the level the export's `example.py` works:
    /// a window of `frames`, the conditioner told that window's length.
    ///
    /// - Parameters:
    ///   - negativePrompt: Steered away from when `cfgScale` is above 1;
    ///     without one, guidance runs against an empty (zero) conditioning,
    ///     as the reference does.
    ///   - noise: The start latent and each step's noise, drawn in that order.
    /// - Returns: Planar stereo, `[2][frames × 4096]` flattened channel-first.
    public func render(prompt: String, negativePrompt: String?, frames: Int, steps: Int, cfgScale: Double,
                       noise: inout some LatentNoise) async throws -> [Float] {
        let seconds = PingPongSampler.seconds(ofFrames: frames)
        let conditioning = try await condition(prompt, seconds: seconds)
        var unconditional: Conditioning?
        if cfgScale != 1 {
            if let negativePrompt {
                unconditional = try await condition(negativePrompt, seconds: seconds)
            } else {
                unconditional = try conditioning.nulled()
            }
        }
        let latent = try await sample(conditioning: conditioning, unconditional: unconditional, frames: frames,
                                      steps: steps, cfgScale: cfgScale, noise: &noise)
        return try await decode(latent, frames: frames)
    }

    // MARK: - The stages

    /// What the conditioner makes of a prompt and a duration.
    public struct Conditioning: Sendable {
        public let crossAttention: NDArray
        public let crossAttentionMask: NDArray
        public let global: NDArray

        /// The reference's unconditional input: zeros in place of the prompt,
        /// the same mask and duration.
        public func nulled() throws -> Conditioning {
            Conditioning(crossAttention: try Tensors.zeros(like: crossAttention),
                         crossAttentionMask: crossAttentionMask, global: global)
        }
    }

    /// The prompt as the conditioner takes it: 256 ids and a 0/1 mask.
    public func tokens(_ prompt: String) -> (ids: [Int32], mask: [Int32]) {
        let ids = tokenizer.encode(prompt).prefix(Self.promptLength).map(Int32.init)
        let padding = Self.promptLength - ids.count
        return (ids + [Int32](repeating: 0, count: padding),
                [Int32](repeating: 1, count: ids.count) + [Int32](repeating: 0, count: padding))
    }

    /// Runs the conditioner.
    public func condition(_ prompt: String, seconds: Double) async throws -> Conditioning {
        let (ids, mask) = tokens(prompt)
        var out = try await run(conditioner, "condition", [
            "input_ids": Tensors.ints(ids, shape: [1, Self.promptLength]),
            "attention_mask": Tensors.ints(mask, shape: [1, Self.promptLength]),
            "seconds": Tensors.floats([Float(seconds)], shape: [1]),
        ])
        return Conditioning(crossAttention: try take(&out, "cross_attn_cond"),
                            crossAttentionMask: try take(&out, "cross_attn_mask"),
                            global: try take(&out, "global_cond"))
    }

    /// One transformer evaluation: the velocity at `t` for latent `x`
    /// (`[256 × frames]`, row-major), text-to-audio (no inpainting context).
    public func velocity(_ x: [Double], t: Double, frames: Int, conditioning: Conditioning,
                         local: NDArray? = nil) async throws -> [Float] {
        let local = local ?? Self.emptyLocal(frames: frames)
        var out = try await run(denoiser, "denoise", [
            "x": Tensors.floats(x, shape: [1, PingPongSampler.latentChannels, frames]),
            "t": Tensors.floats([Float(t)], shape: [1]),
            "cross_attn_cond": conditioning.crossAttention,
            "cross_attn_mask": conditioning.crossAttentionMask,
            "global_cond": conditioning.global,
            "local_add_cond": local,
        ])
        return try Tensors.values(try take(&out, "v"))
    }

    /// The ping-pong sampler over the transformer, from noise to a clean latent.
    public func sample(conditioning: Conditioning, unconditional: Conditioning?, frames: Int, steps: Int,
                       cfgScale: Double, noise: inout some LatentNoise) async throws -> [Double] {
        let count = PingPongSampler.latentChannels * frames
        let local = Self.emptyLocal(frames: frames)
        let t = PingPongSampler.schedule(steps: steps)
        var x = noise.next(count: count).map(Double.init)
        for i in 0..<steps {
            try Task.checkCancellation()
            var v = try await velocity(x, t: t[i], frames: frames, conditioning: conditioning, local: local)
            if let unconditional {
                let vUncond = try await velocity(x, t: t[i], frames: frames, conditioning: unconditional, local: local)
                v = PingPongSampler.guided(x: x, t: t[i], conditional: v, unconditional: vUncond, scale: cfgScale)
            }
            PingPongSampler.step(&x, velocity: v, now: t[i], next: t[i + 1], noise: noise.next(count: count))
        }
        return x
    }

    /// Decodes a latent through the shortest decoder that holds it: zero
    /// padded up to the rung, the audio trimmed back to `frames × 4096`.
    ///
    /// - Returns: Planar stereo, channel 0 then channel 1.
    public func decode(_ latent: [Double], frames: Int) async throws -> [Float] {
        guard let rung = PingPongSampler.rung(for: frames, among: decoderLengths) else {
            throw AudioGenError.invalidInput("\(frames) frames is longer than the longest decoder (\(decoderLengths.last ?? 0))")
        }
        guard let decoder = try asset.loadFunction(named: "\(decoderPrefix)\(rung)") else {
            throw AudioGenError.unavailable("no \(decoderPrefix)\(rung) function")
        }
        let channels = PingPongSampler.latentChannels
        var padded = [Float](repeating: 0, count: channels * rung)
        for c in 0..<channels {
            for f in 0..<frames { padded[c * rung + f] = Float(latent[c * frames + f]) }
        }
        var out = try await run(decoder, "decode", ["latent": Tensors.floats(padded, shape: [1, channels, rung])])
        let audio = try take(&out, "audio")
        let full = try Tensors.values(audio)
        let length = audio.shape.last ?? 0
        let keep = min(length, frames * PingPongSampler.samplesPerFrame)
        return (0..<2).flatMap { channel in full[(channel * length)..<(channel * length + keep)] }
    }

    // MARK: - Helpers

    /// `local_add_cond` for plain text-to-audio: `[mask ; masked input]`, all zeros.
    static func emptyLocal(frames: Int) -> NDArray {
        Tensors.floats([Float](repeating: 0, count: (PingPongSampler.latentChannels + 1) * frames),
                       shape: [1, PingPongSampler.latentChannels + 1, frames])
    }

    /// A float32 tensor's values in row-major order, whatever its strides.
    public static func values(_ array: NDArray) throws -> [Float] { try Tensors.values(array) }

    /// Planar `[channels][n]` as interleaved samples.
    public static func interleaved(_ planar: [Float], channels: Int) -> [Float] {
        let n = planar.count / channels
        var out = [Float](repeating: 0, count: n * channels)
        for c in 0..<channels {
            for i in 0..<n { out[i * channels + c] = planar[c * n + i] }
        }
        return out
    }

    private func run(_ function: InferenceFunction, _ name: String,
                     _ inputs: [String: NDArray]) async throws -> InferenceFunction.Outputs {
        do { return try await function.run(inputs: inputs) } catch {
            throw AudioGenError.unavailable("\(name) failed: \(error)")
        }
    }

    private func take(_ outputs: inout InferenceFunction.Outputs, _ name: String) throws -> NDArray {
        guard let array = outputs.remove(name)?.ndArray else {
            throw AudioGenError.unavailable("the model returned no \(name)")
        }
        return array
    }
}
