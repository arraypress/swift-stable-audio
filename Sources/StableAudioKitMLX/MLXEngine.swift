//
//  MLXEngine.swift
//  StableAudioKitMLX
//
//  On-device Stable Audio via MLX (Mac + iPhone GPU). Loads the converted
//  clean-keyed weights, builds the ported nets, and runs the full pipeline:
//
//      prompt ─▶ [T5Gemma] ─▶ conds ─▶ [DiT ×8 ping-pong] ─▶ [Decoder] ─▶ audio
//
//  The neural code is transcribed 1:1 from the MLX-Python port that was matched
//  to PyTorch offline. Runtime validation of the Swift build needs an Xcode app
//  (mlx-swift's Metal library only bundles in app builds, not `swift test`).
//
//  A `final class` (not an actor) because MLX graph objects are non-Sendable;
//  `@unchecked Sendable` documents prepare-then-generate use.
//

import Foundation
import MLX
import StableAudioKit

/// Supplies Gemma token ids for a prompt (fulfil with swift-transformers in the app).
public protocol SATokenizer: Sendable {
    /// Encode to token ids (BOS-prefixed, as the T5Gemma conditioner expects).
    func encode(_ text: String) -> [Int]
}

/// On-device text-to-audio via the MLX Stable Audio 3 small-sfx model.
public final class MLXEngine: AudioGenEngine, @unchecked Sendable {

    public let model: StableAudioModel
    /// Injected Gemma tokenizer (required before `generate`).
    public var tokenizer: SATokenizer?

    private var weights: [String: MLXArray]?
    private var pipeline: SAPipeline?

    private let promptLength = 256          // T5Gemma conditioner max_length
    private let sampleRate = 44_100
    private let downsample = 4_096

    public init(_ model: StableAudioModel = .sa3SmallSFX, tokenizer: SATokenizer? = nil) {
        self.model = model
        self.tokenizer = tokenizer
    }

    public var isReady: Bool { pipeline != nil }

    // MARK: - Preparation

    /// Not yet wired: download the converted weights from Hugging Face.
    public func prepare() async throws {
        throw AudioGenError.unavailable(
            "MLXEngine.prepare(): HF download not wired — call prepare(weightsDirectory:) " +
            "with a directory containing the converted sa3 safetensors.")
    }

    /// Load a single converted clean-keyed `.safetensors` and build the pipeline.
    public func prepare(weightsFile: URL) throws {
        try loadAndBuild([weightsFile])
    }

    /// Load all `.safetensors` in a directory and build the pipeline.
    public func prepare(weightsDirectory: URL) throws {
        let shards = try FileManager.default
            .contentsOfDirectory(at: weightsDirectory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }
        guard !shards.isEmpty else {
            throw AudioGenError.unavailable("no .safetensors in \(weightsDirectory.path)")
        }
        try loadAndBuild(shards)
    }

    private func loadAndBuild(_ shards: [URL]) throws {
        var all: [String: MLXArray] = [:]
        for shard in shards { for (k, v) in try loadArrays(url: shard) { all[k] = v } }

        // If the file is quantized (small download for iPhone), dequantize to
        // fp16 at load so the nets run unchanged. `dequantized` is lossless
        // vs the quantized weights; 8-bit is near-lossless vs fp32.
        if let bitsArr = all["quant.bits"], let grpArr = all["quant.group"] {
            let bits = bitsArr.asArray(Int32.self)[0]
            let group = grpArr.asArray(Int32.self)[0]
            var out: [String: MLXArray] = [:]
            for (k, v) in all {
                if k.hasSuffix(".scales") || k.hasSuffix(".biases") || k.hasPrefix("quant.") { continue }
                if let s = all[k + ".scales"], let b = all[k + ".biases"] {
                    out[k] = dequantized(v, scales: s, biases: b, groupSize: Int(group), bits: Int(bits)).asType(.float16)
                } else {
                    out[k] = v.asType(.float16)
                }
            }
            all = out
        }
        self.weights = all

        func sub(_ prefix: String) -> [String: MLXArray] {
            var d: [String: MLXArray] = [:]
            for (k, v) in all where k.hasPrefix(prefix) { d[String(k.dropFirst(prefix.count))] = v }
            return d
        }
        guard let pad = all["padding_embedding"], let nw = all["num_w"], let nb = all["num_b"] else {
            throw AudioGenError.unavailable("weights missing conditioning tensors")
        }
        self.pipeline = SAPipeline(
            t5: SAT5Gemma(w: sub("t5.")),
            dit: SADiT(w: sub("dit.")),
            decoder: SADecoder(w: sub("dec.")),
            paddingEmbedding: pad, numberWeight: nw, numberBias: nb)
    }

    // MARK: - Generation

    public func generate(prompt: String, options: GenerationOptions) async throws -> GeneratedAudio {
        guard let pipeline else { throw AudioGenError.notReady }
        guard let tokenizer else { throw AudioGenError.unavailable("set MLXEngine.tokenizer (a Gemma tokenizer)") }
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw AudioGenError.invalidInput("empty prompt")
        }

        // Tokenize -> padded ids + mask (max_length 256).
        var ids = tokenizer.encode(prompt)
        if ids.count > promptLength { ids = Array(ids.prefix(promptLength)) }
        let valid = ids.count
        let idsPadded = ids + Array(repeating: 0, count: promptLength - valid)
        let mask = (0 ..< promptLength).map { Int32($0 < valid ? 1 : 0) }
        let idsMLX = MLXArray(idsPadded.map { Int32($0) }).reshaped([1, promptLength])
        let maskMLX = MLXArray(mask).reshaped([1, promptLength])

        // How many output samples to keep. For a musical loop we snap to an
        // exact whole number of bars at the requested BPM so the loop point
        // lands on a downbeat; `seamlessLoop` then trims the short de-click
        // crossfade back to exactly `loopSamples`.
        let reqSeconds = Double(min(max(options.seconds, 1), model.maxSeconds))
        var keepSamples = Int((reqSeconds * Double(sampleRate)).rounded())
        var barLockedXfMs: Double?
        if options.loop, let bpm = options.loopBPM, bpm > 20 {
            let barSec = 4.0 * 60.0 / bpm                       // 4 beats / bar
            let bars = max(1.0, (reqSeconds / barSec).rounded())
            let loopSamples = Int((bars * barSec * Double(sampleRate)).rounded())
            let xf = Int((0.012 * Double(sampleRate)).rounded())   // 12 ms de-click
            keepSamples = loopSamples + xf
            barLockedXfMs = 12
        }

        // Like the reference pipeline, render a few seconds MORE than we keep and
        // crop back — the model puts a tone artifact on its final frame, so the
        // margin frames push that artifact into the discarded tail.
        let condSeconds = Float(Double(keepSamples) / Double(sampleRate))
        let margin: Float = 3
        var frames = Int(((condSeconds + margin) * Float(sampleRate) / Float(downsample)).rounded())
        frames = max(2, frames + (frames & 1))          // round up to even

        let audio = pipeline.generate(
            ids: idsMLX, mask: maskMLX, seconds: condSeconds,
            frames: frames, seed: UInt64(max(0, options.seed ?? 0)))
        eval(audio)

        // (1, 2, Nfull) -> crop to the kept length -> interleaved [Float].
        let keep = min(audio.dim(2), keepSamples)
        let stereoNT = audio[0].transposed(axes: [1, 0])[0 ..< keep]   // (keep, 2)
        let interleaved = stereoNT.reshaped([keep * 2]).asArray(Float.self)
        let clean = GeneratedAudio(samples: interleaved, sampleRate: sampleRate, channelCount: 2)

        if let xfMs = barLockedXfMs {
            return clean.seamlessLoop(crossfadeMs: xfMs)   // exact-bar musical loop
        } else if options.loop {
            return clean.seamlessLoop()                    // texture loop (long blend)
        } else {
            return clean.faded(inMs: 5, outMs: 8)          // one-shot, de-click only
        }
    }
}
