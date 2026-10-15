//
//  StableAudioKitTests.swift
//  StableAudioKit
//

import Testing
import Foundation
@testable import StableAudioKit

// MARK: - Model registry

@Test func registryIsTheSA3Family() {
    #expect(StableAudioModel.all.count == 3)
    let ids = Set(StableAudioModel.all.map(\.id))
    #expect(ids == ["sa3-small-sfx", "sa3-small-music", "sa3-medium"])
    // legacy Open-series exist but are not in the on-device set
    #expect(!StableAudioModel.all.contains(.legacyOpenSmall))
}

@Test func sa3SmallSFXMetadata() {
    let m = StableAudioModel.sa3SmallSFX
    #expect(m.repoID == "stabilityai/stable-audio-3-small-sfx")
    #expect(m.kind == .soundEffects)
    #expect(m.maxSeconds == 120)
    #expect(m.sampler == .pingpong)
    #expect(m.defaultSteps == 8)
    #expect(m.defaultCFG == 1.0)
    #expect(m.runsOnMobile)
}

@Test func legacyOpen1_0IsHeavyMacTier() {
    let m = StableAudioModel.legacyOpen1_0
    #expect(!m.runsOnMobile)
    #expect(m.sampler == .dpmppSDE)
    #expect(m.defaultSteps == 100)
    #expect(m.defaultCFG == 7.0)
}

@Test func recommendedSFXIsSA3() {
    #expect(StableAudioModel.recommendedSFX(onMobile: true) == .sa3SmallSFX)
    #expect(StableAudioModel.recommendedSFX(onMobile: false) == .sa3SmallSFX)
}

// MARK: - GenerationOptions

@Test func optionsFallBackToModelDefaults() {
    let r = GenerationOptions(seconds: 5).resolved(for: .sa3SmallSFX)
    #expect(r.steps == 8)
    #expect(r.cfgScale == 1.0)
    #expect(r.seconds == 5)
}

@Test func optionsCapSecondsToModelMax() {
    let r = GenerationOptions(seconds: 999).resolved(for: .legacyOpenSmall)
    #expect(r.seconds == 12)          // legacyOpenSmall maxSeconds
}

@Test func optionsOverrideDefaults() {
    let r = GenerationOptions(seconds: 3, steps: 25, cfgScale: 4).resolved(for: .sa3SmallSFX)
    #expect(r.steps == 25)
    #expect(r.cfgScale == 4)
}

// MARK: - GeneratedAudio WAV round-trip

@Test func wavRoundTripPreservesShapeAndSamples() throws {
    // 0.02s stereo ramp
    let sr = 44_100, ch = 2
    var samples = [Float]()
    for i in 0..<(sr / 50) {
        let v = Float(i) / Float(sr / 50) * 2 - 1     // -1 … 1
        samples.append(v); samples.append(-v)
    }
    let audio = GeneratedAudio(samples: samples, sampleRate: sr, channelCount: ch)

    let decoded = try GeneratedAudio(wav: audio.wavData)
    #expect(decoded.sampleRate == sr)
    #expect(decoded.channelCount == ch)
    #expect(decoded.frameCount == audio.frameCount)
    // 16-bit quantization tolerance
    for (a, b) in zip(samples, decoded.samples) {
        #expect(abs(a - b) < 1e-3)
    }
}

@Test func generatedAudioDurationIsCorrect() {
    let audio = GeneratedAudio(samples: [Float](repeating: 0, count: 44_100 * 2), sampleRate: 44_100, channelCount: 2)
    #expect(audio.frameCount == 44_100)
    #expect(abs(audio.duration - 1.0) < 1e-6)
}

@Test func decodingGarbageThrows() {
    #expect(throws: AudioGenError.self) {
        _ = try GeneratedAudio(wav: Data([0, 1, 2, 3]))
    }
}

// MARK: - RemoteEngine request shaping (offline)

@Test func remoteRequestBodyHasPromptAndDuration() {
    let engine = RemoteEngine(.elevenLabsSoundEffects(apiKey: "k"))
    let body = engine.requestBody(prompt: "rain", options: GenerationOptions(seconds: 8))
    #expect(body["text"] as? String == "rain")
    #expect(body["duration_seconds"] as? Double == 8)
}

@Test func remoteRequestHasHeadersAndPOST() throws {
    let engine = RemoteEngine(.elevenLabsSoundEffects(apiKey: "secret"))
    let req = try engine.makeRequest(prompt: "thunder", options: GenerationOptions(seconds: 5))
    #expect(req.httpMethod == "POST")
    #expect(req.url?.absoluteString == "https://api.elevenlabs.io/v1/sound-generation")
    #expect(req.value(forHTTPHeaderField: "xi-api-key") == "secret")
    #expect(req.value(forHTTPHeaderField: "Content-Type") == "application/json")
    #expect(req.httpBody != nil)
}

@Test func remoteRejectsEmptyPrompt() {
    let engine = RemoteEngine(.elevenLabsSoundEffects(apiKey: "k"))
    #expect(throws: AudioGenError.self) {
        _ = try engine.makeRequest(prompt: "   ", options: GenerationOptions())
    }
}

@Test func remoteEngineIsAlwaysReady() {
    #expect(RemoteEngine(.elevenLabsSoundEffects(apiKey: "k")).isReady)
}

// MARK: - Seeds

/// A fixed-sequence generator, so a "random" draw is checkable.
private struct CountingGenerator: RandomNumberGenerator {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

@Test func aGivenSeedIsTheSeedUsed() {
    var generator = CountingGenerator(state: 1)
    #expect(GenerationOptions(seed: 99).resolvedSeed(using: &generator) == 99)
    #expect(GenerationOptions(seed: 0).resolvedSeed() == 0)
}

@Test func noSeedDrawsARandomOneInRange() {
    // The bug this guards: no seed used to mean seed 0, so every run without
    // one made the same sound.
    var generator = CountingGenerator(state: 7)
    let draws = (0..<50).map { _ in GenerationOptions().resolvedSeed(using: &generator) }
    #expect(draws.allSatisfy { GenerationOptions.seedRange.contains($0) })
    #expect(Set(draws).count == 50)
    #expect(!draws.contains(0))
    let system = Set((0..<20).map { _ in GenerationOptions().resolvedSeed() })
    #expect(system.count > 1)
    #expect(GenerationOptions.seedRange == 0...4_294_967_295)
}

@Test func theSeedSurvivesPostProcessing() {
    let audio = GeneratedAudio(samples: [Float](repeating: 0.5, count: 44_100 * 2), sampleRate: 44_100,
                               channelCount: 2, seed: 1234)
    #expect(audio.faded().seed == 1234)
    #expect(audio.seamlessLoop().seed == 1234)
    #expect(audio.trimmingEnd(ms: 10).seed == 1234)
    #expect(GeneratedAudio(samples: [0, 0], sampleRate: 44_100, channelCount: 2).seed == nil)
}

// MARK: - Weights files

/// A tiny safetensors file: one f32 tensor and, optionally, a source.
private func safetensors(metadata: [String: String]?) throws -> URL {
    var header: [String: Any] = ["x": ["dtype": "F32", "shape": [2], "data_offsets": [0, 8]]]
    if let metadata { header["__metadata__"] = metadata }
    var json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
    json.append(contentsOf: [UInt8](repeating: 0x20, count: (8 - json.count % 8) % 8))
    var data = Data()
    var length = UInt64(json.count).littleEndian
    withUnsafeBytes(of: &length) { data.append(contentsOf: $0) }
    data.append(json)
    data.append(contentsOf: [0, 0, 128, 63, 0, 0, 0, 64])     // 1.0, 2.0
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("sa-\(UUID().uuidString).safetensors")
    try data.write(to: url)
    return url
}

@Test func weightsSourceIsReadFromTheHeader() throws {
    let url = try safetensors(metadata: ["source_repository": "stabilityai/stable-audio-3-small-sfx",
                                         "source_revision": "05ca3857881c94a2b84026b1111c537d78f5f9f0"])
    defer { try? FileManager.default.removeItem(at: url) }
    let source = try StableAudioWeights.source(of: url)
    #expect(source.repository == "stabilityai/stable-audio-3-small-sfx")
    #expect(source.revision == "05ca3857881c94a2b84026b1111c537d78f5f9f0")
}

@Test func weightsWithoutASourceSayNothing() throws {
    let url = try safetensors(metadata: nil)
    defer { try? FileManager.default.removeItem(at: url) }
    #expect(try StableAudioWeights.source(of: url) == .init(repository: nil, revision: nil))
}

@Test func weightsDigestIsTheFilesSHA256() throws {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("sa-\(UUID().uuidString).bin")
    try Data("abc".utf8).write(to: url)
    defer { try? FileManager.default.removeItem(at: url) }
    #expect(try StableAudioWeights.sha256(of: url) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
}

@Test func notSafetensorsIsRefused() throws {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("sa-\(UUID().uuidString).bin")
    try Data("abc".utf8).write(to: url)
    defer { try? FileManager.default.removeItem(at: url) }
    #expect(throws: AudioGenError.self) { _ = try StableAudioWeights.source(of: url) }
}
