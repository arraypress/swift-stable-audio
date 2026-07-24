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
