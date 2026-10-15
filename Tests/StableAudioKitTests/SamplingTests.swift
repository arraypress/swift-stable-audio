//
//  SamplingTests.swift
//  StableAudioKit
//
//  The host-side arithmetic the Core AI engine runs between network calls.
//

import Foundation
import Testing
@testable import StableAudioKit

@Test func eightStepScheduleIsTheDistilledOne() {
    // The schedule the MLX engine hard-codes (SAPipeline.sigmas8).
    let distilled: [Double] = [1.0, 0.9943756, 0.9844802, 0.95791227, 0.8909032, 0.7455466, 0.51249737, 0.273885, 0.0]
    let schedule = PingPongSampler.schedule(steps: 8)
    #expect(schedule.count == 9)
    for (a, b) in zip(schedule, distilled) { #expect(abs(a - b) < 1e-6) }
}

@Test func scheduleRunsFromSigmaMaxToZero() {
    let schedule = PingPongSampler.schedule(steps: 20, sigmaMax: 0.6)
    #expect(schedule.first == 0.6)
    #expect(schedule.last == 0)
    #expect(schedule.count == 21)
    // After the pinned start the steps descend.
    #expect(zip(schedule.dropFirst(), schedule.dropFirst(2)).allSatisfy { $0 > $1 })
}

@Test func windowNeverDropsBelowTheTrainedMinimum() {
    #expect(PingPongSampler.windowFrames(seconds: 2) == 256)
    #expect(PingPongSampler.windowFrames(seconds: 20) == 256)
    // 30 s + 3 s of margin: round(33 × 44100 / 4096) = 355.
    #expect(PingPongSampler.windowFrames(seconds: 30) == 355)
    #expect(abs(PingPongSampler.seconds(ofFrames: 256) - 23.777) < 0.001)
}

@Test func shortestDecoderThatFits() {
    let ladder = [2048, 256, 512, 1024]
    #expect(PingPongSampler.rung(for: 256, among: ladder) == 256)
    #expect(PingPongSampler.rung(for: 257, among: ladder) == 512)
    #expect(PingPongSampler.rung(for: 2049, among: ladder) == nil)
}

@Test func pingPongStepRenoisesTheDenoisedEstimate() {
    var x: [Double] = [1, -2]
    PingPongSampler.step(&x, velocity: [0.5, 1], now: 0.8, next: 0.25, noise: [2, -1])
    // denoised = x − 0.8·v = [0.6, −2.8]; x' = 0.75·denoised + 0.25·noise
    #expect(abs(x[0] - 0.95) < 1e-9)
    #expect(abs(x[1] - -2.35) < 1e-9)
}

@Test func guidanceAtOneIsTheConditionalVelocity() {
    let x: [Double] = [0.3, -0.1]
    let v = PingPongSampler.guided(x: x, t: 0.7, conditional: [1, 2], unconditional: [5, -5], scale: 1)
    #expect(abs(v[0] - 1) < 1e-6 && abs(v[1] - 2) < 1e-6)
    // Vanilla CFG: v_cond + (s − 1)(v_cond − v_uncond).
    let g = PingPongSampler.guided(x: x, t: 0.7, conditional: [1, 2], unconditional: [5, -5], scale: 3)
    #expect(abs(g[0] - (1 + 2 * (1 - 5))) < 1e-5)
    #expect(abs(g[1] - (2 + 2 * (2 + 5))) < 1e-5)
}

@Test func seededNoiseRepeatsAndIsStandardNormal() {
    var a = GaussianNoise(seed: 7), b = GaussianNoise(seed: 7), c = GaussianNoise(seed: 8)
    let first = a.next(count: 100_000)
    #expect(first == b.next(count: 100_000))
    #expect(first != c.next(count: 100_000))
    let mean = first.reduce(0) { $0 + Double($1) } / Double(first.count)
    let variance = first.reduce(0) { $0 + (Double($1) - mean) * (Double($1) - mean) } / Double(first.count)
    #expect(abs(mean) < 0.01)
    #expect(abs(variance - 1) < 0.02)
}

@Test func clipPlanKeepsWhatWasAskedFor() {
    let plan = ClipPlan(options: GenerationOptions(seconds: 2), maxSeconds: 120)
    #expect(plan.keepFrames == 88_200)
    let audio = GeneratedAudio(samples: [Float](repeating: 0.5, count: 200_000 * 2), sampleRate: 44_100, channelCount: 2, seed: 3)
    let finished = plan.finish(audio)
    #expect(finished.frameCount == 88_200)
    #expect(finished.seed == 3)
    #expect(finished.samples.first == 0)               // the de-click fade-in
}

@Test func clipPlanSnapsAMusicalLoopToWholeBars() {
    // 120 BPM: a bar is 2 s, so 4.6 s rounds to 2 bars = 4 s, plus 12 ms of crossfade.
    let plan = ClipPlan(options: GenerationOptions(seconds: 4.6, loop: true, loopBPM: 120), maxSeconds: 120)
    #expect(plan.keepFrames == 176_400 + 529)
    #expect(plan.barLockedCrossfadeMs == 12)
    let audio = GeneratedAudio(samples: [Float](repeating: 0.1, count: 300_000 * 2), sampleRate: 44_100, channelCount: 2)
    #expect(plan.finish(audio).frameCount == 176_400)
}

@Test func clipPlanCapsAtTheEnginesLongest() {
    #expect(ClipPlan(options: GenerationOptions(seconds: 500), maxSeconds: 187).keepFrames == 187 * 44_100)
    #expect(ClipPlan(options: GenerationOptions(seconds: 0.2), maxSeconds: 187).keepFrames == 44_100)
}
