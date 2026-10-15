//
//  ClipPlan.swift
//  StableAudioKit
//
//  What an engine keeps of what it generates, and how it finishes it: the
//  samples asked for (snapped to whole bars for a musical loop), then a
//  loop crossfade or a de-click fade. Shared by the MLX and Core AI engines
//  so a clip is cut the same way whichever one made it.
//

import Foundation

/// How much of a generation to keep and how to finish it.
public struct ClipPlan: Sendable, Equatable {

    /// Frames (per channel) to keep from the start of the generation.
    public let keepFrames: Int

    /// Make a seamless loop.
    public let loop: Bool

    /// For a bar-locked musical loop, the short de-click crossfade; `nil` for
    /// a texture loop (long blend) or a one-shot.
    public let barLockedCrossfadeMs: Double?

    /// - Parameters:
    ///   - options: The request.
    ///   - maxSeconds: The longest clip the engine makes with this model.
    ///   - sampleRate: Output samples per second.
    public init(options: GenerationOptions, maxSeconds: Double, sampleRate: Int = 44_100) {
        let seconds = min(max(options.seconds, 1), maxSeconds)
        var keep = Int((seconds * Double(sampleRate)).rounded())
        var crossfade: Double?
        // A musical loop is snapped to a whole number of bars at the tempo,
        // so the loop point lands on a downbeat; `seamlessLoop` then trims
        // the short de-click crossfade back off.
        if options.loop, let bpm = options.loopBPM, bpm > 20 {
            let bar = 4.0 * 60.0 / bpm
            let bars = max(1.0, (seconds / bar).rounded())
            let loopFrames = Int((bars * bar * Double(sampleRate)).rounded())
            keep = loopFrames + Int((0.012 * Double(sampleRate)).rounded())
            crossfade = 12
        }
        keepFrames = keep
        loop = options.loop
        barLockedCrossfadeMs = crossfade
    }

    /// The seconds of audio the plan keeps.
    public func keptSeconds(sampleRate: Int = 44_100) -> Double { Double(keepFrames) / Double(sampleRate) }

    /// `audio` cut to ``keepFrames`` and finished: a bar-locked loop, a
    /// texture loop, or a one-shot with only a de-click fade.
    public func finish(_ audio: GeneratedAudio) -> GeneratedAudio {
        let channels = audio.channelCount
        let keep = min(audio.frameCount, keepFrames)
        let cut = keep == audio.frameCount ? audio : GeneratedAudio(
            samples: Array(audio.samples[0 ..< keep * channels]), sampleRate: audio.sampleRate,
            channelCount: channels, seed: audio.seed)
        if let crossfade = barLockedCrossfadeMs { return cut.seamlessLoop(crossfadeMs: crossfade) }
        if loop { return cut.seamlessLoop() }
        return cut.faded(inMs: 5, outMs: 8)
    }
}
