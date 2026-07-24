//
//  StableAudioModel.swift
//  StableAudioKit
//
//  The curated set of Stable Audio models, with the metadata an app needs to
//  pick one per device (size / mobile-fit / license) and the sampler defaults
//  each variant expects. Verified locally on Apple Silicon (MPS).
//

import Foundation

/// A downloadable Stable Audio model.
public struct StableAudioModel: Sendable, Equatable {

    /// What the model is specialised for.
    public enum Kind: String, Sendable { case soundEffects, music, general }

    /// The diffusion sampler the model was trained/distilled for.
    public enum Sampler: String, Sendable { case pingpong, dpmppSDE }

    /// Short identifier.
    public let id: String

    /// The Hugging Face repo id.
    public let repoID: String

    /// Human-readable name.
    public let displayName: String

    /// What it's best at.
    public let kind: Kind

    /// Output sample rate (Hz).
    public let sampleRate: Int

    /// Longest clip it can generate (seconds).
    public let maxSeconds: Double

    /// Rough on-disk size once downloaded (MB, order-of-magnitude).
    public let approximateSizeMB: Int

    /// The model-weights license.
    public let license: String

    /// Whether it's light/fast enough to embed and run on iPhone.
    public let runsOnMobile: Bool

    /// Default diffusion steps for this variant.
    public let defaultSteps: Int

    /// Default classifier-free-guidance scale (distilled models use 1.0).
    public let defaultCFG: Double

    /// The sampler this variant expects.
    public let sampler: Sampler

    public init(
        id: String,
        repoID: String,
        displayName: String,
        kind: Kind,
        sampleRate: Int = 44_100,
        maxSeconds: Double,
        approximateSizeMB: Int,
        license: String,
        runsOnMobile: Bool,
        defaultSteps: Int,
        defaultCFG: Double,
        sampler: Sampler
    ) {
        self.id = id
        self.repoID = repoID
        self.displayName = displayName
        self.kind = kind
        self.sampleRate = sampleRate
        self.maxSeconds = maxSeconds
        self.approximateSizeMB = approximateSizeMB
        self.license = license
        self.runsOnMobile = runsOnMobile
        self.defaultSteps = defaultSteps
        self.defaultCFG = defaultCFG
        self.sampler = sampler
    }

    // MARK: - Curated presets (all Stability Community License)

    private static let community = "Stability AI Community License"

    // The on-device family is the Stable Audio 3 models — one shared
    // architecture (T5Gemma + DiT + resampling decoder, 8-step ping-pong).
    // The `small` pair is byte-identical in architecture (only weights differ);
    // `medium` is the same family at larger dims. The older Stable Audio Open
    // models are a different architecture and are superseded by these — kept
    // only as `legacy` reference (cloud tier), not on-device targets.

    /// Stable Audio 3 small-sfx — dedicated SFX, T5Gemma text encoder, up to
    /// 120s, ~1s/clip. The recommended on-device sound-effects model.
    public static let sa3SmallSFX = StableAudioModel(
        id: "sa3-small-sfx", repoID: "stabilityai/stable-audio-3-small-sfx",
        displayName: "Stable Audio 3 Small SFX", kind: .soundEffects,
        maxSeconds: 120, approximateSizeMB: 3_500, license: community,
        runsOnMobile: true, defaultSteps: 8, defaultCFG: 1.0, sampler: .pingpong)

    /// Stable Audio 3 small-music — music sibling of small-sfx (identical
    /// architecture, different weights), up to 120s, ~1s/clip.
    public static let sa3SmallMusic = StableAudioModel(
        id: "sa3-small-music", repoID: "stabilityai/stable-audio-3-small-music",
        displayName: "Stable Audio 3 Small Music", kind: .music,
        maxSeconds: 120, approximateSizeMB: 3_500, license: community,
        runsOnMobile: true, defaultSteps: 8, defaultCFG: 1.0, sampler: .pingpong)

    /// Stable Audio 3 medium — the top open tier; same family at larger dims
    /// (embed 1536 / depth 24 / 24 heads, head_dim 64). Mac-class (heavy).
    public static let sa3Medium = StableAudioModel(
        id: "sa3-medium", repoID: "stabilityai/stable-audio-3-medium",
        displayName: "Stable Audio 3 Medium", kind: .general,
        maxSeconds: 120, approximateSizeMB: 10_400, license: community,
        runsOnMobile: false, defaultSteps: 8, defaultCFG: 1.0, sampler: .pingpong)

    /// The curated on-device family (Stable Audio 3). One codebase.
    public static let all: [StableAudioModel] = [
        .sa3SmallSFX, .sa3SmallMusic, .sa3Medium,
    ]

    // MARK: - Legacy (older architecture; cloud/reference only, not on-device)

    /// Stable Audio Open Small (341M, 2025) — superseded by `sa3SmallSFX`.
    /// Different (T5, non-Gemma) architecture; not an MLX-engine target.
    public static let legacyOpenSmall = StableAudioModel(
        id: "open-small", repoID: "stabilityai/stable-audio-open-small",
        displayName: "Stable Audio Open Small (legacy)", kind: .general,
        maxSeconds: 12, approximateSizeMB: 1_000, license: community,
        runsOnMobile: false, defaultSteps: 8, defaultCFG: 1.0, sampler: .pingpong)

    /// Stable Audio Open 1.0 (1.1B, 2024) — superseded by `sa3Medium`.
    /// Different architecture / full diffusion; cloud tier only.
    public static let legacyOpen1_0 = StableAudioModel(
        id: "open-1.0", repoID: "stabilityai/stable-audio-open-1.0",
        displayName: "Stable Audio Open 1.0 (legacy)", kind: .general,
        maxSeconds: 47, approximateSizeMB: 4_500, license: community,
        runsOnMobile: false, defaultSteps: 100, defaultCFG: 7.0, sampler: .dpmppSDE)

    /// The recommended sound-effects model for a target device.
    public static func recommendedSFX(onMobile: Bool) -> StableAudioModel {
        // sa3-small-sfx is the on-device anchor on both tiers today; the
        // hook is here for when a heavier Mac-only SFX option is added.
        onMobile ? .sa3SmallSFX : .sa3SmallSFX
    }
}
