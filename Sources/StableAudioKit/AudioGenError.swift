//
//  AudioGenError.swift
//  StableAudioKit
//

import Foundation

/// Errors thrown by StableAudioKit engines.
public enum AudioGenError: Error, Sendable, Equatable {
    /// The engine needs `prepare()` before it can generate.
    case notReady
    /// The engine/model isn't available on this device or SDK.
    case unavailable(String)
    /// The prompt or options were invalid.
    case invalidInput(String)
    /// A remote request failed (network / HTTP status).
    case requestFailed(String)
    /// Returned audio could not be decoded.
    case decodingFailed(String)
}
