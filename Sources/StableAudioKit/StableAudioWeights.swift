//
//  StableAudioWeights.swift
//  StableAudioKit
//
//  What a converted weights file is: where it came from (the upstream
//  repository and revision `Tools/convert_stream.py --source=` records in
//  the safetensors header) and a digest of its bytes. Generation is
//  deterministic for a given file and seed, so these are what a saved
//  recipe pins: an 8-bit and an fp32 conversion of the same revision make
//  different audio, and only the digest tells them apart.
//

import CryptoKit
import Foundation

/// Facts about a converted `.safetensors` weights file.
public enum StableAudioWeights {

    /// Where a converted file says it came from.
    public struct Source: Sendable, Equatable {

        /// The upstream repository: `stabilityai/stable-audio-3-small-sfx`.
        public let repository: String?

        /// The upstream commit the file was converted from.
        public let revision: String?

        public init(repository: String?, revision: String?) {
            self.repository = repository
            self.revision = revision
        }
    }

    /// The `source_repository` / `source_revision` in the file's
    /// `__metadata__`, each `nil` when the converter did not record it.
    public static func source(of url: URL) throws -> Source {
        let metadata = try header(of: url)["__metadata__"] as? [String: Any] ?? [:]
        return source(fromMetadata: metadata)
    }

    /// ``source(of:)``'s reading of an already-parsed `__metadata__` object.
    public static func source(fromMetadata metadata: [String: Any]) -> Source {
        Source(repository: metadata["source_repository"] as? String,
               revision: metadata["source_revision"] as? String)
    }

    /// The JSON header of a safetensors file: a little-endian u64 length,
    /// then that many bytes of JSON.
    public static func header(of url: URL) throws -> [String: Any] {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        guard let prefix = try handle.read(upToCount: 8), prefix.count == 8 else {
            throw AudioGenError.decodingFailed("\(url.lastPathComponent) is too short to be safetensors")
        }
        let length = prefix.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self).littleEndian }
        // A header past 100 MB is not a header; refuse rather than allocate it.
        guard length > 0, length < 100_000_000,
              let json = try handle.read(upToCount: Int(length)), json.count == Int(length),
              let object = try JSONSerialization.jsonObject(with: json) as? [String: Any]
        else {
            throw AudioGenError.decodingFailed("\(url.lastPathComponent) has no safetensors header")
        }
        return object
    }

    /// The SHA-256 of the whole file, lower-case hex. Reads it in 16 MB
    /// pieces, so memory stays flat however large the file.
    public static func sha256(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let piece = try handle.read(upToCount: 16 << 20), !piece.isEmpty {
            hasher.update(data: piece)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
