//
//  GeneratedAudio.swift
//  StableAudioKit
//
//  A block of generated PCM audio, with lossless WAV (16-bit) encode/decode
//  so a result can be saved, played, or round-tripped without extra deps.
//

import Foundation

/// Generated PCM audio: interleaved float samples in the range `[-1, 1]`.
public struct GeneratedAudio: Sendable, Equatable {

    /// Interleaved samples (`frame0_ch0, frame0_ch1, frame1_ch0, …`), each `[-1, 1]`.
    public let samples: [Float]

    /// Samples per second (e.g. 44_100).
    public let sampleRate: Int

    /// Number of interleaved channels (1 = mono, 2 = stereo).
    public let channelCount: Int

    public init(samples: [Float], sampleRate: Int, channelCount: Int) {
        self.samples = samples
        self.sampleRate = sampleRate
        self.channelCount = channelCount
    }

    /// Frames per channel.
    public var frameCount: Int { channelCount > 0 ? samples.count / channelCount : 0 }

    /// Duration in seconds.
    public var duration: Double {
        (sampleRate > 0) ? Double(frameCount) / Double(sampleRate) : 0
    }
}

// MARK: - WAV (PCM 16-bit) encode / decode

extension GeneratedAudio {

    /// Encode as a 16-bit PCM WAV file.
    public var wavData: Data {
        let bytesPerSample = 2
        let byteRate = sampleRate * channelCount * bytesPerSample
        let blockAlign = channelCount * bytesPerSample
        let dataSize = samples.count * bytesPerSample

        var d = Data(capacity: 44 + dataSize)
        func u32(_ v: UInt32) { var x = v.littleEndian; withUnsafeBytes(of: &x) { d.append(contentsOf: $0) } }
        func u16(_ v: UInt16) { var x = v.littleEndian; withUnsafeBytes(of: &x) { d.append(contentsOf: $0) } }

        d.append(contentsOf: Array("RIFF".utf8)); u32(UInt32(36 + dataSize))
        d.append(contentsOf: Array("WAVE".utf8))
        d.append(contentsOf: Array("fmt ".utf8)); u32(16); u16(1)          // PCM
        u16(UInt16(channelCount)); u32(UInt32(sampleRate)); u32(UInt32(byteRate))
        u16(UInt16(blockAlign)); u16(16)                                    // bits/sample
        d.append(contentsOf: Array("data".utf8)); u32(UInt32(dataSize))

        for s in samples {
            let clamped = max(-1, min(1, s))
            let v = Int16(clamping: Int((clamped * 32767).rounded()))
            u16(UInt16(bitPattern: v))
        }
        return d
    }

    /// Decode a 16-bit PCM WAV file (chunk-scanning; ignores non-`fmt`/`data` chunks).
    public init(wav: Data) throws {
        let b = [UInt8](wav)
        func u32(_ o: Int) -> Int { Int(b[o]) | Int(b[o + 1]) << 8 | Int(b[o + 2]) << 16 | Int(b[o + 3]) << 24 }
        func u16(_ o: Int) -> Int { Int(b[o]) | Int(b[o + 1]) << 8 }

        guard b.count >= 44,
              b[0] == 0x52, b[1] == 0x49, b[2] == 0x46, b[3] == 0x46,      // "RIFF"
              b[8] == 0x57, b[9] == 0x41, b[10] == 0x56, b[11] == 0x45     // "WAVE"
        else { throw AudioGenError.decodingFailed("not a RIFF/WAVE file") }

        var channels = 0, rate = 0, bits = 0
        var dataOffset = -1, dataLen = 0
        var i = 12
        while i + 8 <= b.count {
            let id = String(bytes: b[i..<i + 4], encoding: .ascii) ?? ""
            let sz = u32(i + 4)
            let body = i + 8
            if id == "fmt ", body + 16 <= b.count {
                channels = u16(body + 2); rate = u32(body + 4); bits = u16(body + 14)
            } else if id == "data" {
                dataOffset = body; dataLen = sz
            }
            i = body + sz + (sz & 1)   // chunks are word-aligned
        }

        guard dataOffset >= 0, channels > 0, bits == 16 else {
            throw AudioGenError.decodingFailed("unsupported WAV (need 16-bit PCM with data chunk)")
        }

        let count = min(dataLen, b.count - dataOffset) / 2
        var s = [Float](); s.reserveCapacity(count)
        for k in 0..<count {
            let raw = Int16(bitPattern: UInt16(b[dataOffset + k * 2]) | UInt16(b[dataOffset + k * 2 + 1]) << 8)
            s.append(Float(raw) / 32767.0)
        }
        self.init(samples: s, sampleRate: rate, channelCount: channels)
    }
}
