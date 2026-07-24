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

// MARK: - Post-processing (fades, seamless loop)

extension GeneratedAudio {

    /// Drop the last `ms` milliseconds — Stable Audio appends a brief tone/"horn"
    /// artifact after the content ends; trimming removes it.
    public func trimmingEnd(ms: Double) -> GeneratedAudio {
        let ch = channelCount
        guard ch > 0 else { return self }
        let drop = min(Int(ms / 1000 * Double(sampleRate)), frameCount)
        let keep = max(0, frameCount - drop) * ch
        return GeneratedAudio(samples: Array(samples[0 ..< keep]), sampleRate: sampleRate, channelCount: ch)
    }

    /// Linear fade-in / fade-out (milliseconds) — removes the boundary click or
    /// end "beep" that raw generated clips have. Applied by default on output.
    public func faded(inMs: Double = 20, outMs: Double = 120) -> GeneratedAudio {
        guard channelCount > 0, frameCount > 0 else { return self }
        var s = samples
        let ch = channelCount
        let fin = max(0, min(Int(inMs / 1000 * Double(sampleRate)), frameCount / 2))
        let fout = max(0, min(Int(outMs / 1000 * Double(sampleRate)), frameCount / 2))
        for f in 0 ..< fin {
            let g = Float(f) / Float(fin)
            for c in 0 ..< ch { s[f * ch + c] *= g }
        }
        for f in 0 ..< fout {
            let g = Float(f) / Float(fout)
            let idx = frameCount - 1 - f
            for c in 0 ..< ch { s[idx * ch + c] *= g }
        }
        return GeneratedAudio(samples: s, sampleRate: sampleRate, channelCount: ch)
    }

    /// Turn the clip into a **seamless loop** by equal-gain crossfading the tail
    /// back into the head. The returned clip is `crossfadeMs` shorter and loops
    /// with no click. Best for steady content (rain, wind, fire, drones, pads).
    public func seamlessLoop(crossfadeMs: Double = 250) -> GeneratedAudio {
        let ch = channelCount
        guard ch > 0, frameCount > 4 else { return self }
        let xf = max(1, min(Int(crossfadeMs / 1000 * Double(sampleRate)), frameCount / 3))
        let outLen = frameCount - xf
        var s = [Float](repeating: 0, count: outLen * ch)
        for f in 0 ..< outLen {
            for c in 0 ..< ch { s[f * ch + c] = samples[f * ch + c] }
        }
        // First `xf` samples blend the head (fading in) with the removed tail
        // (fading out), so the wrap point outLen-1 -> 0 is continuous.
        for i in 0 ..< xf {
            let g = Float(i) / Float(xf)
            for c in 0 ..< ch {
                let head = samples[i * ch + c]
                let tail = samples[(outLen + i) * ch + c]
                s[i * ch + c] = head * g + tail * (1 - g)
            }
        }
        return GeneratedAudio(samples: s, sampleRate: sampleRate, channelCount: ch)
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
