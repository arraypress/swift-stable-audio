//
//  main.swift — sa-generate
//
//  End-to-end text -> audio via the real MLXEngine. A fixed-ids tokenizer
//  stands in for the swift-transformers Gemma tokenizer (app-level plumbing);
//  everything downstream is the shipping engine path.
//
//  Usage: sa-generate <weightsDir> <out.wav>
//

import Foundation
import MLX
import StableAudioKit
import StableAudioKitMLX

/// Returns pre-tokenized Gemma ids for the demo prompt.
struct FixedTokenizer: SATokenizer {
    let ids: [Int]
    func encode(_ text: String) -> [Int] { ids }
}

let args = CommandLine.arguments
guard args.count == 3 else { fatalError("usage: sa-generate <weightsDir> <out.wav>") }

let prompt = "chugging train coming into station with horn"
let engine = MLXEngine(.sa3SmallSFX,
    tokenizer: FixedTokenizer(ids: [530, 1057, 3360, 6588, 5163, 1280, 7148, 675, 28896]))

print("loading weights…")
try engine.prepare(weightsDirectory: URL(fileURLWithPath: args[1]))

print("generating: \"\(prompt)\"")
let t0 = Date()
let audio = try await engine.generate(prompt: prompt, options: GenerationOptions(seconds: 7, seed: 0))
print(String(format: "done in %.1fs — %.1fs audio, %d Hz, %d ch, %d samples",
             -t0.timeIntervalSinceNow, audio.duration, audio.sampleRate, audio.channelCount, audio.samples.count))

try audio.wavData.write(to: URL(fileURLWithPath: args[2]))
print("wrote \(args[2])")
