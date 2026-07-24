//
//  main.swift — sa-generate
//
//  End-to-end text -> audio via the real MLXEngine. A fixed-ids tokenizer
//  stands in for the swift-transformers Gemma tokenizer (app-level plumbing);
//  everything downstream is the shipping engine path.
//
//  Usage: sa-generate <weights.safetensors> <out.wav> [idsCSV] [seconds]
//

import Foundation
import MLX
import StableAudioKit
import StableAudioKitMLX

struct FixedTokenizer: SATokenizer {
    let ids: [Int]
    func encode(_ text: String) -> [Int] { ids }
}

let args = CommandLine.arguments
guard args.count >= 3 else { fatalError("usage: sa-generate <weights.safetensors> <out.wav> [idsCSV] [seconds]") }
let weightsFile = URL(fileURLWithPath: args[1])
let outWav = args[2]
let idsCSV = args.count > 3 ? args[3] : "530,1057,3360,6588,5163,1280,7148,675,28896"
let ids = idsCSV.split(separator: ",").compactMap { Int($0) }
let seconds = args.count > 4 ? (Double(args[4]) ?? 7) : 7

let engine = MLXEngine(.sa3SmallSFX, tokenizer: FixedTokenizer(ids: ids))
print("loading weights…")
try engine.prepare(weightsFile: weightsFile)

print("generating (\(ids.count) tokens, \(seconds)s)…")
let t0 = Date()
let audio = try await engine.generate(prompt: "prompt", options: GenerationOptions(seconds: seconds, seed: 0))
print(String(format: "done in %.1fs — %.1fs audio, %d Hz, %d ch",
             -t0.timeIntervalSinceNow, audio.duration, audio.sampleRate, audio.channelCount))

try audio.wavData.write(to: URL(fileURLWithPath: outWav))
print("wrote \(outWav)")
