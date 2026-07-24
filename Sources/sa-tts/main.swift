//
//  main.swift — sa-tts
//
//  The whole thing: typed prompt -> Gemma tokenizer -> MLXEngine -> audio.
//  Usage: sa-tts <tokenizerFolder> <weights.safetensors> <out.wav> <prompt> [seconds]
//

import Foundation
import MLX
import StableAudioKit
import StableAudioKitMLX
import StableAudioKitGemma

let args = CommandLine.arguments
guard args.count >= 5 else { fatalError("usage: sa-tts <tokDir> <weights> <out.wav> <prompt> [seconds]") }
let prompt = args[4]
let seconds = args.count > 5 ? (Double(args[5]) ?? 7) : 7
let bpm = args.count > 6 ? Double(args[6]) : nil     // optional: bar-locked musical loop

print("loading tokenizer + weights…")
let tok = try await GemmaTokenizer(modelFolder: URL(fileURLWithPath: args[1]))
let engine = MLXEngine(.sa3SmallSFX, tokenizer: tok)
try engine.prepare(weightsFile: URL(fileURLWithPath: args[2]))

print("generating: \"\(prompt)\"")
let t0 = Date()
let audio = try await engine.generate(prompt: prompt, options: GenerationOptions(
    seconds: seconds, seed: 0, loop: bpm != nil, loopBPM: bpm))
print(String(format: "done in %.1fs — %.1fs audio", -t0.timeIntervalSinceNow, audio.duration))

try audio.wavData.write(to: URL(fileURLWithPath: args[3]))
print("wrote \(args[3])")
