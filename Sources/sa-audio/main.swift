//
//  main.swift — sa-audio
//
//  The whole thing: typed prompt -> Gemma tokenizer -> MLXEngine -> audio.
//  (Named sa-audio, not sa-tts — this is text-to-AUDIO, not speech.)
//  Usage: sa-audio <tokenizerFolder> <weights.safetensors> <out.wav> <prompt> [seconds] [bpm]
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
// arg 6 is optional: a BPM -> bar-locked musical loop; the literal "loop" ->
// texture loop (long equal-power crossfade, no bar cropping) for SFX/ambience.
let loopArg = args.count > 6 ? args[6] : nil
let bpm = loopArg.flatMap(Double.init)
// "none"/"-" = explicitly no loop (lets you pass a seed without looping).
let wantLoop = loopArg != nil && loopArg != "none" && loopArg != "-"
// arg 7: seed. Generation is deterministic, so varying this is the only way to
// get different takes of the SAME prompt.
let seed = args.count > 7 ? (Int(args[7]) ?? 0) : 0

print("loading tokenizer + weights…")
let tok = try await GemmaTokenizer(modelFolder: URL(fileURLWithPath: args[1]))
let wn = URL(fileURLWithPath: args[2]).lastPathComponent.lowercased()
let model: StableAudioModel =
    wn.contains("medium") ? .sa3Medium :
    wn.contains("music")  ? .sa3SmallMusic : .sa3SmallSFX
let engine = MLXEngine(model, tokenizer: tok)
try engine.prepare(weightsFile: URL(fileURLWithPath: args[2]))

print("generating: \"\(prompt)\"")
let t0 = Date()
let audio = try await engine.generate(prompt: prompt, options: GenerationOptions(
    seconds: seconds, seed: seed, loop: wantLoop, loopBPM: bpm))
print(String(format: "done in %.1fs — %.1fs audio", -t0.timeIntervalSinceNow, audio.duration))

try audio.wavData.write(to: URL(fileURLWithPath: args[3]))
print("wrote \(args[3])")
