//
//  main.swift — sa-validate
//
//  Runtime-validate the MLX Swift nets against the PyTorch-derived fixtures.
//  Usage: sa-validate <weights.safetensors> <fixtures.safetensors>
//

import Foundation
import MLX
import StableAudioKitMLX

let args = CommandLine.arguments
guard args.count == 3 else { fatalError("usage: sa-validate <weights> <fixtures>") }

let weights = try loadArrays(url: URL(fileURLWithPath: args[1]))
let fx = try loadArrays(url: URL(fileURLWithPath: args[2]))

func sub(_ prefix: String, _ src: [String: MLXArray]) -> [String: MLXArray] {
    var d: [String: MLXArray] = [:]
    for (k, v) in src where k.hasPrefix(prefix) { d[String(k.dropFirst(prefix.count))] = v }
    return d
}
func report(_ name: String, _ a: MLXArray, _ b: MLXArray) {
    let d = abs(a.asType(.float32) - b.asType(.float32)); eval(d)
    let mx = d.max().item(Float.self), mn = d.mean().item(Float.self)
    let ok = mx < 5e-3 ? "MATCH ✓" : "DIFF"
    print(String(format: "%-9@ max %.3e  mean %.3e   %@", name as NSString, mx, mn, ok as NSString))
}

print("Loaded weights: \(weights.count) tensors, fixtures: \(fx.count)\n")

// Decoder
let decOut = SADecoder(w: sub("dec.", weights)).decode(fx["dec_latent"]!)
report("decoder", decOut, fx["dec_ref_audio"]!)

// DiT
let ditOut = SADiT(w: sub("dit.", weights)).forward(
    fx["dit_x"]!, fx["dit_t"]!, fx["dit_cross_attn_cond"]!, fx["dit_global_embed"]!, fx["dit_local_add_cond"]!)
report("dit", ditOut, fx["dit_output"]!)

// T5Gemma
let t5Out = SAT5Gemma(w: sub("t5.", weights)).encode(fx["t5_ids"]!, fx["t5_mask"]!)
report("t5gemma", t5Out, fx["t5_ref"]!)

print("\ndone.")
