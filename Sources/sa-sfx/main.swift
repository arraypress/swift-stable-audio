//
//  main.swift — sa-sfx
//
//  Batch loopable-SFX generator for on-device Stable Audio 3. Picks the model
//  from the weights filename (…medium… → sa3-medium, …music… → small-music,
//  else small-sfx) and writes one WAV per prompt.
//
//  Usage:
//    sa-sfx <tokDir> <weights> <outDir> [seconds] [name prompt] [--noloop]
//    sa-sfx <tokDir> <weights> <outDir> <seconds> --list <file> [--noloop]
//
//  These SA3 models are distilled to a fixed 8-step ping-pong schedule (see
//  SAPipeline.sigmas8) at CFG 1.0 — that IS the model's highest-quality point,
//  so steps/CFG are not exposed. `--noloop` skips the loop crossfade (raw
//  evolving output — the right choice for PaulStretch stretch fuel).
//  A `name` may contain "/" for category subfolders. Existing outputs skip.
//

import Foundation
import MLX
import StableAudioKit
import StableAudioKitMLX
import StableAudioKitGemma

// ---- split argv into positionals + flags ----
var pos: [String] = []
var flags: [String: String] = [:]
var boolFlags: Set<String> = []
do {
    let raw = Array(CommandLine.arguments.dropFirst())
    var i = 0
    let valued: Set<String> = ["--list"]
    while i < raw.count {
        let a = raw[i]
        if valued.contains(a), i + 1 < raw.count { flags[a] = raw[i + 1]; i += 2 }
        else if a.hasPrefix("--") { boolFlags.insert(a); i += 1 }
        else { pos.append(a); i += 1 }
    }
}
guard pos.count >= 3 else {
    fatalError("usage: sa-sfx <tokDir> <weights> <outDir> [seconds] [--list f | name prompt] [--noloop]")
}

let tokDir  = URL(fileURLWithPath: pos[0])
let weights = URL(fileURLWithPath: pos[1])
let outDir  = URL(fileURLWithPath: pos[2], isDirectory: true)
let seconds = pos.count > 3 ? (Double(pos[3]) ?? 30) : 30
let doLoop  = !boolFlags.contains("--noloop")

let wname = weights.lastPathComponent.lowercased()
let model: StableAudioModel =
    wname.contains("medium") ? .sa3Medium :
    wname.contains("music")  ? .sa3SmallMusic : .sa3SmallSFX

let builtIn: [(String, String)] = [
    ("rain-metal-roof",   "Heavy rain falling steadily on a metal roof, constant downpour, no music, field recording"),
    ("campfire",          "Crackling campfire, wood popping and hissing, close warm ambience, no music"),
    ("thunderstorm",      "Distant thunderstorm with steady rain and low rolling thunder, calm night, no music"),
    ("ocean-waves",       "Gentle ocean waves washing onto a sandy beach, slow calm swell, no music, field recording"),
    ("forest-night",      "Forest at night, crickets chirping and soft wind through leaves, no music"),
    ("wind-howling",      "Howling wind over open moorland, airy low drone, gusts, no music"),
    ("stream",            "A gentle babbling stream trickling over rocks, water bubbling, no music, field recording"),
    ("fireplace-roaring", "Roaring indoor fireplace, crackling burning logs, cozy room tone, no music"),
]

func parseList(_ url: URL) throws -> [(String, String)] {
    let text = try String(contentsOf: url, encoding: .utf8)
    return text.split(separator: "\n").compactMap { raw in
        let line = raw.trimmingCharacters(in: .whitespaces)
        if line.isEmpty || line.hasPrefix("#") { return nil }
        let parts = line.components(separatedBy: "\t")
        guard parts.count >= 2 else { return nil }
        let name = parts[0].trimmingCharacters(in: .whitespaces)
        let prompt = parts[1...].joined(separator: " ").trimmingCharacters(in: .whitespaces)
        return name.isEmpty || prompt.isEmpty ? nil : (name, prompt)
    }
}

let jobs: [(String, String)]
if let listPath = flags["--list"] {
    jobs = try parseList(URL(fileURLWithPath: listPath))
} else if pos.count >= 6 {
    jobs = [(pos[4], pos[5])]
} else {
    jobs = builtIn
}

try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

print("model:    \(model.displayName)  (\(model.id))  [8-step ping-pong, CFG 1.0 — model's max-quality point]")
print("weights:  \(weights.lastPathComponent)")
print("out:      \(outDir.path)")
print("seconds:  \(seconds)   loop: \(doLoop ? "texture-crossfade" : "off (raw)")   jobs: \(jobs.count)")
print("loading tokenizer + weights…")

let tok = try await GemmaTokenizer(modelFolder: tokDir)
let engine = MLXEngine(model, tokenizer: tok)
try engine.prepare(weightsFile: weights)
print("ready.\n")

let fm = FileManager.default
var ok = 0, fail = 0, skip = 0
let batchStart = Date()
for (i, (name, prompt)) in jobs.enumerated() {
    let dst = outDir.appendingPathComponent("\(name).wav")
    if fm.fileExists(atPath: dst.path) {
        print("[\(i + 1)/\(jobs.count)] \(name) — exists, skip"); skip += 1; continue
    }
    print("[\(i + 1)/\(jobs.count)] \(name)")
    let t0 = Date()
    do {
        let audio = try await engine.generate(prompt: prompt, options: GenerationOptions(
            seconds: seconds, seed: 0, loop: doLoop, loopBPM: nil))
        try fm.createDirectory(at: dst.deletingLastPathComponent(), withIntermediateDirectories: true)
        try audio.wavData.write(to: dst)
        print(String(format: "    OK  %.1fs gen -> %.1fs audio", -t0.timeIntervalSinceNow, audio.duration))
        ok += 1
    } catch {
        print("    FAIL \(name): \(error)"); fail += 1
    }
}
print(String(format: "\ndone: %d ok, %d skip, %d fail in %.0fs", ok, skip, fail, -batchStart.timeIntervalSinceNow))
