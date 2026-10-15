//
//  main.swift — sa-coreai
//
//  The Core AI engine from the command line: load an export folder (the
//  `.aimodel` and its `tokenizer/`), generate, time each part, and dump the
//  stages for the parity check against the export's own example.py.
//
//    sa-coreai info     <export-dir>
//    sa-coreai generate <export-dir> <out.wav> <prompt> [seconds] [seed] [steps] [cfg] [negative]
//    sa-coreai parity   <export-dir> <fixtures-dir>
//
//  `parity` reads what Tools/coreai_parity.py wrote (meta.json, noise.f32,
//  latent.f32) and writes swift_*.f32 beside them for it to compare.
//

import CoreAI
import Foundation
import StableAudioKit
import StableAudioKitCoreAI
import StableAudioKitGemma

@available(macOS 27.0, *)
func main() async throws {

    let args = CommandLine.arguments
    func usage() -> Never {
        print("usage: sa-coreai info|generate|parity <export-dir> …")
        exit(2)
    }
    guard args.count >= 3 else { usage() }
    let folder = URL(fileURLWithPath: args[2])

    /// The `.aimodel` in an export folder, and which model it is.
    func export(_ folder: URL) throws -> (asset: URL, model: StableAudioModel) {
        let names = try FileManager.default.contentsOfDirectory(atPath: folder.path)
        guard let name = names.first(where: { $0.hasSuffix(".aimodel") }) else { fatalError("no .aimodel in \(folder.path)") }
        let model: StableAudioModel = name.contains("medium") ? .sa3Medium : name.contains("music") ? .sa3SmallMusic : .sa3SmallSFX
        return (folder.appendingPathComponent(name), model)
    }

    func load() async throws -> CoreAIEngine {
        let (asset, model) = try export(folder)
        let start = Date()
        let tokenizer = try await GemmaTokenizer(modelFolder: folder.appendingPathComponent("tokenizer"))
        let engine = try await CoreAIEngine(contentsOf: asset, model: model, tokenizer: tokenizer)
        FileHandle.standardError.write(Data(String(format: "loaded %@ in %.1fs (decoders %@)\n", asset.lastPathComponent,
                                                   -start.timeIntervalSinceNow, engine.decoderLengths.description).utf8))
        return engine
    }

    /// Noise read from a file, in the order the sampler draws it.
    struct RecordedNoise: LatentNoise {
        var values: [Float]
        var offset = 0
        mutating func next(count: Int) -> [Float] {
            defer { offset += count }
            return Array(values[offset ..< offset + count])
        }
    }

    func readFloats(_ url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    func write(_ values: [Float], _ url: URL) throws {
        try values.withUnsafeBufferPointer { Data(buffer: $0) }.write(to: url)
    }

    switch args[1] {
    case "info":
        let (asset, _) = try export(folder)
        let start = Date()
        let model = try await AIModel(contentsOf: asset, options: SpecializationOptions(preferredComputeUnitKind: .gpu))
        print(String(format: "load %.1fs", -start.timeIntervalSinceNow))
        for name in model.functionNames.sorted() {
            guard let descriptor = model.functionDescriptor(for: name) else { continue }
            func describe(_ names: [String], _ lookup: (String) -> InferenceValue.Descriptor?) -> String {
                names.map { n in
                    if case .ndArray(let d) = lookup(n) { return "\(n) \(d.scalarType) \(d.shape)" }
                    return n
                }.joined(separator: ", ")
            }
            print("\(name): (\(describe(descriptor.inputNames, descriptor.inputDescriptor(of:)))) -> (\(describe(descriptor.outputNames, descriptor.outputDescriptor(of:))))")
        }

    case "generate":
        guard args.count >= 5 else { usage() }
        let engine = try await load()
        let seconds = args.count > 5 ? Double(args[5]) ?? 10 : 10
        let seed = args.count > 6 ? Int(args[6]) : nil
        let steps = args.count > 7 ? Int(args[7]) : nil
        let cfg = args.count > 8 ? Double(args[8]) : nil
        let negative = args.count > 9 ? args[9] : nil
        let start = Date()
        let audio = try await engine.generate(prompt: args[4], options: GenerationOptions(
            seconds: seconds, steps: steps, cfgScale: cfg, seed: seed, negativePrompt: negative))
        let took = -start.timeIntervalSinceNow
        try audio.wavData.write(to: URL(fileURLWithPath: args[3]))
        let peak = audio.samples.map(abs).max() ?? 0
        print(String(format: "%.2fs audio in %.2fs, seed %ld, peak %.3f -> %@", audio.duration, took, audio.seed ?? -1, peak, args[3]))

    case "parity":
        guard args.count >= 4 else { usage() }
        let fixtures = URL(fileURLWithPath: args[3])
        let meta = try JSONSerialization.jsonObject(with: Data(contentsOf: fixtures.appendingPathComponent("meta.json"))) as! [String: Any]
        let prompt = meta["prompt"] as! String
        let frames = meta["frames"] as! Int
        let steps = meta["steps"] as! Int
        let cfg = meta["cfg_scale"] as? Double ?? 1
        let negative = meta["negative"] as? String
        let engine = try await load()
        let noise = try readFloats(fixtures.appendingPathComponent("noise.f32"))
        let seconds = PingPongSampler.seconds(ofFrames: frames)

        // Tokens.
        let (ids, mask) = engine.tokens(prompt)
        try write(ids.map(Float.init), fixtures.appendingPathComponent("swift_ids.f32"))
        try write(mask.map(Float.init), fixtures.appendingPathComponent("swift_mask.f32"))

        // Conditioner.
        var clock = Date()
        let conditioning = try await engine.condition(prompt, seconds: seconds)
        let conditionTime = -clock.timeIntervalSinceNow
        let crossAttention = try CoreAIEngine.values(conditioning.crossAttention)
        let global = try CoreAIEngine.values(conditioning.global)
        try write(crossAttention, fixtures.appendingPathComponent("swift_cross.f32"))
        try write(global, fixtures.appendingPathComponent("swift_global.f32"))

        // One transformer step from the first noise.
        let count = PingPongSampler.latentChannels * frames
        let t = PingPongSampler.schedule(steps: steps)
        clock = Date()
        let v0 = try await engine.velocity(noise[0 ..< count].map(Double.init), t: t[0], frames: frames, conditioning: conditioning)
        let stepTime = -clock.timeIntervalSinceNow
        try write(v0, fixtures.appendingPathComponent("swift_v0.f32"))

        // The decoder on Python's final latent.
        let pythonLatent = try readFloats(fixtures.appendingPathComponent("latent.f32")).map(Double.init)
        clock = Date()
        let decoded = try await engine.decode(pythonLatent, frames: frames)
        let decodeTime = -clock.timeIntervalSinceNow
        try write(decoded, fixtures.appendingPathComponent("swift_decoded.f32"))

        // The whole thing, on the same noise.
        var recorded = RecordedNoise(values: noise)
        clock = Date()
        let audio = try await engine.render(prompt: prompt, negativePrompt: negative, frames: frames, steps: steps,
                                            cfgScale: cfg, noise: &recorded)
        let renderTime = -clock.timeIntervalSinceNow
        try write(audio, fixtures.appendingPathComponent("swift_audio.f32"))
        print(String(format: "condition %.3fs, one step %.3fs, decode %.3fs, render %.2fs", conditionTime, stepTime, decodeTime, renderTime))

    default:
        usage()
    }
}

if #available(macOS 27.0, *) {
    try await main()
} else {
    fatalError("Core AI needs macOS 27")
}
