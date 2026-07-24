//
//  main.swift — sa-tokenize
//
//  Verify the Gemma tokenizer matches the reference ids.
//  Usage: sa-tokenize <tokenizerFolder> <text>
//

import Foundation
import StableAudioKitGemma

let args = CommandLine.arguments
guard args.count == 3 else { fatalError("usage: sa-tokenize <tokenizerFolder> <text>") }

let tok = try await GemmaTokenizer(modelFolder: URL(fileURLWithPath: args[1]))
let ids = tok.encode(args[2])
print("text: \"\(args[2])\"")
print("ids (\(ids.count)): \(ids)")
