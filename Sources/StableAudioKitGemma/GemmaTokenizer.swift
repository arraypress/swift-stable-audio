//
//  GemmaTokenizer.swift
//  StableAudioKitGemma
//
//  A `SATokenizer` backed by swift-transformers' Gemma tokenizer, so an app can
//  feed arbitrary text prompts to `MLXEngine` or `CoreAIEngine`. Load once (async) from the
//  model's `t5gemma-b-b-ul2` tokenizer files, then encode synchronously.
//

import Foundation
import Tokenizers
import StableAudioKit

/// Gemma tokenizer wrapper. `@unchecked Sendable`: the underlying tokenizer is
/// used read-only after construction.
public final class GemmaTokenizer: SATokenizer, @unchecked Sendable {

    private let tokenizer: any Tokenizer

    /// Load from a folder containing `tokenizer.json` + `tokenizer_config.json`.
    public init(modelFolder: URL) async throws {
        self.tokenizer = try await AutoTokenizer.from(modelFolder: modelFolder)
    }

    /// Load a pre-built tokenizer.
    public init(_ tokenizer: any Tokenizer) {
        self.tokenizer = tokenizer
    }

    public func encode(_ text: String) -> [Int] {
        tokenizer.encode(text: text)
    }
}
