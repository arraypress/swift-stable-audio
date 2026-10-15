//
//  SATokenizer.swift
//  StableAudioKit
//
//  The prompt's token ids, for whichever engine runs the T5Gemma
//  conditioner. Here rather than beside one engine since 1.2.0, so the
//  Core AI engine and the Gemma tokenizer need no MLX.
//

import Foundation

/// Supplies Gemma token ids for a prompt (fulfil with swift-transformers in the app).
public protocol SATokenizer: Sendable {
    /// Encode to token ids (BOS-prefixed, as the T5Gemma conditioner expects).
    func encode(_ text: String) -> [Int]
}
