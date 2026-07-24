//
//  RemoteEngine.swift
//  StableAudioKit
//
//  The cloud tier: any HTTP text-to-audio endpoint that takes a JSON prompt
//  and returns an audio file. Zero dependencies — pure URLSession. Request
//  shaping is factored into pure functions so it's unit-testable offline.
//
//  Note: this core engine decodes WAV responses into `GeneratedAudio`. Some
//  providers (e.g. ElevenLabs) return MP3 by default — use `generateFile()`
//  to get the raw bytes and decode with AVFoundation in the app layer.
//

import Foundation

/// Describes a remote text-to-audio endpoint.
public struct RemoteEndpoint: Sendable {

    /// Full request URL.
    public var url: URL

    /// API key value.
    public var apiKey: String

    /// Header the key goes in (e.g. `xi-api-key`, `Authorization`).
    public var apiKeyHeader: String

    /// JSON field name for the text prompt (e.g. `text`, `prompt`).
    public var promptField: String

    /// JSON field name for the duration in seconds, if the API accepts one.
    public var durationField: String?

    public init(
        url: URL,
        apiKey: String,
        apiKeyHeader: String,
        promptField: String,
        durationField: String? = nil
    ) {
        self.url = url
        self.apiKey = apiKey
        self.apiKeyHeader = apiKeyHeader
        self.promptField = promptField
        self.durationField = durationField
    }

    /// ElevenLabs Sound Effects (`/v1/sound-generation`). Returns MP3 —
    /// use `RemoteEngine.generateFile` and decode in-app.
    public static func elevenLabsSoundEffects(apiKey: String) -> RemoteEndpoint {
        RemoteEndpoint(
            url: URL(string: "https://api.elevenlabs.io/v1/sound-generation")!,
            apiKey: apiKey, apiKeyHeader: "xi-api-key",
            promptField: "text", durationField: "duration_seconds"
        )
    }
}

/// A cloud ``AudioGenEngine`` for any OpenAI/ElevenLabs-style audio endpoint.
public struct RemoteEngine: AudioGenEngine {

    public let endpoint: RemoteEndpoint
    private let session: URLSession

    public init(_ endpoint: RemoteEndpoint, session: URLSession = .shared) {
        self.endpoint = endpoint
        self.session = session
    }

    public var isReady: Bool { true }

    // MARK: - Pure request shaping (testable offline)

    /// The JSON body for a request.
    public func requestBody(prompt: String, options: GenerationOptions) -> [String: Any] {
        var body: [String: Any] = [endpoint.promptField: prompt]
        if let field = endpoint.durationField { body[field] = options.seconds }
        return body
    }

    /// The fully-formed `URLRequest` (method, headers, JSON body).
    public func makeRequest(prompt: String, options: GenerationOptions) throws -> URLRequest {
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw AudioGenError.invalidInput("empty prompt")
        }
        var req = URLRequest(url: endpoint.url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.setValue(endpoint.apiKey, forHTTPHeaderField: endpoint.apiKeyHeader)
        req.httpBody = try JSONSerialization.data(
            withJSONObject: requestBody(prompt: prompt, options: options)
        )
        return req
    }

    // MARK: - Network

    /// Fetch the raw audio file bytes (WAV, MP3, …) as returned by the provider.
    public func generateFile(prompt: String, options: GenerationOptions) async throws -> Data {
        let req = try makeRequest(prompt: prompt, options: options)
        let (data, resp): (Data, URLResponse)
        do {
            (data, resp) = try await session.data(for: req)
        } catch {
            throw AudioGenError.requestFailed(String(describing: error))
        }
        guard let http = resp as? HTTPURLResponse else {
            throw AudioGenError.requestFailed("no HTTP response")
        }
        guard (200..<300).contains(http.statusCode) else {
            throw AudioGenError.requestFailed("HTTP \(http.statusCode)")
        }
        return data
    }

    /// Generate and decode a WAV response into `GeneratedAudio`.
    /// (For MP3-returning providers, call `generateFile` and decode in-app.)
    public func generate(prompt: String, options: GenerationOptions) async throws -> GeneratedAudio {
        let data = try await generateFile(prompt: prompt, options: options)
        return try GeneratedAudio(wav: data)
    }
}
