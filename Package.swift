// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "StableAudioKit",
    platforms: [
        .macOS(.v26),
        .iOS(.v26)
    ],
    products: [
        // Core: model registry + shared types + engine protocol + the
        // zero-dependency cloud engine (any HTTP text-to-audio endpoint).
        .library(name: "StableAudioKit", targets: ["StableAudioKit"]),
        // On-device Stable Audio via MLX (Mac + iPhone GPU) — opt-in.
        .library(name: "StableAudioKitMLX", targets: ["StableAudioKitMLX"]),
        // Gemma tokenizer (swift-transformers) — opt-in.
        .library(name: "StableAudioKitGemma", targets: ["StableAudioKitGemma"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", .upToNextMajor(from: "0.31.6")),
        .package(url: "https://github.com/huggingface/swift-transformers.git", .upToNextMajor(from: "1.1.6")),
    ],
    targets: [
        .target(
            name: "StableAudioKit",
            dependencies: []
        ),
        .target(
            name: "StableAudioKitMLX",
            dependencies: [
                "StableAudioKit",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ]
        ),
        .testTarget(
            name: "StableAudioKitTests",
            dependencies: ["StableAudioKit"]
        ),
        // Offline numeric validator for the MLX Swift nets (run with the
        // metallib colocated). Not part of the shipped products.
        .executableTarget(
            name: "sa-validate",
            dependencies: [
                "StableAudioKitMLX",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        // End-to-end text -> audio driver (fixed-ids tokenizer stub).
        .executableTarget(
            name: "sa-generate",
            dependencies: [
                "StableAudioKit",
                "StableAudioKitMLX",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        // Gemma tokenizer (swift-transformers) conforming to SATokenizer — opt-in.
        .target(
            name: "StableAudioKitGemma",
            dependencies: [
                "StableAudioKitMLX",
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        .executableTarget(
            name: "sa-tokenize",
            dependencies: ["StableAudioKitGemma"]
        ),
        // Real text -> audio: Gemma tokenizer + MLXEngine, any typed prompt.
        // (Named sa-audio, not sa-tts — text-to-AUDIO, not speech.)
        .executableTarget(
            name: "sa-audio",
            dependencies: [
                "StableAudioKit", "StableAudioKitMLX", "StableAudioKitGemma",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        // Batch loopable-SFX generator (model picked from weights filename).
        .executableTarget(
            name: "sa-sfx",
            dependencies: [
                "StableAudioKit", "StableAudioKitMLX", "StableAudioKitGemma",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
    ]
)
