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
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", .upToNextMajor(from: "0.31.6")),
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
    ]
)
