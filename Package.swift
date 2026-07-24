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
    ],
    targets: [
        .target(
            name: "StableAudioKit",
            dependencies: []
        ),
        .testTarget(
            name: "StableAudioKitTests",
            dependencies: ["StableAudioKit"]
        ),
    ]
)
