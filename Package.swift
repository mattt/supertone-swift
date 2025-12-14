// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let swiftSettings: [SwiftSetting] = [
    .enableExperimentalFeature("StrictConcurrency")
]

let package = Package(
    name: "Supertone",
    platforms: [
        .macOS(.v13),
        .iOS(.v16)
    ],
    products: [
        .library(
            name: "Supertone",
            targets: ["Supertone"]
        ),
        .executable(
            name: "supersay",
            targets: ["CLI"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.4.0"),
        .package(
            url: "https://github.com/microsoft/onnxruntime-swift-package-manager.git",
            from: "1.16.0"),
    ],
    targets: [
        .target(
            name: "Supertone",
            dependencies: [
                .product(name: "onnxruntime", package: "onnxruntime-swift-package-manager")
            ],
            swiftSettings: swiftSettings
        ),
        .executableTarget(
            name: "CLI",
            dependencies: [
                "Supertone",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources/CLI",
            swiftSettings: swiftSettings
        ),
        .testTarget(
            name: "SupertoneTests",
            dependencies: ["Supertone"],
            swiftSettings: swiftSettings
        ),
    ]
)
