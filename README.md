# Supertone

A Swift wrapper for the [Supertone](https://supertone.ai) ONNX text-to-speech pipeline.

> [!WARNING] 
> This project is in active development. 
> Features and APIs may change.

Generate high-quality speech synthesis from text using voice style embeddings.
Includes a `supersay` CLI tool similar to the macOS `say` command.

## Requirements

- Swift 6.0+ / Xcode 16+
- macOS 13+

## Installation

### Swift Package Manager

Add the following to your `Package.swift` file:

```swift
dependencies: [
    .package(url: "https://github.com/example/supertone-swift.git", from: "1.0.0")
]
```

Then add the dependency to your target:

```swift
.target(
    name: "YourTarget",
    dependencies: [
        .product(name: "Supertone", package: "supertone-swift")
    ]
)
```

## Usage

### Basic Speech Synthesis

```swift
import Supertone

// Create a synthesizer with ONNX model directory
let synthesizer = try await Supertone(onnxDirectory: "path/to/onnx")

// Synthesize speech from text
let audio = try await synthesizer.synthesize(
    "Hello, world!",
    voiceStylePaths: ["path/to/voice_styles/M1.json"]
)

// Write to WAV file
try writeWavFile("output.wav", audio.samples, audio.sampleRate)
```

### Synthesis Options

```swift
// Configure synthesis parameters
let options = Supertone.Options(
    steps: 5,              // Denoising steps (higher = better quality)
    speed: 1.05,           // Speed multiplier (> 1.0 = faster)
    silenceDuration: 0.3,  // Silence between chunks (seconds)
    verbose: true          // Print progress
)

let audio = try await synthesizer.synthesize(
    "Hello, world!",
    voiceStylePaths: ["path/to/voice_styles/M1.json"],
    options: options
)
```

### Batch Synthesis

```swift
// Synthesize multiple texts with different voices
let texts = ["Hello!", "Goodbye!"]
let voices = ["voice_styles/M1.json", "voice_styles/F1.json"]

let audioOutputs = try await synthesizer.synthesizeBatch(
    texts,
    voiceStylePaths: voices,
    options: .init(verbose: true)
)

for (i, audio) in audioOutputs.enumerated() {
    try writeWavFile("output_\(i).wav", audio.samples, audio.sampleRate)
}
```

### Audio Output

The `Supertone.Audio` struct provides:

```swift
audio.samples    // [Float] - Raw audio samples (mono, normalized to [-1, 1])
audio.sampleRate // Int - Sample rate in Hz (typically 24000)
audio.duration   // Float - Duration in seconds
```

## CLI

The package includes `supersay`, a command-line tool for speech synthesis.

First, download and locate the model from Hugging Face:

```bash
hf download Supertone/supertonic

find ~/.cache/huggingface/hub/models--Supertone--supertonic/**/onnx -type d

find ~/.cache/huggingface/hub/models--Supertone--supertonic/**/voice_styles/*.json
```

```bash
# Basic usage
swift run supersay "Hello from Supertone"

# With custom model and voice
swift run supersay -m path/to/onnx --voice M1.json "Hello, world!"

# Read from stdin
echo "Hello from stdin" | swift run supersay

# Save to file instead of playing
swift run supersay -o hello.wav "Hello, world!"

# Batch mode (multiple texts with multiple voices)
swift run supersay --batch --voice M1.json --voice F1.json "Hello" "Goodbye"

# Adjust synthesis parameters
swift run supersay --steps 10 -s 0.9 "Slower, higher quality"

# Verbose output
swift run supersay --verbose "Hello"
```

### CLI Options

| Short | Long | Description |
|-------|------|-------------|
| `-m` | `--model` | Path to ONNX model directory |
| | `--voice` | Voice style JSON file(s) |
| `-o` | `--output` | Output WAV file path |
| `-s` | `--speed` | Speech rate multiplier (default: 1.05) |
| | `--steps` | Denoising steps, 1-20 (default: 5) |
| | `--silence` | Silence between chunks in seconds (default: 0.3) |
| `-v` | `--verbose` | Show detailed progress |
| `-q` | `--quiet` | Suppress non-essential output |
| | `--no-play` | Skip audio playback |
| | `--batch` | Batch mode (texts must match voice count) |
| | `--version` | Show version |
| `-h` | `--help` | Show help |

## Contributing

Contributions are welcome! Please ensure your code passes the build and test suite before submitting a pull request.

```bash
swift build
swift test
```

## License

This project is available under the MIT license.
See the LICENSE file for more info.
