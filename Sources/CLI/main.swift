import AVFoundation
import ArgumentParser
import Foundation
import Supertone

@main
struct Supersay: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "supersay",
        abstract: "Generate speech from text using the Supertone TTS pipeline.",
        discussion: """
            Synthesizes speech from text and plays it through the default audio output.
            Text can be provided as arguments or piped via stdin.

            Examples:
              supersay "Hello, world!"
              echo "Hello from stdin" | supersay
              supersay -o greeting.wav "Hello, world!"
            """,
        version: "1.0.0"
    )

    // MARK: - Arguments

    @Argument(help: "Text to synthesize. Reads from stdin if omitted and not a TTY.")
    var text: [String] = []

    // MARK: - Options

    @Option(name: [.long, .customShort("m")], help: "Path to ONNX model directory.")
    var model: String =
        "assets/onnx"

    @Option(name: .long, parsing: .upToNextOption, help: "Voice style JSON file(s).")
    var voice: [String] = [
        "assets/voice_styles/M1.json"
    ]

    @Option(name: .shortAndLong, help: "Output WAV file path.")
    var output: String?

    @Option(name: .long, help: "Number of denoising steps (1-20). Higher = better quality.")
    var steps: Int = 5

    @Option(name: .shortAndLong, help: "Speech rate multiplier. >1.0 = faster, <1.0 = slower.")
    var speed: Float = 1.05

    @Option(name: .long, help: "Silence between chunks in seconds.")
    var silence: Float = 0.3

    // MARK: - Flags

    @Flag(name: .shortAndLong, help: "Show detailed progress information.")
    var verbose: Bool = false

    @Flag(name: .shortAndLong, help: "Suppress all non-essential output.")
    var quiet: Bool = false

    @Flag(name: .long, help: "Skip audio playback after synthesis.")
    var noPlay: Bool = false

    @Flag(name: .long, help: "Batch mode: synthesize multiple texts with matching voice styles.")
    var batch: Bool = false

    // MARK: - Validation

    func validate() throws {
        guard steps >= 1 && steps <= 20 else {
            throw ValidationError("Steps must be between 1 and 20.")
        }
        guard speed > 0 else {
            throw ValidationError("Speed must be greater than 0.")
        }
        guard silence >= 0 else {
            throw ValidationError("Silence duration cannot be negative.")
        }
        if batch && text.count != voice.count {
            throw ValidationError(
                "Batch mode requires equal number of texts (\(text.count)) and voices (\(voice.count))."
            )
        }
    }

    // MARK: - Execution

    func run() async throws {
        let resolvedText = try resolveTextInput()

        log("Loading models from \(model)...")
        let synthesizer = try await Supertone(onnxDirectory: model)

        let options = Supertone.Options(
            steps: steps,
            speed: speed,
            silenceDuration: silence,
            verbose: verbose
        )

        if batch {
            log("Synthesizing \(resolvedText.count) texts in batch mode...")
            let outputs = try await synthesizer.synthesizeBatch(
                resolvedText, voiceStylePaths: voice, options: options
            )
            try await handleOutputs(outputs)
        } else {
            let combined = resolvedText.joined(separator: " ")
            log("Synthesizing \(combined.count) characters...")
            let audio = try await synthesizer.synthesize(
                combined, voiceStylePaths: voice, options: options
            )
            try await handleOutputs([audio])
        }
    }

    // MARK: - Input

    private func resolveTextInput() throws -> [String] {
        if !text.isEmpty {
            return text
        }

        // Check if stdin is a TTY (interactive terminal)
        guard !isatty(STDIN_FILENO).boolValue else {
            throw ValidationError(
                """
                No text provided. Usage:
                  supersay "Hello, world!"
                  echo "Hello" | supersay
                  supersay --help
                """
            )
        }

        let data = FileHandle.standardInput.readDataToEndOfFile()
        guard
            let string = String(data: data, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines),
            !string.isEmpty
        else {
            throw ValidationError("No text received from stdin.")
        }

        return [string]
    }

    // MARK: - Output

    private func handleOutputs(_ outputs: [Supertone.Audio]) async throws {
        for (index, audio) in outputs.enumerated() {
            let url = try outputURL(index: index, total: outputs.count)
            try writeWavFile(url.path, audio.samples, audio.sampleRate)

            if !quiet {
                let duration = String(format: "%.1fs", audio.duration)
                print("Wrote \(url.path) (\(duration))", to: &standardError)
            }

            if !noPlay {
                try await play(url: url)
            }
        }
    }

    private func outputURL(index: Int, total: Int) throws -> URL {
        if let path = output {
            var url = URL(fileURLWithPath: path)
            if total > 1 {
                let base = url.deletingPathExtension().path
                let ext = url.pathExtension.isEmpty ? "wav" : url.pathExtension
                url = URL(fileURLWithPath: "\(base)_\(index + 1).\(ext)")
            }
            return url
        }

        let name =
            voice.first.map {
                URL(fileURLWithPath: $0).deletingPathExtension().lastPathComponent
            } ?? "output"

        let suffix = total > 1 ? "_\(index + 1)" : ""
        return URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("\(name)\(suffix).wav")
    }

    // MARK: - Playback

    private func play(url: URL) async throws {
        let player = try AVAudioPlayer(contentsOf: url)
        player.prepareToPlay()
        player.play()
        while player.isPlaying {
            try await Task.sleep(nanoseconds: 100_000_000)
        }
    }

    // MARK: - Logging

    private func log(_ message: String) {
        guard verbose else { return }
        print(message, to: &standardError)
    }
}

// MARK: - Helpers

nonisolated(unsafe) private var standardError = FileHandle.standardError

extension Int32 {
    fileprivate var boolValue: Bool { self != 0 }
}
