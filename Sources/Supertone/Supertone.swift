import Foundation
@preconcurrency import OnnxRuntimeBindings

/// Text-to-speech synthesis using the Supertone ONNX pipeline.
public actor Supertone {
    private let textToSpeech: TextToSpeech

    /// Creates a new Supertone synthesizer.
    /// - Parameters:
    ///   - onnxDirectory: Path to the directory containing ONNX model files.
    ///   - useGPU: Whether to use GPU acceleration (not yet supported).
    public init(onnxDirectory: String = "assets/onnx", useGPU: Bool = false) async throws {
        let env = try ORTEnv(loggingLevel: .warning)
        self.textToSpeech = try await Task.detached {
            try loadTextToSpeech(onnxDirectory, useGPU, env)
        }.value
    }

    /// Synthesizes speech from text using the provided voice styles.
    /// - Parameters:
    ///   - text: The text to synthesize.
    ///   - voiceStylePaths: Paths to voice style JSON files.
    ///   - options: Synthesis options.
    /// - Returns: The synthesized audio.
    public func synthesize(
        _ text: String,
        voiceStylePaths: [String],
        options: Options = .init()
    ) async throws -> Audio {
        let style = try loadVoiceStyle(voiceStylePaths, verbose: options.verbose)
        let result = try textToSpeech.call(
            text, style, options.steps,
            speed: options.speed,
            silenceDuration: options.silenceDuration
        )
        return Audio(
            samples: result.wav,
            sampleRate: textToSpeech.sampleRate,
            duration: result.duration
        )
    }

    /// Synthesizes multiple texts in batch.
    /// - Parameters:
    ///   - texts: The texts to synthesize. Count must match `voiceStylePaths`.
    ///   - voiceStylePaths: Paths to voice style JSON files.
    ///   - options: Synthesis options.
    /// - Returns: Array of synthesized audio, one per input text.
    public func synthesizeBatch(
        _ texts: [String],
        voiceStylePaths: [String],
        options: Options = .init()
    ) async throws -> [Audio] {
        guard !texts.isEmpty else { return [] }
        guard texts.count == voiceStylePaths.count else {
            throw SynthesisError.mismatchedInputCounts(
                texts: texts.count,
                styles: voiceStylePaths.count
            )
        }

        let style = try loadVoiceStyle(voiceStylePaths, verbose: options.verbose)
        let result = try textToSpeech.batch(texts, style, options.steps, speed: options.speed)

        let batchSize = texts.count
        let wavLength = result.wav.count / batchSize

        return (0..<batchSize).map { i in
            let actualLength = Int(Float(textToSpeech.sampleRate) * result.duration[i])
            let start = i * wavLength
            let end = min(start + actualLength, start + wavLength)
            return Audio(
                samples: Array(result.wav[start..<end]),
                sampleRate: textToSpeech.sampleRate,
                duration: result.duration[i]
            )
        }
    }
}

// MARK: - Nested Types

extension Supertone {
    /// Options for speech synthesis.
    public struct Options: Sendable {
        /// Number of denoising steps (higher = better quality, slower).
        public var steps: Int

        /// Speed multiplier (> 1.0 = faster speech).
        public var speed: Float

        /// Duration of silence between text chunks (seconds).
        public var silenceDuration: Float

        /// Whether to print verbose output.
        public var verbose: Bool

        public init(
            steps: Int = 5,
            speed: Float = 1.05,
            silenceDuration: Float = 0.3,
            verbose: Bool = false
        ) {
            self.steps = steps
            self.speed = speed
            self.silenceDuration = silenceDuration
            self.verbose = verbose
        }
    }

    /// Synthesized audio output.
    public struct Audio: Sendable {
        /// Raw audio samples (mono, float32, normalized to [-1, 1]).
        public let samples: [Float]

        /// Sample rate in Hz.
        public let sampleRate: Int

        /// Duration in seconds.
        public let duration: Float
    }

    /// Errors that can occur during synthesis.
    public enum SynthesisError: Error, LocalizedError {
        case mismatchedInputCounts(texts: Int, styles: Int)
        case gpuNotSupported

        public var errorDescription: String? {
            switch self {
            case .mismatchedInputCounts(let texts, let styles):
                return "Number of texts (\(texts)) must match number of voice styles (\(styles))"
            case .gpuNotSupported:
                return "GPU acceleration is not yet supported"
            }
        }
    }
}
