import Accelerate
import Foundation
@preconcurrency import OnnxRuntimeBindings

// MARK: - Latent Sampling

private func sampleNoisyLatent(
    duration: [Float],
    sampleRate: Int,
    baseChunkSize: Int,
    chunkCompress: Int,
    latentDim: Int
) -> (latent: [[[Float]]], mask: [[[Float]]]) {
    let batchSize = duration.count
    let maxDuration = duration.max() ?? 0
    let maxWavLength = Int(maxDuration * Float(sampleRate))
    let wavLengths = duration.map { Int($0 * Float(sampleRate)) }

    let chunkSize = baseChunkSize * chunkCompress
    let latentLength = (maxWavLength + chunkSize - 1) / chunkSize
    let latentDimTotal = latentDim * chunkCompress

    // Generate Gaussian noise
    var latent = [[[Float]]](
        repeating: [[Float]](
            repeating: [Float](repeating: 0, count: latentLength), count: latentDimTotal),
        count: batchSize)

    for b in 0..<batchSize {
        for d in 0..<latentDimTotal {
            for t in 0..<latentLength {
                let u1 = Float.random(in: 0.0001...1.0)
                let u2 = Float.random(in: 0...1)
                latent[b][d][t] = sqrt(-2 * log(u1)) * cos(2 * .pi * u2)
            }
        }
    }

    // Create and apply mask
    let latentLengths = wavLengths.map { ($0 + chunkSize - 1) / chunkSize }
    let mask = MaskUtilities.makeMask(lengths: latentLengths, maxLength: latentLength)

    for b in 0..<batchSize {
        for d in 0..<latentDimTotal {
            for t in 0..<latentLength {
                latent[b][d][t] *= mask[b][0][t]
            }
        }
    }

    return (latent, mask)
}

// MARK: - ONNX Session Manager

final class TextToSpeech: @unchecked Sendable {
    let config: ModelConfig
    let textProcessor: TextProcessor
    let sampleRate: Int

    private let durationPredictor: ORTSession
    private let textEncoder: ORTSession
    private let vectorEstimator: ORTSession
    private let vocoder: ORTSession

    init(
        config: ModelConfig,
        textProcessor: TextProcessor,
        durationPredictor: ORTSession,
        textEncoder: ORTSession,
        vectorEstimator: ORTSession,
        vocoder: ORTSession
    ) {
        self.config = config
        self.textProcessor = textProcessor
        self.sampleRate = config.autoEncoder.sampleRate
        self.durationPredictor = durationPredictor
        self.textEncoder = textEncoder
        self.vectorEstimator = vectorEstimator
        self.vocoder = vocoder
    }

    func call(
        _ text: String,
        _ style: Style,
        _ steps: Int,
        speed: Float = 1.05,
        silenceDuration: Float = 0.3
    ) throws -> (wav: [Float], duration: Float) {
        let chunks = TextChunker.chunk(text)

        var wavCat: [Float] = []
        var durCat: Float = 0

        for (i, chunk) in chunks.enumerated() {
            let result = try infer([chunk], style, steps, speed: speed)
            let dur = result.duration[0]
            let wavLen = Int(Float(sampleRate) * dur)
            let wavChunk = Array(result.wav.prefix(wavLen))

            if i == 0 {
                wavCat = wavChunk
                durCat = dur
            } else {
                let silenceLen = Int(silenceDuration * Float(sampleRate))
                wavCat.append(contentsOf: [Float](repeating: 0, count: silenceLen))
                wavCat.append(contentsOf: wavChunk)
                durCat += silenceDuration + dur
            }
        }

        return (wavCat, durCat)
    }

    func batch(
        _ texts: [String],
        _ style: Style,
        _ steps: Int,
        speed: Float = 1.05
    ) throws -> (wav: [Float], duration: [Float]) {
        try infer(texts, style, steps, speed: speed)
    }

    private func infer(
        _ texts: [String],
        _ style: Style,
        _ steps: Int,
        speed: Float
    ) throws -> (wav: [Float], duration: [Float]) {
        let batchSize = texts.count
        let (textIds, textMask) = textProcessor.process(texts)

        // Prepare text inputs
        let textIdsFlat = textIds.flatMap { $0 }
        let textIdsValue = try ORTValue(
            tensorData: NSMutableData(
                bytes: textIdsFlat, length: textIdsFlat.count * MemoryLayout<Int64>.size),
            elementType: .int64,
            shape: [NSNumber(value: batchSize), NSNumber(value: textIds[0].count)]
        )

        let textMaskFlat = textMask.flatMap { $0.flatMap { $0 } }
        let textMaskValue = try ORTValue(
            tensorData: NSMutableData(
                bytes: textMaskFlat, length: textMaskFlat.count * MemoryLayout<Float>.size),
            elementType: .float,
            shape: [NSNumber(value: batchSize), 1, NSNumber(value: textMask[0][0].count)]
        )

        // Duration prediction
        let durationOutputs = try durationPredictor.run(
            withInputs: [
                "text_ids": textIdsValue, "style_dp": style.dp, "text_mask": textMaskValue,
            ],
            outputNames: ["duration"],
            runOptions: nil
        )

        let durationData = try durationOutputs["duration"]!.tensorData() as Data
        var duration = durationData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        for i in 0..<duration.count {
            duration[i] /= speed
        }

        // Text encoding
        let textEncOutputs = try textEncoder.run(
            withInputs: [
                "text_ids": textIdsValue, "style_ttl": style.ttl, "text_mask": textMaskValue,
            ],
            outputNames: ["text_emb"],
            runOptions: nil
        )
        let textEmb = textEncOutputs["text_emb"]!

        // Sample noisy latent
        var (xt, latentMask) = sampleNoisyLatent(
            duration: duration,
            sampleRate: sampleRate,
            baseChunkSize: config.autoEncoder.baseChunkSize,
            chunkCompress: config.textToLatent.chunkCompressFactor,
            latentDim: config.textToLatent.latentDim
        )

        // Prepare step tensors
        let totalStepArray = [Float](repeating: Float(steps), count: batchSize)
        let totalStepValue = try ORTValue(
            tensorData: NSMutableData(
                bytes: totalStepArray, length: totalStepArray.count * MemoryLayout<Float>.size),
            elementType: .float,
            shape: [NSNumber(value: batchSize)]
        )

        // Denoising loop
        for step in 0..<steps {
            let currentStepArray = [Float](repeating: Float(step), count: batchSize)
            let currentStepValue = try ORTValue(
                tensorData: NSMutableData(
                    bytes: currentStepArray,
                    length: currentStepArray.count * MemoryLayout<Float>.size),
                elementType: .float,
                shape: [NSNumber(value: batchSize)]
            )

            let xtFlat = xt.flatMap { $0.flatMap { $0 } }
            let xtValue = try ORTValue(
                tensorData: NSMutableData(
                    bytes: xtFlat, length: xtFlat.count * MemoryLayout<Float>.size),
                elementType: .float,
                shape: [
                    NSNumber(value: batchSize), NSNumber(value: xt[0].count),
                    NSNumber(value: xt[0][0].count),
                ]
            )

            let latentMaskFlat = latentMask.flatMap { $0.flatMap { $0 } }
            let latentMaskValue = try ORTValue(
                tensorData: NSMutableData(
                    bytes: latentMaskFlat, length: latentMaskFlat.count * MemoryLayout<Float>.size),
                elementType: .float,
                shape: [NSNumber(value: batchSize), 1, NSNumber(value: latentMask[0][0].count)]
            )

            let outputs = try vectorEstimator.run(
                withInputs: [
                    "noisy_latent": xtValue,
                    "text_emb": textEmb,
                    "style_ttl": style.ttl,
                    "latent_mask": latentMaskValue,
                    "text_mask": textMaskValue,
                    "current_step": currentStepValue,
                    "total_step": totalStepValue,
                ],
                outputNames: ["denoised_latent"],
                runOptions: nil
            )

            let denoisedData = try outputs["denoised_latent"]!.tensorData() as Data
            let denoisedFlat = denoisedData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }

            let latentDim = xt[0].count
            let latentLen = xt[0][0].count
            xt = stride(from: 0, to: denoisedFlat.count, by: latentDim * latentLen).map { start in
                stride(from: 0, to: latentDim * latentLen, by: latentLen).map { dimStart in
                    Array(denoisedFlat[(start + dimStart)..<(start + dimStart + latentLen)])
                }
            }
        }

        // Vocoder
        let finalXtFlat = xt.flatMap { $0.flatMap { $0 } }
        let finalXtValue = try ORTValue(
            tensorData: NSMutableData(
                bytes: finalXtFlat, length: finalXtFlat.count * MemoryLayout<Float>.size),
            elementType: .float,
            shape: [
                NSNumber(value: batchSize), NSNumber(value: xt[0].count),
                NSNumber(value: xt[0][0].count),
            ]
        )

        let vocoderOutputs = try vocoder.run(
            withInputs: ["latent": finalXtValue],
            outputNames: ["wav_tts"],
            runOptions: nil
        )

        let wavData = try vocoderOutputs["wav_tts"]!.tensorData() as Data
        let wav = wavData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }

        return (wav, duration)
    }
}

// MARK: - Model Loading

func loadTextToSpeech(_ onnxDir: String, _ useGPU: Bool, _ env: ORTEnv) throws -> TextToSpeech {
    if useGPU {
        throw Supertone.SynthesisError.gpuNotSupported
    }
    print("Using CPU for inference", to: &standardError)

    let configPath = "\(onnxDir)/tts.json"
    let configData = try Data(contentsOf: URL(fileURLWithPath: configPath))
    let config = try JSONDecoder().decode(ModelConfig.self, from: configData)

    let sessionOptions = try ORTSessionOptions()

    let durationPredictor = try ORTSession(
        env: env, modelPath: "\(onnxDir)/duration_predictor.onnx", sessionOptions: sessionOptions)
    let textEncoder = try ORTSession(
        env: env, modelPath: "\(onnxDir)/text_encoder.onnx", sessionOptions: sessionOptions)
    let vectorEstimator = try ORTSession(
        env: env, modelPath: "\(onnxDir)/vector_estimator.onnx", sessionOptions: sessionOptions)
    let vocoder = try ORTSession(
        env: env, modelPath: "\(onnxDir)/vocoder.onnx", sessionOptions: sessionOptions)

    let textProcessor = try TextProcessor(indexerPath: "\(onnxDir)/unicode_indexer.json")

    return TextToSpeech(
        config: config,
        textProcessor: textProcessor,
        durationPredictor: durationPredictor,
        textEncoder: textEncoder,
        vectorEstimator: vectorEstimator,
        vocoder: vocoder
    )
}

// MARK: - Standard Error

nonisolated(unsafe) private var standardError = FileHandle.standardError

extension FileHandle: @retroactive TextOutputStream {
    public func write(_ string: String) {
        let data = Data(string.utf8)
        self.write(data)
    }
}
