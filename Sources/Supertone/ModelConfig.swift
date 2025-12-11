import Foundation
@preconcurrency import OnnxRuntimeBindings

// MARK: - Model Configuration

struct ModelConfig: Codable {
    struct AutoEncoder: Codable {
        let sampleRate: Int
        let baseChunkSize: Int

        private enum CodingKeys: String, CodingKey {
            case sampleRate = "sample_rate"
            case baseChunkSize = "base_chunk_size"
        }
    }

    struct TextToLatent: Codable {
        let chunkCompressFactor: Int
        let latentDim: Int

        private enum CodingKeys: String, CodingKey {
            case chunkCompressFactor = "chunk_compress_factor"
            case latentDim = "latent_dim"
        }
    }

    let autoEncoder: AutoEncoder
    let textToLatent: TextToLatent

    private enum CodingKeys: String, CodingKey {
        case autoEncoder = "ae"
        case textToLatent = "ttl"
    }
}

// MARK: - Voice Style

struct VoiceStyleData: Codable {
    struct Component: Codable {
        let data: [[[Float]]]
        let dims: [Int]
        let type: String
    }

    let ttl: Component
    let dp: Component

    private enum CodingKeys: String, CodingKey {
        case ttl = "style_ttl"
        case dp = "style_dp"
    }
}

struct Style: @unchecked Sendable {
    let ttl: ORTValue
    let dp: ORTValue
}

// MARK: - Style Loading

func loadVoiceStyle(_ paths: [String], verbose: Bool) throws -> Style {
    let batchSize = paths.count

    let firstData = try Data(contentsOf: URL(fileURLWithPath: paths[0]))
    let firstStyle = try JSONDecoder().decode(VoiceStyleData.self, from: firstData)

    let ttlDims = firstStyle.ttl.dims
    let dpDims = firstStyle.dp.dims
    let ttlSize = batchSize * ttlDims[1] * ttlDims[2]
    let dpSize = batchSize * dpDims[1] * dpDims[2]

    var ttlFlat = [Float](repeating: 0, count: ttlSize)
    var dpFlat = [Float](repeating: 0, count: dpSize)

    for (i, path) in paths.enumerated() {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let style = try JSONDecoder().decode(VoiceStyleData.self, from: data)

        let ttlOffset = i * ttlDims[1] * ttlDims[2]
        var idx = 0
        for batch in style.ttl.data {
            for row in batch {
                for val in row {
                    ttlFlat[ttlOffset + idx] = val
                    idx += 1
                }
            }
        }

        let dpOffset = i * dpDims[1] * dpDims[2]
        idx = 0
        for batch in style.dp.data {
            for row in batch {
                for val in row {
                    dpFlat[dpOffset + idx] = val
                    idx += 1
                }
            }
        }
    }

    let ttlValue = try ORTValue(
        tensorData: NSMutableData(
            bytes: &ttlFlat, length: ttlFlat.count * MemoryLayout<Float>.size),
        elementType: .float,
        shape: [
            NSNumber(value: batchSize), NSNumber(value: ttlDims[1]), NSNumber(value: ttlDims[2]),
        ]
    )

    let dpValue = try ORTValue(
        tensorData: NSMutableData(bytes: &dpFlat, length: dpFlat.count * MemoryLayout<Float>.size),
        elementType: .float,
        shape: [NSNumber(value: batchSize), NSNumber(value: dpDims[1]), NSNumber(value: dpDims[2])]
    )

    if verbose {
        print("Loaded \(batchSize) voice style(s)", to: &standardError)
    }

    return Style(ttl: ttlValue, dp: dpValue)
}

// MARK: - Standard Error

nonisolated(unsafe) private var standardError = FileHandle.standardError
