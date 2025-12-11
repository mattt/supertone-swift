import Foundation

/// Writes audio samples to a WAV file.
/// - Parameters:
///   - filename: Path to the output file.
///   - samples: Audio samples (mono, float32, normalized to [-1, 1]).
///   - sampleRate: Sample rate in Hz.
public func writeWavFile(_ filename: String, _ samples: [Float], _ sampleRate: Int) throws {
    let url = URL(fileURLWithPath: filename)

    // Convert to 16-bit PCM
    let pcmData = samples.map { sample -> Int16 in
        Int16(max(-1, min(1, sample)) * 32767)
    }

    let numChannels: UInt16 = 1
    let bitsPerSample: UInt16 = 16
    let byteRate = UInt32(sampleRate) * UInt32(numChannels) * UInt32(bitsPerSample) / 8
    let blockAlign = numChannels * bitsPerSample / 8
    let dataSize = UInt32(pcmData.count * 2)

    var data = Data()
    data.reserveCapacity(44 + Int(dataSize))

    // RIFF header
    data.append("RIFF".data(using: .ascii)!)
    withUnsafeBytes(of: UInt32(36 + dataSize).littleEndian) { data.append(contentsOf: $0) }
    data.append("WAVE".data(using: .ascii)!)

    // fmt chunk
    data.append("fmt ".data(using: .ascii)!)
    withUnsafeBytes(of: UInt32(16).littleEndian) { data.append(contentsOf: $0) }
    withUnsafeBytes(of: UInt16(1).littleEndian) { data.append(contentsOf: $0) }  // PCM
    withUnsafeBytes(of: numChannels.littleEndian) { data.append(contentsOf: $0) }
    withUnsafeBytes(of: UInt32(sampleRate).littleEndian) { data.append(contentsOf: $0) }
    withUnsafeBytes(of: byteRate.littleEndian) { data.append(contentsOf: $0) }
    withUnsafeBytes(of: blockAlign.littleEndian) { data.append(contentsOf: $0) }
    withUnsafeBytes(of: bitsPerSample.littleEndian) { data.append(contentsOf: $0) }

    // data chunk
    data.append("data".data(using: .ascii)!)
    withUnsafeBytes(of: dataSize.littleEndian) { data.append(contentsOf: $0) }
    pcmData.withUnsafeBytes { data.append(contentsOf: $0) }

    try data.write(to: url)
}
