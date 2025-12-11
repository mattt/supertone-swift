import Foundation

// MARK: - Text Processor

final class TextProcessor: Sendable {
    private let indexer: [Int64]

    init(indexerPath: String) throws {
        let data = try Data(contentsOf: URL(fileURLWithPath: indexerPath))
        self.indexer = try JSONDecoder().decode([Int64].self, from: data)
    }

    func process(_ texts: [String]) -> (ids: [[Int64]], mask: [[[Float]]]) {
        let processed = texts.map { TextNormalizer.normalize($0) }
        let lengths = processed.map(\.count)
        let maxLength = lengths.max() ?? 0

        let ids: [[Int64]] = processed.map { text in
            var row = [Int64](repeating: 0, count: maxLength)
            for (i, scalar) in text.unicodeScalars.enumerated() {
                let value = Int(scalar.value)
                row[i] = value < indexer.count ? indexer[value] : -1
            }
            return row
        }

        let mask = MaskUtilities.makeMask(lengths: lengths, maxLength: maxLength)
        return (ids, mask)
    }
}

// MARK: - Text Normalization

enum TextNormalizer {
    // Emoji Unicode ranges
    private static let emojiRanges: [ClosedRange<UInt32>] = [
        0x1F600...0x1F64F,  // Emoticons
        0x1F300...0x1F5FF,  // Misc Symbols and Pictographs
        0x1F680...0x1F6FF,  // Transport and Map
        0x1F700...0x1F77F,  // Alchemical Symbols
        0x1F780...0x1F7FF,  // Geometric Shapes Extended
        0x1F800...0x1F8FF,  // Supplemental Arrows-C
        0x1F900...0x1F9FF,  // Supplemental Symbols and Pictographs
        0x1FA00...0x1FA6F,  // Chess Symbols
        0x1FA70...0x1FAFF,  // Symbols and Pictographs Extended-A
        0x2600...0x26FF,  // Misc symbols
        0x2700...0x27BF,  // Dingbats
        0x1F1E6...0x1F1FF,  // Flags
    ]

    // Swift 6 Regex patterns (nonisolated unsafe since Regex is immutable in practice)
    nonisolated(unsafe) private static let diacriticsRegex =
        /[\u{0302}\u{0303}\u{0304}\u{0305}\u{0306}\u{0307}\u{0308}\u{030A}\u{030B}\u{030C}\u{0327}\u{0328}\u{0329}\u{032A}\u{032B}\u{032C}\u{032D}\u{032E}\u{032F}]/
    nonisolated(unsafe) private static let whitespaceRegex = /\s+/
    nonisolated(unsafe) private static let terminalPunctuationRegex =
        /[.!?;:,'"\u{201C}\u{201D}\u{2018}\u{2019})\]}…。」』】〉》›»]$/

    // Character replacement maps
    private static let replacements: [Character: Character] = [
        "–": "-", "‑": "-", "—": "-",  // dashes
        "¯": " ", "_": " ",  // macron, underscore
        "\u{201C}": "\"", "\u{201D}": "\"",  // curly double quotes
        "\u{2018}": "'", "\u{2019}": "'",  // curly single quotes
        "´": "'", "`": "'",  // accents
        "[": " ", "]": " ",  // brackets
        "|": " ", "/": " ", "#": " ",  // symbols
        "→": " ", "←": " ",  // arrows
    ]

    private static let removals: Set<Character> = ["♥", "☆", "♡", "©", "\\"]

    private static let expressionReplacements: [(String, String)] = [
        ("@", " at "),
        ("e.g.,", "for example, "),
        ("i.e.,", "that is, "),
    ]

    static func normalize(_ text: String) -> String {
        var result = text.precomposedStringWithCompatibilityMapping

        // Remove emojis (single pass)
        result = String(result.unicodeScalars.filter { !isEmoji($0) })

        // Apply character replacements (single pass)
        result = String(
            result.compactMap { char in
                if removals.contains(char) { return nil }
                return replacements[char] ?? char
            }
        )

        // Remove combining diacritics
        result = result.replacing(diacriticsRegex, with: "")

        // Expression replacements
        for (old, new) in expressionReplacements {
            result = result.replacingOccurrences(of: old, with: new)
        }

        // Fix spacing around punctuation
        result = fixPunctuationSpacing(result)

        // Collapse duplicate quotes
        result = collapseDuplicateQuotes(result)

        // Normalize whitespace
        result = result.replacing(whitespaceRegex, with: " ")
        result = result.trimmingCharacters(in: .whitespacesAndNewlines)

        // Ensure terminal punctuation
        if !result.isEmpty && result.firstMatch(of: terminalPunctuationRegex) == nil {
            result += "."
        }

        return result
    }

    private static func isEmoji(_ scalar: Unicode.Scalar) -> Bool {
        let value = scalar.value
        return emojiRanges.contains { $0.contains(value) }
    }

    private static func fixPunctuationSpacing(_ text: String) -> String {
        var result = text
        for punc in [",", ".", "!", "?", ";", ":", "'"] {
            result = result.replacingOccurrences(of: " \(punc)", with: punc)
        }
        return result
    }

    private static func collapseDuplicateQuotes(_ text: String) -> String {
        var result = text
        let doubleQuoteRegex = /"{2,}/
        let singleQuoteRegex = /'{2,}/
        let backtickRegex = /`{2,}/

        result = result.replacing(doubleQuoteRegex, with: "\"")
        result = result.replacing(singleQuoteRegex, with: "'")
        result = result.replacing(backtickRegex, with: "`")
        return result
    }
}

// MARK: - Text Chunking

enum TextChunker {
    private static let maxChunkLength = 300

    private static let abbreviations: Set<String> = [
        "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "Sr.", "Jr.",
        "St.", "Ave.", "Rd.", "Blvd.", "Dept.", "Inc.", "Ltd.",
        "Co.", "Corp.", "etc.", "vs.", "i.e.", "e.g.", "Ph.D.",
    ]

    nonisolated(unsafe) private static let paragraphRegex = /\n\s*\n/
    nonisolated(unsafe) private static let sentenceRegex = /([.!?])\s+/

    static func chunk(_ text: String, maxLength: Int = 0) -> [String] {
        let limit = maxLength > 0 ? maxLength : maxChunkLength
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)

        guard !trimmed.isEmpty else { return [""] }

        let paragraphs = splitByRegex(trimmed, regex: paragraphRegex)
        var chunks: [String] = []

        for paragraph in paragraphs {
            let para = paragraph.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !para.isEmpty else { continue }

            if para.count <= limit {
                chunks.append(para)
                continue
            }

            chunks.append(contentsOf: chunkParagraph(para, limit: limit))
        }

        return chunks.isEmpty ? [""] : chunks
    }

    private static func splitByRegex(_ text: String, regex: some RegexComponent) -> [String] {
        var results: [String] = []
        var lastEnd = text.startIndex

        for match in text.matches(of: regex) {
            results.append(String(text[lastEnd..<match.range.lowerBound]))
            lastEnd = match.range.upperBound
        }

        if lastEnd < text.endIndex {
            results.append(String(text[lastEnd...]))
        }

        return results.isEmpty ? [text] : results
    }

    private static func chunkParagraph(_ paragraph: String, limit: Int) -> [String] {
        let sentences = splitSentences(paragraph)
        var chunks: [String] = []
        var current = ""

        for sentence in sentences {
            let trimmed = sentence.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }

            if trimmed.count > limit {
                if !current.isEmpty {
                    chunks.append(current.trimmingCharacters(in: .whitespacesAndNewlines))
                    current = ""
                }
                chunks.append(contentsOf: chunkLongSentence(trimmed, limit: limit))
                continue
            }

            if current.count + trimmed.count + 1 > limit && !current.isEmpty {
                chunks.append(current.trimmingCharacters(in: .whitespacesAndNewlines))
                current = ""
            }

            current += current.isEmpty ? trimmed : " " + trimmed
        }

        if !current.isEmpty {
            chunks.append(current.trimmingCharacters(in: .whitespacesAndNewlines))
        }

        return chunks
    }

    private static func chunkLongSentence(_ sentence: String, limit: Int) -> [String] {
        let parts = sentence.components(separatedBy: ",")
        var chunks: [String] = []
        var current = ""

        for part in parts {
            let trimmed = part.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }

            if trimmed.count > limit {
                if !current.isEmpty {
                    chunks.append(current.trimmingCharacters(in: .whitespacesAndNewlines))
                    current = ""
                }
                chunks.append(contentsOf: chunkByWords(trimmed, limit: limit))
                continue
            }

            if current.count + trimmed.count + 2 > limit && !current.isEmpty {
                chunks.append(current.trimmingCharacters(in: .whitespacesAndNewlines))
                current = ""
            }

            current += current.isEmpty ? trimmed : ", " + trimmed
        }

        if !current.isEmpty {
            chunks.append(current.trimmingCharacters(in: .whitespacesAndNewlines))
        }

        return chunks
    }

    private static func chunkByWords(_ text: String, limit: Int) -> [String] {
        let words = text.split(separator: " ", omittingEmptySubsequences: true)
        var chunks: [String] = []
        var current = ""

        for word in words {
            if current.count + word.count + 1 > limit && !current.isEmpty {
                chunks.append(current.trimmingCharacters(in: .whitespacesAndNewlines))
                current = ""
            }
            current += current.isEmpty ? String(word) : " " + word
        }

        if !current.isEmpty {
            chunks.append(current.trimmingCharacters(in: .whitespacesAndNewlines))
        }

        return chunks
    }

    private static func splitSentences(_ text: String) -> [String] {
        let matches = text.matches(of: sentenceRegex)
        guard !matches.isEmpty else { return [text] }

        var sentences: [String] = []
        var lastEnd = text.startIndex

        for match in matches {
            let beforePunc = String(text[lastEnd..<match.range.lowerBound])
            let punc = String(text[match.range.lowerBound])
            let combined = beforePunc.trimmingCharacters(in: .whitespaces) + punc

            let isAbbreviation = abbreviations.contains { combined.hasSuffix($0) }
            if !isAbbreviation {
                sentences.append(String(text[lastEnd..<match.range.upperBound]))
                lastEnd = match.range.upperBound
            }
        }

        if lastEnd < text.endIndex {
            sentences.append(String(text[lastEnd...]))
        }

        return sentences.isEmpty ? [text] : sentences
    }
}

// MARK: - Mask Utilities

enum MaskUtilities {
    static func makeMask(lengths: [Int], maxLength: Int? = nil) -> [[[Float]]] {
        let actualMax = maxLength ?? (lengths.max() ?? 0)
        return lengths.map { length in
            let row = (0..<actualMax).map { $0 < length ? Float(1) : Float(0) }
            return [row]
        }
    }
}
