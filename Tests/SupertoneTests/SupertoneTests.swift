import Testing
@testable import Supertone

@Test func preprocessAppendsTerminalPunctuation() async throws {
    let text = "Hello world"
    let result = preprocessText(text)
    if !result.hasSuffix(".") {
        Issue.record("Expected trailing punctuation, got \(result)")
    }
}

@Test func chunkingRespectsMaximumLength() async throws {
    let longText = Array(repeating: "Sentence.", count: 80).joined(separator: " ")
    let chunks = chunkText(longText, maxLen: 100)
    if chunks.contains(where: { $0.count > 100 }) {
        Issue.record("Chunk exceeded length limit: \(chunks)")
    }
    if chunks.isEmpty {
        Issue.record("Expected at least one chunk")
    }
}
