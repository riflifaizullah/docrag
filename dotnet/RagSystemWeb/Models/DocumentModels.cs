using System.Text.Json.Serialization;

namespace RagSystemWeb.Models;

/// <summary>One row of GET /documents -- all documents returned in one call,
/// no pagination, cheap to filter client-side.</summary>
public sealed class DocumentInfo
{
    [JsonPropertyName("filename")]
    public string Filename { get; init; } = "";

    [JsonPropertyName("type")]
    public string Type { get; init; } = "";

    [JsonPropertyName("page_count")]
    public int PageCount { get; init; }

    [JsonPropertyName("size_kb")]
    public double SizeKb { get; init; }
}

public sealed class DocumentListResponse
{
    [JsonPropertyName("documents")]
    public List<DocumentInfo> Documents { get; init; } = [];
}
