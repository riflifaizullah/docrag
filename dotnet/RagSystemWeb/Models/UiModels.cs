using System.Text.Json.Serialization;

namespace RagSystemWeb.Models;

public sealed class UiStringsResponse
{
    [JsonPropertyName("language")]
    public string Language { get; init; } = "en";

    [JsonPropertyName("strings")]
    public Dictionary<string, string> Strings { get; init; } = [];
}
