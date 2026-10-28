using System.Text.Json.Serialization;

namespace RagSystemWeb.Models;

/// <summary>What POST /sync found in the corpus folder.</summary>
public sealed class SyncResult
{
    [JsonPropertyName("new")]
    public List<string> New { get; init; } = [];

    [JsonPropertyName("updated")]
    public List<string> Updated { get; init; } = [];

    [JsonPropertyName("deleted")]
    public List<string> Deleted { get; init; } = [];

    public bool HasChanges => New.Count + Updated.Count + Deleted.Count > 0;
}
