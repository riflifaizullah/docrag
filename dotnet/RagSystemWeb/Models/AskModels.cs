using System.Text.Json.Serialization;

namespace RagSystemWeb.Models;

/// <summary>
/// Field names match backend/app/api.py's Pydantic models exactly (plain
/// snake_case, no camelCase aliasing anywhere in the backend) -- confirmed
/// by reading api.py directly, not guessed. Ported unchanged from the WPF
/// attempt (dotnet/RagSystemDesktop/Models/AskModels.cs) -- these are
/// framework-agnostic POCOs.
/// </summary>
public sealed class AskRequest
{
    [JsonPropertyName("question")]
    public required string Question { get; init; }

    [JsonPropertyName("session_id")]
    public string? SessionId { get; init; }
}

public sealed class QuestionAnswer
{
    [JsonPropertyName("question")]
    public string Question { get; init; } = "";

    [JsonPropertyName("answer")]
    public string Answer { get; init; } = "";

    [JsonPropertyName("answered")]
    public bool Answered { get; init; }

    [JsonPropertyName("refused")]
    public bool Refused { get; init; }

    [JsonPropertyName("needs_clarification")]
    public bool NeedsClarification { get; init; }

    [JsonPropertyName("candidate_documents")]
    public List<string> CandidateDocuments { get; init; } = [];

    [JsonPropertyName("sources")]
    public List<string> Sources { get; init; } = [];

    [JsonPropertyName("log_id")]
    public int? LogId { get; init; }
}

public sealed class AskResponse
{
    [JsonPropertyName("results")]
    public List<QuestionAnswer> Results { get; init; } = [];

    [JsonPropertyName("answered")]
    public bool Answered { get; init; }

    [JsonPropertyName("refused")]
    public bool Refused { get; init; }

    [JsonPropertyName("needs_clarification")]
    public bool NeedsClarification { get; init; }

    [JsonPropertyName("candidate_documents")]
    public List<string> CandidateDocuments { get; init; } = [];
}
