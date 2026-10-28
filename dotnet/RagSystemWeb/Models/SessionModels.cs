using System.Text.Json.Serialization;

namespace RagSystemWeb.Models;

/// <summary>POST /sessions response.</summary>
public sealed class CreateSessionResponse
{
    [JsonPropertyName("session_id")]
    public string SessionId { get; init; } = "";
}

/// <summary>One row of GET /sessions -- most-recent-first, title is a
/// server-truncated preview of the first message (or "(sesi kosong)").
/// Full message history (GET /sessions/{id}/history) lands in Phase 3.</summary>
public sealed class SessionSummary
{
    [JsonPropertyName("session_id")]
    public string SessionId { get; init; } = "";

    [JsonPropertyName("created_at")]
    public string CreatedAt { get; init; } = "";

    [JsonPropertyName("title")]
    public string Title { get; init; } = "";

    [JsonPropertyName("prompt_tokens")]
    public int PromptTokens { get; init; }

    [JsonPropertyName("response_tokens")]
    public int ResponseTokens { get; init; }

    [JsonPropertyName("total_tokens")]
    public int TotalTokens { get; init; }
}

public sealed class SessionListResponse
{
    [JsonPropertyName("sessions")]
    public List<SessionSummary> Sessions { get; init; } = [];
}

/// <summary>One row of GET /sessions/{id}/history. The backend's `messages`
/// table only stores role/content/created_at (database.py's get_history) --
/// no per-message sources or answer_log id, so a reloaded historical
/// message can't show source chips or a flag button, unlike one just
/// asked in the current circuit. Known, accepted limitation.</summary>
public sealed class HistoryMessage
{
    [JsonPropertyName("role")]
    public string Role { get; init; } = "";

    [JsonPropertyName("content")]
    public string Content { get; init; } = "";
}

public sealed class HistoryResponse
{
    [JsonPropertyName("messages")]
    public List<HistoryMessage> Messages { get; init; } = [];
}
