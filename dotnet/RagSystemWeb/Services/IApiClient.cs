using RagSystemWeb.Models;

namespace RagSystemWeb.Services;

/// <summary>Just the four calls ChatState.cs actually makes -- extracted so tests can supply a fake instead of a real HttpClient hitting localhost:8000.</summary>
public interface IApiClient
{
    Task<string> CreateSessionAsync(CancellationToken ct = default);
    Task<AskResponse> AskAsync(string question, string sessionId, CancellationToken ct = default);
    Task<List<HistoryMessage>> GetHistoryAsync(string sessionId, CancellationToken ct = default);
    Task FlagAsync(int logId, string category, string? correctedAnswer = null, CancellationToken ct = default);
}
