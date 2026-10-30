using System.Net.Http.Json;
using Microsoft.AspNetCore.WebUtilities;
using RagSystemWeb.Models;

namespace RagSystemWeb.Services;

/// <summary>Thin HTTP client for the FastAPI backend.</summary>
public sealed class ApiClient : IApiClient
{
    // Where the browser reaches the backend (the PDF preview iframe and download link).
    public const string BackendBaseUrl = "http://localhost:8000";

    // Where this server reaches the backend. Docker Compose points it at the backend container.
    public static readonly string ServerBaseUrl =
        Environment.GetEnvironmentVariable("DOCRAG_BACKEND_URL") ?? BackendBaseUrl;

    private readonly HttpClient _http;

    public ApiClient()
    {
        _http = new HttpClient
        {
            BaseAddress = new Uri(ServerBaseUrl),
            Timeout = TimeSpan.FromSeconds(450),
        };
    }

    public async Task<bool> IsHealthyAsync(CancellationToken ct = default)
    {
        try
        {
            var response = await _http.GetAsync("/health", ct);
            return response.IsSuccessStatusCode;
        }
        catch
        {
            return false;
        }
    }

    public async Task<UiStringsResponse> GetUiStringsAsync(CancellationToken ct = default)
    {
        var response = await _http.GetAsync("/ui", ct);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<UiStringsResponse>(cancellationToken: ct)
            ?? throw new HttpRequestException("GET /ui returned an empty body");
    }

    public async Task<SyncResult> SyncDocumentsAsync(CancellationToken ct = default)
    {
        var response = await _http.PostAsync("/sync", content: null, ct);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<SyncResult>(cancellationToken: ct)
            ?? throw new HttpRequestException("POST /sync returned an empty body");
    }

    public async Task<string> CreateSessionAsync(CancellationToken ct = default)
    {
        var response = await _http.PostAsync("/sessions", content: null, ct);
        response.EnsureSuccessStatusCode();
        var body = await response.Content.ReadFromJsonAsync<CreateSessionResponse>(cancellationToken: ct)
            ?? throw new HttpRequestException("POST /sessions returned an empty body");
        return body.SessionId;
    }

    public async Task<AskResponse> AskAsync(string question, string sessionId, CancellationToken ct = default)
    {
        var request = new AskRequest { Question = question, SessionId = sessionId };
        var response = await _http.PostAsJsonAsync("/ask", request, ct);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<AskResponse>(cancellationToken: ct)
            ?? throw new HttpRequestException("POST /ask returned an empty body");
    }

    public async Task<List<DocumentInfo>> GetDocumentsAsync(CancellationToken ct = default)
    {
        var response = await _http.GetAsync("/documents", ct);
        response.EnsureSuccessStatusCode();
        var body = await response.Content.ReadFromJsonAsync<DocumentListResponse>(cancellationToken: ct)
            ?? throw new HttpRequestException("GET /documents returned an empty body");
        return body.Documents;
    }

    public async Task<List<HistoryMessage>> GetHistoryAsync(string sessionId, CancellationToken ct = default)
    {
        var response = await _http.GetAsync($"/sessions/{Uri.EscapeDataString(sessionId)}/history", ct);
        response.EnsureSuccessStatusCode();
        var body = await response.Content.ReadFromJsonAsync<HistoryResponse>(cancellationToken: ct)
            ?? throw new HttpRequestException("GET /sessions/{id}/history returned an empty body");
        return body.Messages;
    }

    public async Task<List<SessionSummary>> GetSessionsAsync(CancellationToken ct = default)
    {
        var response = await _http.GetAsync("/sessions", ct);
        response.EnsureSuccessStatusCode();
        var body = await response.Content.ReadFromJsonAsync<SessionListResponse>(cancellationToken: ct)
            ?? throw new HttpRequestException("GET /sessions returned an empty body");
        return body.Sessions;
    }

    public async Task FlagAsync(int logId, string category, string? correctedAnswer = null, CancellationToken ct = default)
    {
        var queryParams = new Dictionary<string, string?> { ["category"] = category };
        if (correctedAnswer is not null)
            queryParams["corrected_answer"] = correctedAnswer;
        var url = QueryHelpers.AddQueryString($"/flag/{logId}", queryParams);

        var response = await _http.PostAsync(url, content: null, ct);
        response.EnsureSuccessStatusCode();
    }
}
