using System.Net.Http.Json;
using Microsoft.AspNetCore.WebUtilities;
using RagSystemWeb.Models;

namespace RagSystemWeb.Services;

/// <summary>
/// Thin HTTP client for the FastAPI backend. Registered as a singleton in
/// Program.cs -- one shared HttpClient for the app's life, never
/// `new HttpClient()` per call. Ported unchanged from the WPF attempt
/// (dotnet/RagSystemDesktop/Services/ApiClient.cs) -- this logic doesn't
/// care whether the UI is WPF or Blazor.
/// </summary>
public sealed class ApiClient : IApiClient
{
    // Shared with PreviewPanel.razor so the <iframe>/download link don't
    // duplicate this literal.
    public const string BackendBaseUrl = "http://localhost:8000";

    private readonly HttpClient _http;

    public ApiClient()
    {
        _http = new HttpClient
        {
            BaseAddress = new Uri(BackendBaseUrl),
            // Full-document-fallback answers can legitimately take minutes
            // (OLLAMA_FULL_DOC_TIMEOUT_SECONDS = 420 in the backend config) -- a
            // short client timeout would cut those off mid-answer.
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
        // backend/app/api.py's flag() takes category/corrected_answer as plain
        // FastAPI query params, NOT a JSON body -- PostAsJsonAsync would send
        // a body the endpoint never reads and fail. Build the query string
        // explicitly instead. This exact mistake was flagged as a risk in
        // the plan file for both the WPF and the Blazor attempts.
        var queryParams = new Dictionary<string, string?> { ["category"] = category };
        if (correctedAnswer is not null)
            queryParams["corrected_answer"] = correctedAnswer;
        var url = QueryHelpers.AddQueryString($"/flag/{logId}", queryParams);

        var response = await _http.PostAsync(url, content: null, ct);
        response.EnsureSuccessStatusCode();
    }
}
