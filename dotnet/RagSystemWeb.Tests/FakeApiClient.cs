using RagSystemWeb.Models;
using RagSystemWeb.Services;

namespace RagSystemWeb.Tests;

/// <summary>
/// Hand-rolled fake, not a mocking library -- four methods, no behavior
/// worth a framework for. Each call is counted and each response/exception
/// is settable per test via the public fields.
/// </summary>
public sealed class FakeApiClient : IApiClient
{
    public int CreateSessionCallCount { get; private set; }
    public int AskCallCount { get; private set; }

    public Func<string>? CreateSessionResult { get; set; }
    public Exception? CreateSessionThrows { get; set; }

    // Returns a Task, not a value, so a test can hand back an uncompleted
    // TaskCompletionSource.Task to control exactly when AskAsync "returns"
    // -- needed to test the IsBusy re-entrancy guard, which only matters
    // while a real call is genuinely still in flight.
    public Func<string, Task<AskResponse>>? AskResult { get; set; }
    public Exception? AskThrows { get; set; }

    public List<HistoryMessage> HistoryResult { get; set; } = [];

    public Exception? FlagThrows { get; set; }

    public Task<string> CreateSessionAsync(CancellationToken ct = default)
    {
        CreateSessionCallCount++;
        if (CreateSessionThrows is not null) throw CreateSessionThrows;
        return Task.FromResult(CreateSessionResult?.Invoke() ?? "fake-session-id");
    }

    public async Task<AskResponse> AskAsync(string question, string sessionId, CancellationToken ct = default)
    {
        AskCallCount++;
        if (AskThrows is not null) throw AskThrows;
        return AskResult is not null ? await AskResult(question) : new AskResponse();
    }

    public Task<List<HistoryMessage>> GetHistoryAsync(string sessionId, CancellationToken ct = default)
        => Task.FromResult(HistoryResult);

    public Task FlagAsync(int logId, string category, string? correctedAnswer = null, CancellationToken ct = default)
    {
        if (FlagThrows is not null) throw FlagThrows;
        return Task.CompletedTask;
    }
}
