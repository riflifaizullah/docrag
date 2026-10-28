using RagSystemWeb.Models;
using RagSystemWeb.State;
using Xunit;

namespace RagSystemWeb.Tests;

/// <summary>
/// Covers ChatState's non-trivial branching logic -- each test here pins
/// down a behavior that has already regressed once in this project's
/// history (see the comments at the matching line in ChatState.cs for the
/// real bug each one guards against). Deliberately not exhaustive: no
/// bUnit/DOM tests, the .razor files are thin views over this class.
/// </summary>
public class ChatStateTests
{
    // Regression test for the historical Streamlit double-submit bug
    // (ChatState.cs's AskAsync IsBusy guard comment): a second SendAsync
    // while the first is still in flight must be a no-op, not a second
    // concurrent /ask call.
    [Fact]
    public async Task SendAsync_SecondCallWhileBusy_IsIgnored()
    {
        var tcs = new TaskCompletionSource<AskResponse>();
        var api = new FakeApiClient { AskResult = _ => tcs.Task }; // never returns until we say so
        var chat = new ChatState(api);
        chat.InputText = "first question";

        var firstSend = chat.SendAsync(); // starts, blocks inside AskAsync
        chat.InputText = "second question";
        await chat.SendAsync(); // should return immediately, IsBusy guard

        Assert.Equal(1, api.AskCallCount);

        tcs.SetResult(new AskResponse());
        await firstSend;
    }

    // Regression test: CreateSessionAsync failing used to run BEFORE the
    // try/catch, leaving IsBusy stuck true forever and the typed question
    // lost. Now it must hit the same error handling every other backend
    // failure gets, and leave the composer usable again.
    [Fact]
    public async Task SendAsync_SessionCreationFails_ResetsIsBusyAndShowsError()
    {
        var api = new FakeApiClient { CreateSessionThrows = new HttpRequestException("backend down") };
        var chat = new ChatState(api);
        chat.InputText = "apa isi pasal 8?";

        await chat.SendAsync();

        Assert.False(chat.IsBusy);
        Assert.True(chat.CanSend is false); // InputText was cleared, not stuck
        Assert.Contains(chat.Messages, m => m.Role == "assistant" && m.Text.Contains("backend down"));
    }

    // Regression test: Changed must fire on the first message of a chat
    // (so Sidebar picks up the newly-created session) but NOT on every
    // follow-up turn (that would be one wasted GET /sessions per message).
    [Fact]
    public async Task Changed_FiresOnlyOnFirstMessageOfAChat()
    {
        var api = new FakeApiClient { AskResult = _ => Task.FromResult(new AskResponse()) };
        var chat = new ChatState(api);
        var changedCount = 0;
        chat.Changed += () => changedCount++;

        chat.InputText = "first";
        await chat.SendAsync();
        Assert.Equal(1, changedCount);

        chat.InputText = "second, same session";
        await chat.SendAsync();
        Assert.Equal(1, changedCount); // unchanged -- no second fire
    }

    // The composed resume string must match generation.py's
    // _RESUME_WITH_DOCUMENT_RE exactly (backend/app/generation.py:762) --
    // this is a cross-repo contract nothing else enforces.
    [Fact]
    public async Task PickCandidateAsync_ComposesExactResumeFormat()
    {
        string? askedQuestion = null;
        var api = new FakeApiClient
        {
            AskResult = q =>
            {
                askedQuestion = q;
                // First call returns candidates (sets _questionAwaitingCandidate,
                // matching real AskAsync); second call (from PickCandidateAsync)
                // just needs to succeed.
                return Task.FromResult(new AskResponse
                {
                    Results = [new QuestionAnswer { Question = q, CandidateDocuments = ["a.pdf", "b.pdf"] }],
                });
            },
        };
        var chat = new ChatState(api);
        chat.InputText = "Apa isi Lampiran 12?";
        await chat.SendAsync();

        await chat.PickCandidateAsync("a.pdf");

        Assert.Equal("Apa isi Lampiran 12? untuk dokumen a.pdf", askedQuestion);
    }

    // PickCandidateAsync must no-op on a stale click (candidates already
    // dismissed), not resubmit a stale/garbage question.
    [Fact]
    public async Task PickCandidateAsync_NoOpWhenNoCandidateAwaiting()
    {
        var api = new FakeApiClient();
        var chat = new ChatState(api);

        await chat.PickCandidateAsync("whatever.pdf");

        Assert.Equal(0, api.AskCallCount);
    }

    // StartNewChat must no-op (no wasted Changed broadcast) when the chat
    // is already empty -- there's nothing to reset.
    [Fact]
    public void StartNewChat_NoOpWhenAlreadyEmpty()
    {
        var chat = new ChatState(new FakeApiClient());
        var changedCount = 0;
        chat.Changed += () => changedCount++;

        chat.StartNewChat();

        Assert.Equal(0, changedCount);
    }

    // FlagMessageAsync must catch a failed /flag call and surface it on
    // the message, not let it propagate and tear down the whole circuit
    // (ApiClient.FlagAsync's EnsureSuccessStatusCode throws on any non-2xx).
    [Fact]
    public async Task FlagMessageAsync_FailureSetsErrorInsteadOfThrowing()
    {
        var api = new FakeApiClient { FlagThrows = new HttpRequestException("500") };
        var chat = new ChatState(api);
        var message = new ChatMessageState { Role = "assistant", Text = "answer", LogId = 42 };

        await chat.FlagMessageAsync(message, "bad_retrieval");

        Assert.False(message.IsFlagged);
        Assert.NotNull(message.FlagError);
    }

    // SelectSessionAsync must refuse to run while an ask is in flight
    // (IsBusy), and report that it didn't apply.
    [Fact]
    public async Task SelectSessionAsync_ReturnsFalseWhileBusy()
    {
        var tcs = new TaskCompletionSource<AskResponse>();
        var api = new FakeApiClient { AskResult = _ => tcs.Task };
        var chat = new ChatState(api);
        chat.InputText = "question";
        var send = chat.SendAsync();

        var applied = await chat.SelectSessionAsync("some-other-session");

        Assert.False(applied);
        tcs.SetResult(new AskResponse());
        await send;
    }
}
