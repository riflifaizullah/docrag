using RagSystemWeb.Models;
using RagSystemWeb.State;
using Xunit;

namespace RagSystemWeb.Tests;

/// <summary>Covers ChatState's non-trivial branching logic -- each test here pins down a behavior that has already regressed once in this project's.</summary>
public class ChatStateTests
{
    [Fact]
    public async Task SendAsync_SecondCallWhileBusy_IsIgnored()
    {
        var tcs = new TaskCompletionSource<AskResponse>();
        var api = new FakeApiClient { AskResult = _ => tcs.Task }; // never returns until we say so
        var chat = new ChatState(api, TestStrings.Indonesian());
        chat.InputText = "first question";

        var firstSend = chat.SendAsync(); // starts, blocks inside AskAsync
        chat.InputText = "second question";
        await chat.SendAsync(); // should return immediately, IsBusy guard

        Assert.Equal(1, api.AskCallCount);

        tcs.SetResult(new AskResponse());
        await firstSend;
    }

    [Fact]
    public async Task SendAsync_SessionCreationFails_ResetsIsBusyAndShowsError()
    {
        var api = new FakeApiClient { CreateSessionThrows = new HttpRequestException("backend down") };
        var chat = new ChatState(api, TestStrings.Indonesian());
        chat.InputText = "apa isi pasal 8?";

        await chat.SendAsync();

        Assert.False(chat.IsBusy);
        Assert.True(chat.CanSend is false); // InputText was cleared, not stuck
        Assert.Contains(chat.Messages, m => m.Role == "assistant" && m.Text.Contains("backend down"));
    }

    [Fact]
    public async Task Changed_FiresOnlyOnFirstMessageOfAChat()
    {
        var api = new FakeApiClient { AskResult = _ => Task.FromResult(new AskResponse()) };
        var chat = new ChatState(api, TestStrings.Indonesian());
        var changedCount = 0;
        chat.Changed += () => changedCount++;

        chat.InputText = "first";
        await chat.SendAsync();
        Assert.Equal(1, changedCount);

        chat.InputText = "second, same session";
        await chat.SendAsync();
        Assert.Equal(1, changedCount); // unchanged -- no second fire
    }

    [Fact]
    public async Task PickCandidateAsync_ComposesExactResumeFormat()
    {
        string? askedQuestion = null;
        var api = new FakeApiClient
        {
            AskResult = q =>
            {
                askedQuestion = q;
                return Task.FromResult(new AskResponse
                {
                    Results = [new QuestionAnswer { Question = q, CandidateDocuments = ["a.pdf", "b.pdf"] }],
                });
            },
        };
        var chat = new ChatState(api, TestStrings.Indonesian());
        chat.InputText = "Apa isi Lampiran 12?";
        await chat.SendAsync();

        await chat.PickCandidateAsync("a.pdf");

        Assert.Equal("Apa isi Lampiran 12? untuk dokumen a.pdf", askedQuestion);
    }

    [Fact]
    public async Task PickCandidateAsync_English_ComposesEnglishResumeText()
    {
        string? askedQuestion = null;
        var api = new FakeApiClient
        {
            AskResult = q =>
            {
                askedQuestion = q;
                return Task.FromResult(new AskResponse
                {
                    Results = [new QuestionAnswer { Question = q, CandidateDocuments = ["a.pdf", "b.pdf"] }],
                });
            },
        };
        var chat = new ChatState(api, TestStrings.English());
        chat.InputText = "What does Appendix 1 say?";
        await chat.SendAsync();

        await chat.PickCandidateAsync("a.pdf");

        Assert.Equal("What does Appendix 1 say? for document a.pdf", askedQuestion);
    }

    // PickCandidateAsync must no-op on a stale click (candidates already
    // dismissed), not resubmit a stale/garbage question.
    [Fact]
    public async Task PickCandidateAsync_NoOpWhenNoCandidateAwaiting()
    {
        var api = new FakeApiClient();
        var chat = new ChatState(api, TestStrings.Indonesian());

        await chat.PickCandidateAsync("whatever.pdf");

        Assert.Equal(0, api.AskCallCount);
    }

    // StartNewChat must no-op (no wasted Changed broadcast) when the chat
    // is already empty -- there's nothing to reset.
    [Fact]
    public void StartNewChat_NoOpWhenAlreadyEmpty()
    {
        var chat = new ChatState(new FakeApiClient(), TestStrings.Indonesian());
        var changedCount = 0;
        chat.Changed += () => changedCount++;

        chat.StartNewChat();

        Assert.Equal(0, changedCount);
    }

    [Fact]
    public async Task FlagMessageAsync_FailureSetsErrorInsteadOfThrowing()
    {
        var api = new FakeApiClient { FlagThrows = new HttpRequestException("500") };
        var chat = new ChatState(api, TestStrings.Indonesian());
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
        var chat = new ChatState(api, TestStrings.Indonesian());
        chat.InputText = "question";
        var send = chat.SendAsync();

        var applied = await chat.SelectSessionAsync("some-other-session");

        Assert.False(applied);
        tcs.SetResult(new AskResponse());
        await send;
    }
}
