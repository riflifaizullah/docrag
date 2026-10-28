using RagSystemWeb.Services;

namespace RagSystemWeb.State;

/// <summary>
/// Ported from dotnet/RagSystemDesktop/ViewModels/ChatViewModel.cs with
/// CommunityToolkit.Mvvm's ObservableObject/[RelayCommand] dropped (no
/// Blazor equivalent needed): a Razor component owns an instance of this,
/// calls its methods directly from @onclick/@onkeydown, and re-renders by
/// calling StateHasChanged() itself after an awaited call returns --
/// there's no automatic property-changed notification to wire up the way
/// WPF's data binding needed.
/// </summary>
public sealed class ChatState
{
    private readonly IApiClient _api;
    private string? _sessionId;

    // Bumped on every SelectSessionAsync call -- found in code review: two
    // session clicks in quick succession (session A, then B before A's
    // GetHistoryAsync resolves) could race, with A's late-arriving history
    // clobbering B's just-cleared Messages. Whichever call's token no
    // longer matches when its await returns was superseded and discards
    // its own result instead of applying it.
    private int _sessionSwitchToken;

    // The question that produced the current candidate list -- needed to
    // compose "{question} untuk dokumen {filename}" when a candidate is
    // clicked, matching generation.py's _RESUME_WITH_DOCUMENT_RE exactly
    // (backend/app/generation.py:762) so the backend treats it as a
    // resumed single question with a forced source, not two split
    // fragments. Ported unchanged from the WPF version.
    private string? _questionAwaitingCandidate;

    public List<ChatMessageState> Messages { get; } = [];
    public List<string> CandidateDocuments { get; } = [];

    public string InputText { get; set; } = "";

    private bool _isBusy;
    public bool IsBusy
    {
        get => _isBusy;
        private set
        {
            if (_isBusy == value) return;
            _isBusy = value;
            BusyChanged?.Invoke();
        }
    }

    public bool ShowCandidateBar { get; private set; }

    // Separate from Changed (code review, defense in depth): Sidebar needs
    // to disable session rows for the whole duration of an in-flight ask
    // (which can run minutes on the full-document-fallback path) so a
    // session switch is never reachable mid-answer, not just rejected
    // silently by SelectSessionAsync's own IsBusy guard. Deliberately NOT
    // folded into Changed -- that event also triggers a GET /sessions
    // refetch, which IsBusy flips on every single message, not just the
    // first; this fires a plain re-render with no network call instead.
    public event Action? BusyChanged;

    // True whenever there's nothing to lose by starting a fresh chat --
    // either no backend session exists yet for the current chat, or one
    // does but nothing has been asked in it. Sidebar disables "Chat baru"
    // on this to stop the pile-up of empty sessions that used to happen
    // every page load (see the removed InitializeAsync below) and every
    // "Chat baru" click before this fix.
    public bool IsEmpty => Messages.Count == 0;

    // Fired whenever Messages/the active session changes, so sibling
    // components sharing this Scoped instance (Sidebar) can react --
    // there's no property-changed system otherwise (see the class comment).
    public event Action? Changed;

    public ChatState(IApiClient api)
    {
        _api = api;
    }

    // Resets to a fresh, not-yet-started chat -- no API call. A session is
    // only ever created lazily, in AskAsync, the moment a question is
    // actually sent. Previously this (and the old InitializeAsync, called
    // once per page load) called POST /sessions immediately, which is why
    // "(sesi kosong)" rows piled up: every page refresh, and every "Chat
    // baru" click, created one whether or not the user ever typed anything.
    public void StartNewChat()
    {
        if (IsEmpty) return;
        _sessionId = null;
        _questionAwaitingCandidate = null;
        DismissCandidates();
        Messages.Clear();
        Changed?.Invoke();
    }

    // Wired from Sidebar's session click / "Chat baru" (via Index.razor) --
    // switches which backend session subsequent Ask calls target and
    // reloads that session's messages. Found missing entirely: clicking a
    // Riwayat row only updated Index's SelectedSessionId for highlighting,
    // it never actually loaded history or redirected Ask, so every message
    // silently went to whatever session InitializeAsync() had created.
    //
    // Returns whether the switch actually applied, so Index.razor only
    // commits its own highlighted-session state on success (code review
    // finding: it used to commit unconditionally, so clicking a session row
    // while IsBusy -- this silently no-ops below -- left the sidebar
    // highlighting a session ChatColumn wasn't actually showing).
    public async Task<bool> SelectSessionAsync(string sessionId)
    {
        if (IsBusy) return false;
        var requestId = ++_sessionSwitchToken;
        _questionAwaitingCandidate = null;
        DismissCandidates();

        var history = await _api.GetHistoryAsync(sessionId);

        // Stale-response guard: a second SelectSessionAsync call (a fast
        // second click) can start and finish while this one was still
        // awaiting history -- discard this call's result instead of
        // clobbering whatever the newer call already applied.
        if (requestId != _sessionSwitchToken) return false;

        _sessionId = sessionId;
        Messages.Clear();
        foreach (var message in history)
        {
            Messages.Add(new ChatMessageState { Role = message.Role, Text = message.Content });
        }
        // No Changed?.Invoke() here (code review finding): Sidebar's handler
        // refetches the session list, which doesn't change just from
        // switching which one is selected -- the highlight is already
        // driven directly by the SelectedSessionId parameter. ChatColumn's
        // own redraw comes from Index.razor's StateHasChanged() after this
        // call returns, same as before ChatColumn subscribed to Changed.
        return true;
    }

    public bool CanSend => !IsBusy && !string.IsNullOrWhiteSpace(InputText);

    public async Task SendAsync()
    {
        if (!CanSend) return;
        var question = InputText.Trim();
        InputText = "";
        await AskAsync(question);
    }

    public async Task PickCandidateAsync(string filename)
    {
        if (_questionAwaitingCandidate is null) return;
        DismissCandidates();
        await AskAsync($"{_questionAwaitingCandidate} untuk dokumen {filename}");
    }

    public void DismissCandidates()
    {
        ShowCandidateBar = false;
        CandidateDocuments.Clear();
    }

    public async Task FlagMessageAsync(ChatMessageState message, string category, string? correctedAnswer = null)
    {
        if (message.LogId is not { } logId) return;
        // Found in code review: FlagAsync's EnsureSuccessStatusCode() throws
        // on any non-2xx (backend down, timeout, 500). Left uncaught, that
        // exception tears down the whole Blazor Server circuit -- the user
        // loses the entire chat session over a failed flag click, a much
        // worse failure mode than AskAsync's own try/catch below already
        // guards against for the ask path. Same defensive pattern here.
        try
        {
            await _api.FlagAsync(logId, category, correctedAnswer);
            message.IsFlagged = true;
        }
        catch (Exception ex)
        {
            message.FlagError = $"Gagal menandai: {ex.Message}";
        }
    }

    private async Task AskAsync(string question)
    {
        // Busy-flag guard, not just a UI nicety: a real
        // Streamlit bug where a second submission while one was in flight
        // produced a duplicated/garbled render. The Razor component disables
        // the composer via IsBusy in its markup; this is the defense-in-depth
        // check, ported unchanged from the WPF version.
        if (IsBusy) return;
        IsBusy = true;
        DismissCandidates();

        var isFirstMessage = _sessionId is null;
        Messages.Add(new ChatMessageState { Role = "user", Text = question });

        try
        {
            // Lazy session creation -- only now, on the first real question,
            // not on page load or "Chat baru" (see StartNewChat's comment).
            // Must stay inside this try: found in code review -- it used to
            // run before the try, so a failed POST /sessions (backend down)
            // threw past the `finally` below, leaving IsBusy stuck true
            // forever (composer permanently disabled, question already
            // cleared from InputText and lost) instead of hitting the same
            // error handling every other backend failure here already gets.
            _sessionId ??= await _api.CreateSessionAsync();

            var response = await _api.AskAsync(question, _sessionId);
            foreach (var result in response.Results)
            {
                Messages.Add(new ChatMessageState
                {
                    Role = "assistant",
                    Text = result.Answer,
                    Sources = result.Sources,
                    LogId = result.LogId,
                });

                if (result.CandidateDocuments.Count > 0)
                {
                    _questionAwaitingCandidate = result.Question;
                    CandidateDocuments.Clear();
                    CandidateDocuments.AddRange(result.CandidateDocuments);
                    ShowCandidateBar = true;
                }
            }
        }
        catch (Exception ex)
        {
            Messages.Add(new ChatMessageState
            {
                Role = "assistant",
                Text = $"[Gagal menghubungi backend: {ex.Message}]",
            });
        }
        finally
        {
            IsBusy = false;
            // Only on the first message of a chat (code review finding):
            // Sidebar's handler does a full GET /sessions round-trip, and
            // the list/title only actually change once, when the session is
            // created -- firing this on every follow-up turn would be one
            // wasted HTTP call per message for no visible difference.
            if (isFirstMessage) Changed?.Invoke();
        }
    }
}
