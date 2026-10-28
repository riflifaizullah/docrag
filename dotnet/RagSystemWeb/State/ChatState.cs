using RagSystemWeb.Services;

namespace RagSystemWeb.State;

/// <summary>Ported from dotnet/RagSystemDesktop/ViewModels/ChatViewModel.cs with CommunityToolkit.Mvvm's ObservableObject/[RelayCommand] dropped (no.</summary>
public sealed class ChatState
{
    private readonly IApiClient _api;
    private readonly UiStrings _ui;
    private string? _sessionId;

    private int _sessionSwitchToken;

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

    public event Action? BusyChanged;

    public bool IsEmpty => Messages.Count == 0;

    public event Action? Changed;

    public ChatState(IApiClient api, UiStrings ui)
    {
        _api = api;
        _ui = ui;
    }

    public void StartNewChat()
    {
        if (IsEmpty) return;
        _sessionId = null;
        _questionAwaitingCandidate = null;
        DismissCandidates();
        Messages.Clear();
        Changed?.Invoke();
    }

    public async Task<bool> SelectSessionAsync(string sessionId)
    {
        if (IsBusy) return false;
        var requestId = ++_sessionSwitchToken;
        _questionAwaitingCandidate = null;
        DismissCandidates();

        var history = await _api.GetHistoryAsync(sessionId);

        if (requestId != _sessionSwitchToken) return false;

        _sessionId = sessionId;
        Messages.Clear();
        foreach (var message in history)
        {
            Messages.Add(new ChatMessageState { Role = message.Role, Text = message.Content });
        }
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
        await AskAsync(_ui.Resume(_questionAwaitingCandidate, filename));
    }

    public void DismissCandidates()
    {
        ShowCandidateBar = false;
        CandidateDocuments.Clear();
    }

    public async Task FlagMessageAsync(ChatMessageState message, string category, string? correctedAnswer = null)
    {
        if (message.LogId is not { } logId) return;
        try
        {
            await _api.FlagAsync(logId, category, correctedAnswer);
            message.IsFlagged = true;
        }
        catch (Exception ex)
        {
            message.FlagError = _ui.Format("flag_failed", ex.Message);
        }
    }

    private async Task AskAsync(string question)
    {
        if (IsBusy) return;
        IsBusy = true;
        DismissCandidates();

        var isFirstMessage = _sessionId is null;
        Messages.Add(new ChatMessageState { Role = "user", Text = question });

        try
        {
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
                Text = _ui.Format("backend_unreachable", ex.Message),
            });
        }
        finally
        {
            IsBusy = false;
            if (isFirstMessage) Changed?.Invoke();
        }
    }
}
