using System.Net;
using System.Text.RegularExpressions;

namespace RagSystemWeb.State;

/// <summary>
/// Ported from dotnet/RagSystemDesktop/ViewModels/ChatMessageViewModel.cs
/// unchanged in substance -- a plain data holder either way, WPF vs
/// Blazor doesn't matter for this one.
/// </summary>
public sealed class ChatMessageState
{
    public required string Role { get; init; } // "user" | "assistant"
    public required string Text { get; init; }
    public List<string> Sources { get; init; } = [];
    public int? LogId { get; init; }

    public bool IsAssistant => Role == "assistant";

    private static readonly Regex BoldMarkdown = new(@"\*\*(.+?)\*\*", RegexOptions.Compiled);

    // The LLM writes real Markdown (mostly **bold**) into its answers, but
    // the UI was rendering it as plain text -- literal asterisks instead of
    // bold. HTML-encode first (this is LLM/document-derived text, not a
    // trusted string) so only the specific <strong> tags this method adds
    // itself ever reach the page, then convert just that one construct --
    // not a full Markdown renderer, nothing else has shown up in practice.
    public string FormattedHtml => BoldMarkdown.Replace(WebUtility.HtmlEncode(Text), "<strong>$1</strong>");

    // Exact disclaimer text, must stay identical across
    // every frontend that's ever been built for this project.
    public string? Disclaimer => IsAssistant
        ? "⚠️ AI dapat membuat kesalahan. Mohon periksa kembali dokumen sumber sebelum digunakan."
        : null;

    // Flagging is per-message UI state, not resubmitted -- once flagged,
    // the "Tandai" button shows a confirmation instead of reopening the
    // category picker. CanFlag is false whenever LogId is null (per
    // backend/app/generation.py: a combined multi-source answer's inner
    // recursive call may not produce a single log row).
    public bool IsFlagged { get; set; }
    public bool CanFlag => LogId is not null;

    // Set when FlagMessageAsync's try/catch catches a failed /flag call
    // (network blip, backend down) -- surfaced inline instead of letting
    // the exception propagate and tear down the whole Blazor circuit.
    public string? FlagError { get; set; }
}
