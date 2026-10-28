using System.Net;
using System.Text.RegularExpressions;

namespace RagSystemWeb.State;

/// <summary>Ported from dotnet/RagSystemDesktop/ViewModels/ChatMessageViewModel.cs unchanged in substance -- a plain data holder either way, WPF vs.</summary>
public sealed class ChatMessageState
{
    public required string Role { get; init; } // "user" | "assistant"
    public required string Text { get; init; }
    public List<string> Sources { get; init; } = [];
    public int? LogId { get; init; }

    public bool IsAssistant => Role == "assistant";

    private static readonly Regex BoldMarkdown = new(@"\*\*(.+?)\*\*", RegexOptions.Compiled);

    // The backend appends "\n\n---\n" + notes + a short disclaimer to every answer. The page shows its own
    // disclaimer, so the backend one is dropped and the remaining notes are shown apart from the answer.
    private const string FooterDivider = "\n\n---\n";
    private static readonly string[] BackendDisclaimers = ["Generated automatically", "Dihasilkan otomatis"];

    private int FooterStart => IsAssistant ? Text.IndexOf(FooterDivider, StringComparison.Ordinal) : -1;

    public string Body => FooterStart < 0 ? Text : Text[..FooterStart];

    public List<string> Notes => FooterStart < 0
        ? []
        : Text[(FooterStart + FooterDivider.Length)..]
            .Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Where(line => !BackendDisclaimers.Any(d => line.StartsWith(d, StringComparison.Ordinal)))
            .ToList();

    // "(p. 2)", "[p. 2]", "(hlm. 3, 4)": the page references the model writes after a claim.
    private static readonly Regex PageCitation = new(@"[\(\[](?:p|pp|hlm)\.\s*\d+(?:\s*[,\-\u2013]\s*\d+)*[\)\]]", RegexOptions.Compiled);

    public string FormattedHtml =>
        PageCitation.Replace(
            BoldMarkdown.Replace(WebUtility.HtmlEncode(Body), "<strong>$1</strong>"),
            "<span class=\"rag-cite\">$0</span>");

    public bool IsFlagged { get; set; }
    public bool CanFlag => LogId is not null;

    public string? FlagError { get; set; }
}
