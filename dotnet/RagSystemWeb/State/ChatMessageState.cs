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

    // The backend appends "\n\n---\n" + notes + a short disclaimer to every answer, and a chat that held several
    // questions is stored as those answers joined together. The page shows its own disclaimer, so the backend one
    // is dropped, and each answer is shown with its own notes.
    private const string FooterDivider = "\n\n---\n";
    private static readonly string[] BackendDisclaimers = ["Generated automatically", "Dihasilkan otomatis"];
    private static readonly string[] NotePrefixes = ["Note:", "Catatan:"];

    // "(p. 2)", "[p. 2]", "(hlm. 3, 4)": the page references the model writes after a claim.
    private static readonly Regex PageCitation = new(@"[\(\[](?:p|pp|hlm)\.\s*\d+(?:\s*[,\-\u2013]\s*\d+)*[\)\]]", RegexOptions.Compiled);

    public sealed record AnswerSegment(string Html, List<string> Notes);

    public List<AnswerSegment> Segments => IsAssistant ? ParseSegments() : [new(Format(Text), [])];

    private static string Format(string text) =>
        PageCitation.Replace(
            BoldMarkdown.Replace(WebUtility.HtmlEncode(text), "<strong>$1</strong>"),
            "<span class=\"rag-cite\">$0</span>");

    private static List<string> NoteLines(string block) =>
        block.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Where(line => !BackendDisclaimers.Any(d => line.StartsWith(d, StringComparison.Ordinal)))
            .ToList();

    private List<AnswerSegment> ParseSegments()
    {
        var parts = Text.Split(FooterDivider);
        var segments = new List<AnswerSegment>();
        var body = parts[0];
        if (parts.Length == 1) return [new(Format(body), [])];

        for (var i = 1; i < parts.Length; i++)
        {
            var rest = parts[i];
            var next = "";
            var extraNotes = new List<string>();
            if (i < parts.Length - 1)
            {
                // Another answer follows: its text starts after the first blank line of this block.
                var cut = rest.IndexOf("\n\n", StringComparison.Ordinal);
                if (cut >= 0)
                {
                    next = rest[(cut + 2)..];
                    rest = rest[..cut];
                    // A multi-document note sits between the footer and the next answer.
                    while (NotePrefixes.Any(p => next.StartsWith(p, StringComparison.Ordinal)))
                    {
                        var end = next.IndexOf("\n\n", StringComparison.Ordinal);
                        extraNotes.Add((end < 0 ? next : next[..end]).Trim());
                        next = end < 0 ? "" : next[(end + 2)..];
                    }
                }
            }
            var notes = NoteLines(rest);
            notes.AddRange(extraNotes);
            segments.Add(new(Format(body), notes));
            body = next;
        }
        return segments;
    }

    public bool IsFlagged { get; set; }
    public bool CanFlag => LogId is not null;

    public string? FlagError { get; set; }
}
