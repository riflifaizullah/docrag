using RagSystemWeb.State;
using Xunit;

namespace RagSystemWeb.Tests;

public class ChatMessageStateTests
{
    private const string Footer = "\n\n---\nGenerated from the whole document, not from text excerpts.\nGenerated automatically -- verify anything important.";

    private static ChatMessageState Assistant(string text) => new() { Role = "assistant", Text = text };

    [Fact]
    public void Segments_SingleAnswer_DropsBackendDisclaimerKeepsNotes()
    {
        var segments = Assistant("Answer one (p. 1)." + Footer).Segments;

        var only = Assert.Single(segments);
        Assert.Contains("Answer one", only.Html);
        Assert.Equal(["Generated from the whole document, not from text excerpts."], only.Notes);
    }

    [Fact]
    public void Segments_CombinedAnswers_SplitIntoOneSegmentEach()
    {
        var text = "First answer." + Footer + "\n\nSecond answer." + Footer;

        var segments = Assistant(text).Segments;

        Assert.Equal(2, segments.Count);
        Assert.Contains("First answer", segments[0].Html);
        Assert.Contains("Second answer", segments[1].Html);
        Assert.All(segments, s => Assert.DoesNotContain("--", s.Html));
        Assert.All(segments, s => Assert.Single(s.Notes));
    }

    [Fact]
    public void Segments_MultiDocumentNote_StaysWithItsAnswer()
    {
        var text = "First answer." + Footer + "\n\nNote: this question also matches 2 other documents.\n\nSecond answer." + Footer;

        var segments = Assistant(text).Segments;

        Assert.Equal(2, segments.Count);
        Assert.Contains(segments[0].Notes, n => n.StartsWith("Note: this question also matches"));
        Assert.Contains("Second answer", segments[1].Html);
    }

    [Fact]
    public void Segments_PageCitationBecomesPill_AndTextIsEncoded()
    {
        var segment = Assistant("Costs <b>$30</b> (p. 1).").Segments.Single();

        Assert.Contains("<span class=\"rag-cite\">(p. 1)</span>", segment.Html);
        Assert.DoesNotContain("<b>", segment.Html);
    }

    [Fact]
    public void Segments_UserMessage_IsNeverSplit()
    {
        var user = new ChatMessageState { Role = "user", Text = "a\n\n---\nb" };

        Assert.Single(user.Segments);
    }
}
