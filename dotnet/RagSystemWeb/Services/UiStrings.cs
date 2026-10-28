namespace RagSystemWeb.Services;

/// <summary>Interface text in the backend's language, fetched once from GET /ui so each language has one source.</summary>
public sealed class UiStrings
{
    private Dictionary<string, string> _strings;

    public string Language { get; private set; }

    public UiStrings() : this("en", new Dictionary<string, string>()) { }

    public UiStrings(string language, Dictionary<string, string> strings)
    {
        Language = language;
        _strings = strings;
    }

    public void Load(string language, Dictionary<string, string> strings)
    {
        Language = language;
        _strings = strings;
    }

    /// <summary>A missing key shows the key itself, which is easy to spot.</summary>
    public string this[string key] => _strings.TryGetValue(key, out var value) ? value : key;

    public string Format(string key, params object[] args) => string.Format(this[key], args);

    /// <summary>The text sent when the user picks one of several candidate documents.</summary>
    public string Resume(string question, string document) =>
        this["resume_template"].Replace("{question}", question).Replace("{document}", document);
}
