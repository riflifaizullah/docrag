using RagSystemWeb.Services;

namespace RagSystemWeb.Tests;

public static class TestStrings
{
    public static UiStrings Indonesian() => new("id", new Dictionary<string, string>
    {
        ["resume_template"] = "{question} untuk dokumen {document}",
        ["flag_failed"] = "Gagal menandai: {0}",
        ["backend_unreachable"] = "[Gagal menghubungi backend: {0}]",
    });

    public static UiStrings English() => new("en", new Dictionary<string, string>
    {
        ["resume_template"] = "{question} for document {document}",
        ["flag_failed"] = "Could not flag: {0}",
        ["backend_unreachable"] = "[Could not reach the backend: {0}]",
    });
}
