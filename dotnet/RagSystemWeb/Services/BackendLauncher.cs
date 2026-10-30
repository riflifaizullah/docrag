using System.Diagnostics;

namespace RagSystemWeb.Services;

/// <summary>Ported from dotnet/RagSystemDesktop/Services/BackendLauncher.cs (the WPF attempt) with the splash-window coupling dropped: Program.cs.</summary>
public static class BackendLauncher
{
    private static readonly string ApiBase = ApiClient.ServerBaseUrl;

    // With DOCRAG_BACKEND_URL set the backend is someone else's job (a container), so only wait for it.
    private static readonly bool ExternalBackend =
        Environment.GetEnvironmentVariable("DOCRAG_BACKEND_URL") is not null;

    private static readonly string BackendDir = Path.GetFullPath(
        Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "backend"));
    private static readonly string LockPath = Path.Combine(BackendDir, "data", ".api_starting.lock");
    private static readonly string LogPath = Path.Combine(BackendDir, "data", "dotnet_api_launch.log");

    // Set DOCRAG_PYTHON to use a specific interpreter (a venv or conda env).
    private static readonly string[] PythonCandidates =
    [
        Environment.GetEnvironmentVariable("DOCRAG_PYTHON") ?? "python",
        "py",
    ];

    public static async Task<bool> EnsureRunningAsync(CancellationToken ct = default)
    {
        using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(3) };

        if (await IsUpAsync(http, ct))
        {
            Console.WriteLine("[BackendLauncher] Backend already up.");
            return true;
        }

        Console.WriteLine("[BackendLauncher] Starting the backend API (can take 30-60 seconds on first start)...");

        if (!ExternalBackend && TryAcquireLock())
        {
            try
            {
                SpawnApiServer();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[BackendLauncher] SpawnApiServer failed: {ex}");
                try { File.Delete(LockPath); } catch { /* best-effort */ }
            }
        }

        var deadline = DateTime.UtcNow + TimeSpan.FromSeconds(120);
        while (DateTime.UtcNow < deadline)
        {
            if (await IsUpAsync(http, ct))
            {
                try { File.Delete(LockPath); } catch { /* best-effort */ }
                Console.WriteLine("[BackendLauncher] Backend is up.");
                return true;
            }
            await Task.Delay(2000, ct);
        }

        Console.WriteLine("[BackendLauncher] Backend did not respond after 120 seconds. Check the log or start it manually.");
        return false;
    }

    /// <summary>Atomic exclusive-create (not read-then-write) -- a real TOCTOU race was found and fixed at this exact spot in the WPF version's code.</summary>
    private static bool TryAcquireLock()
    {
        Directory.CreateDirectory(Path.GetDirectoryName(LockPath)!);

        if (File.Exists(LockPath) &&
            (DateTime.UtcNow - File.GetLastWriteTimeUtc(LockPath)) >= TimeSpan.FromSeconds(130))
        {
            try { File.Delete(LockPath); } catch { /* another instance may have just cleared it */ }
        }

        try
        {
            using var lockFile = new FileStream(LockPath, FileMode.CreateNew, FileAccess.Write, FileShare.None);
            return true;
        }
        catch (IOException)
        {
            return false;
        }
    }

    private static async Task<bool> IsUpAsync(HttpClient http, CancellationToken ct)
    {
        try
        {
            var response = await http.GetAsync($"{ApiBase}/health", ct);
            return response.IsSuccessStatusCode;
        }
        catch
        {
            return false;
        }
    }

    private static void SpawnApiServer()
    {
        var python = PythonCandidates.FirstOrDefault(File.Exists) ?? PythonCandidates[^1];

        Directory.CreateDirectory(Path.GetDirectoryName(LogPath)!);
        var psi = new ProcessStartInfo
        {
            FileName = python,
            WorkingDirectory = BackendDir,
            UseShellExecute = false,
            CreateNoWindow = true,
            WindowStyle = ProcessWindowStyle.Hidden,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        psi.ArgumentList.Add("-m");
        psi.ArgumentList.Add("uvicorn");
        psi.ArgumentList.Add("app.api:app");
        psi.ArgumentList.Add("--host");
        // The API has no sign-in, so it listens on this machine only. Set DOCRAG_API_HOST to change that on purpose.
        psi.ArgumentList.Add(Environment.GetEnvironmentVariable("DOCRAG_API_HOST") ?? "127.0.0.1");
        psi.ArgumentList.Add("--port");
        psi.ArgumentList.Add("8000");

        var process = new Process { StartInfo = psi };
        process.OutputDataReceived += (_, e) => AppendLog(e.Data);
        process.ErrorDataReceived += (_, e) => AppendLog(e.Data);
        process.Start();
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();
    }

    private static void AppendLog(string? line)
    {
        if (line is null) return;
        try { File.AppendAllText(LogPath, line + Environment.NewLine); } catch { /* best-effort */ }
    }
}
