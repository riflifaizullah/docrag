using System.Diagnostics;

namespace RagSystemWeb.Services;

/// <summary>
/// Ported from dotnet/RagSystemDesktop/Services/BackendLauncher.cs (the WPF
/// attempt) with the splash-window coupling dropped: Program.cs awaits
/// EnsureRunningAsync() before app.Run(), so no HTTP request is served
/// until the backend is confirmed up -- there's nothing else to notify.
/// Status goes to the console instead of a StatusChanged event.
///
/// Confirmed live during Phase 0 verification: under `dotnet run` (not a
/// published exe), the spawned uvicorn process is bound to the same Job
/// Object as the `dotnet run` process tree, so killing the running app
/// (or its inner apphost) also kills the backend -- unlike Streamlit's
/// plain subprocess.Popen, which truly outlives its parent. This only
/// showed up when killing the *inner* apphost process specifically; the
/// outer `dotnet run` CLI muxer process survives independently. Not
/// fixed here (would need CREATE_BREAKAWAY_FROM_JOB on the spawned
/// process) -- just means the dev inner-loop respawns the backend on
/// every `dotnet run` restart, which is slower but not broken.
/// </summary>
public static class BackendLauncher
{
    private const string ApiBase = "http://localhost:8000";

    // Same depth as the WPF project: dotnet/RagSystemWeb/bin/Debug/net8.0/
    // is the same number of levels below the repo root as
    // dotnet/RagSystemDesktop/bin/Debug/net8.0-windows/ was (the TFM folder
    // name differs, the depth doesn't). Still a Phase-0-only assumption --
    // breaks under `dotnet publish`, not fixed here, same as the WPF version.
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

        Console.WriteLine("[BackendLauncher] Menyalakan backend API (bisa 30-60 detik pada startup pertama)...");

        if (TryAcquireLock())
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

        Console.WriteLine("[BackendLauncher] Backend tidak merespons setelah 120 detik. Periksa log atau jalankan manual.");
        return false;
    }

    /// <summary>
    /// Atomic exclusive-create (not read-then-write) -- a real TOCTOU race
    /// was found and fixed at this exact spot in the WPF version's code
    /// review; ported with the fix already applied, not reintroduced.
    /// </summary>
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
        psi.ArgumentList.Add("0.0.0.0");
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
