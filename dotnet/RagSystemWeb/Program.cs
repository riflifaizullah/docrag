using RagSystemWeb.Services;
using RagSystemWeb.State;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddRazorPages();
builder.Services.AddServerSideBlazor();
builder.Services.AddSingleton<ApiClient>();
// ChatState depends on IApiClient (not the concrete ApiClient) so tests can
// supply a fake -- same singleton instance either way, Sidebar/PreviewPanel
// still inject the concrete ApiClient directly for the methods/const not on
// the interface.
builder.Services.AddSingleton<IApiClient>(sp => sp.GetRequiredService<ApiClient>());
// Scoped, not singleton: each Blazor circuit (one per open browser tab) needs
// its own chat session/messages, not one shared across every visitor.
builder.Services.AddScoped<ChatState>();

var app = builder.Build();

if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error");
}

app.UseStaticFiles();
app.UseRouting();

app.MapBlazorHub();
app.MapFallbackToPage("/_Host");

// Blocking on startup is acceptable here: this is a local dev tool run via
// `dotnet run`, not a scaled web deploy, and matches Streamlit's own
// synchronous "wait for the backend before serving" gate.
await BackendLauncher.EnsureRunningAsync();

app.Run();
