using RagSystemWeb.Services;
using RagSystemWeb.State;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddRazorPages();
builder.Services.AddServerSideBlazor();
builder.Services.AddSingleton<ApiClient>();
builder.Services.AddSingleton<IApiClient>(sp => sp.GetRequiredService<ApiClient>());
builder.Services.AddSingleton<UiStrings>();
// Scoped, not singleton: each Blazor circuit (one per open browser tab) needs
// its own chat session/messages, not one shared across every visitor.
builder.Services.AddScoped<ChatState>();

var app = builder.Build();

if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error");
}

app.Use(async (context, next) =>
{
    context.Response.Headers["X-Content-Type-Options"] = "nosniff";
    context.Response.Headers["X-Frame-Options"] = "DENY";
    context.Response.Headers["Referrer-Policy"] = "no-referrer";
    await next();
});

app.UseStaticFiles();
app.UseRouting();

app.MapBlazorHub();
app.MapFallbackToPage("/_Host");

await BackendLauncher.EnsureRunningAsync();

var ui = await app.Services.GetRequiredService<ApiClient>().GetUiStringsAsync();
app.Services.GetRequiredService<UiStrings>().Load(ui.Language, ui.Strings);

app.Run();
