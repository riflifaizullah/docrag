// Phase 5 theme toggle. The initial theme is already applied synchronously
// by the inline script in _Layout.cshtml (before Blazor even connects) --
// this file only handles the interactive toggle and reports the current
// value back to Blazor via JS interop.
window.ragTheme = {
    get: function () {
        try { return localStorage.getItem('rag-theme') || 'dark'; } catch (e) { return 'dark'; }
    },
    set: function (theme) {
        try { localStorage.setItem('rag-theme', theme); } catch (e) { }
        document.documentElement.setAttribute('data-theme', theme);
    }
};
