window.ragTheme = {
    get: function () {
        try { return localStorage.getItem('rag-theme') || 'dark'; } catch (e) { return 'dark'; }
    },
    set: function (theme) {
        try { localStorage.setItem('rag-theme', theme); } catch (e) { }
        document.documentElement.setAttribute('data-theme', theme);
    }
};
