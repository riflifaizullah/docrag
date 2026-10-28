window.ragResize = {
    init: function (handle, panel, storageKey, min, max) {
        if (!handle || !panel) return;

        try {
            var saved = localStorage.getItem(storageKey);
            if (saved) panel.style.width = saved + 'px';
        } catch (e) { }

        var dragging = false, startX = 0, startWidth = 0;

        function onMove(e) {
            if (!dragging) return;
            var delta = startX - e.clientX;
            var width = Math.min(max, Math.max(min, startWidth + delta));
            panel.style.width = width + 'px';
        }

        function onUp() {
            if (!dragging) return;
            dragging = false;
            document.body.style.userSelect = '';
            document.body.style.cursor = '';
            try { localStorage.setItem(storageKey, parseInt(panel.style.width, 10)); } catch (e) { }
            document.removeEventListener('mousemove', onMove);
            document.removeEventListener('mouseup', onUp);
        }

        handle.addEventListener('mousedown', function (e) {
            dragging = true;
            startX = e.clientX;
            startWidth = panel.getBoundingClientRect().width;
            document.body.style.userSelect = 'none';
            document.body.style.cursor = 'ew-resize';
            document.addEventListener('mousemove', onMove);
            document.addEventListener('mouseup', onUp);
            e.preventDefault();
        });
    }
};
