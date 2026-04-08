window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clients: {
        animateMessage: function(message) {
            const el = document.getElementById('loading-message');
            if (!el) return window.dash_clientside.no_update;

            // Remove class, force reflow, re-add to trigger animation
            el.classList.remove('fade-in');
            void el.offsetWidth;  // Force reflow
            el.classList.add('fade-in');
            el.textContent = message;
            return null;
        }
    }
});