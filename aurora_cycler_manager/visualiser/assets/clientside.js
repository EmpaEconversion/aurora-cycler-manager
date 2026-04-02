window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clients: {
        animateMessage: (function() {
            let hideTimeout = null;

            return function(message) {
                const el = document.getElementById('loading-message');
                const spinner = document.getElementById('spinner-overlay');

                if (!el || !spinner) return window.dash_clientside.no_update;

                const isLoading = message && message !== '';

                // If a new message comes in cancel any pending hide
                if (hideTimeout) {
                    clearTimeout(hideTimeout);
                    hideTimeout = null;
                }

                if (isLoading) {
                    spinner.classList.add('is-loading');
                    el.classList.add('is-loading');
                    return message;
                } else {
                    // Delay hiding
                    hideTimeout = setTimeout(() => {
                        spinner.classList.remove('is-loading');
                        el.classList.remove('is-loading');
                    }, 300);

                    return window.dash_clientside.no_update;
                }
            };
        })()
    }
});