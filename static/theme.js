(function () {
  const STORAGE_KEY = 'careerpath-theme';
  const root = document.documentElement;

  function applyTheme(theme) {
    root.setAttribute('data-theme', theme);
    const icon = document.querySelector('[data-theme-icon]');
    const label = document.querySelector('[data-theme-label]');

    if (icon) {
      icon.textContent = theme === 'dark' ? String.fromCodePoint(0x2600) : String.fromCodePoint(0x1F319);
    }

    if (label) {
      label.textContent = theme === 'dark' ? 'Original' : 'Dark';
    }
  }

  function animateThemeSwitch(nextTheme, toggle) {
    const body = document.body;

    // Use requestAnimationFrame for smoother animation timing
    requestAnimationFrame(() => {
      body.classList.add('theme-switching');
      if (toggle) {
        toggle.classList.add('is-switching');
      }

      // Allow paint frame before changing theme
      requestAnimationFrame(() => {
        applyTheme(nextTheme);

        // Wait for theme change to paint, then remove animation classes
        requestAnimationFrame(() => {
          setTimeout(() => {
            body.classList.remove('theme-switching');
            if (toggle) {
              toggle.classList.remove('is-switching');
            }
          }, 350);
        });
      });
    });
  }

  function detectInitialTheme() {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved === 'dark' || saved === 'seabreeze') {
      return saved;
    }
    return 'seabreeze';
  }

  document.addEventListener('DOMContentLoaded', function () {
    const initialTheme = detectInitialTheme();
    applyTheme(initialTheme);

    const toggle = document.getElementById('themeToggle');
    if (!toggle) {
      return;
    }

    toggle.addEventListener('click', function () {
      const current = root.getAttribute('data-theme');
      const next = current === 'dark' ? 'seabreeze' : 'dark';
      animateThemeSwitch(next, toggle);
      localStorage.setItem(STORAGE_KEY, next);
    });
  });
})();
