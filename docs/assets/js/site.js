/* ==========================================================================
   Shared site behaviour: navigation, theme, contents rail, filtering.
   Written to run from file:// as well as a web server, so faculty can open
   the pages directly from a downloaded folder.
   ========================================================================== */

(function () {
  "use strict";

  var PAGES = [
    { file: "index.html",      label: "Overview" },
    { file: "how-it-works.html", label: "How It Works" },
    { file: "controls.html",   label: "Sidebar Controls" },
    { file: "techniques.html", label: "Advanced Techniques" },
    { file: "settings.html",   label: "Model Settings" },
    { file: "evaluation.html", label: "Evaluation" },
    { file: "glossary.html",   label: "Glossary" }
  ];

  /* ---------------------------------------------------------------------
     Theme: remembered per browser, defaults to the operating system.
     --------------------------------------------------------------------- */
  var THEME_KEY = "llmdocs-theme";

  function storedTheme() {
    try { return localStorage.getItem(THEME_KEY); } catch (e) { return null; }
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute("data-bs-theme", theme);
    var btn = document.getElementById("themeToggle");
    if (btn) {
      var dark = theme === "dark";
      btn.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
      btn.textContent = dark ? "☀" : "☾";
    }
  }

  function initTheme() {
    var saved = storedTheme();
    var prefersDark = window.matchMedia &&
      window.matchMedia("(prefers-color-scheme: dark)").matches;
    applyTheme(saved || (prefersDark ? "dark" : "light"));
  }

  function toggleTheme() {
    var next = document.documentElement.getAttribute("data-bs-theme") === "dark" ? "light" : "dark";
    applyTheme(next);
    try { localStorage.setItem(THEME_KEY, next); } catch (e) { /* private mode */ }
  }

  /* ---------------------------------------------------------------------
     Navigation bar, built once and injected into every page.
     --------------------------------------------------------------------- */
  function currentFile() {
    var path = window.location.pathname.split("/").pop();
    return path === "" ? "index.html" : path;
  }

  function buildNav() {
    var mount = document.getElementById("siteNav");
    if (!mount) return;

    var here = currentFile();
    var links = PAGES.map(function (p) {
      var active = p.file === here ? " active" : "";
      var current = p.file === here ? ' aria-current="page"' : "";
      return '<li class="nav-item">' +
             '<a class="nav-link' + active + '" href="' + p.file + '"' + current + '>' +
             p.label + "</a></li>";
    }).join("");

    mount.innerHTML =
      '<nav class="navbar navbar-expand-lg site-nav sticky-top">' +
        '<div class="container">' +
          '<a class="navbar-brand" href="index.html">Local LLM Assistant <span class="text-body-secondary fw-normal">· Docs</span></a>' +
          '<button class="navbar-toggler" type="button" data-bs-toggle="collapse" ' +
            'data-bs-target="#navMenu" aria-controls="navMenu" aria-expanded="false" ' +
            'aria-label="Toggle navigation"><span class="navbar-toggler-icon"></span></button>' +
          '<div class="collapse navbar-collapse" id="navMenu">' +
            '<ul class="navbar-nav ms-auto align-items-lg-center">' + links +
              '<li class="nav-item ms-lg-3">' +
                '<button id="themeToggle" class="btn btn-sm filter-btn" type="button">☾</button>' +
              '</li>' +
            '</ul>' +
          '</div>' +
        '</div>' +
      '</nav>';

    var btn = document.getElementById("themeToggle");
    if (btn) btn.addEventListener("click", toggleTheme);
    applyTheme(document.documentElement.getAttribute("data-bs-theme") || "light");
  }

  /* ---------------------------------------------------------------------
     Contents rail: generated from the page's own h2/h3 headings, so it can
     never drift out of sync with the content.
     --------------------------------------------------------------------- */
  function buildToc() {
    var mount = document.getElementById("tocList");
    if (!mount) return;

    var main = document.getElementById("main");
    if (!main) return;

    var headings = main.querySelectorAll("h2[id], h3[id]");
    if (!headings.length) { mount.closest(".toc")?.remove(); return; }

    var html = "";
    headings.forEach(function (h) {
      var sub = h.tagName === "H3" ? " toc-sub" : "";
      var text = h.getAttribute("data-toc") || h.textContent.trim();
      html += '<a class="nav-link' + sub + '" href="#' + h.id + '">' + text + "</a>";
    });
    mount.innerHTML = html;

    // Bootstrap scrollspy highlights the section currently in view
    if (window.bootstrap && window.bootstrap.ScrollSpy) {
      new window.bootstrap.ScrollSpy(document.body, {
        target: "#tocList",
        rootMargin: "0px 0px -70%"
      });
    }
  }

  /* ---------------------------------------------------------------------
     Filtering: search box + category buttons over .filterable cards.
     Each card carries data-tags and its own searchable text.
     --------------------------------------------------------------------- */
  function initFilters() {
    var search = document.getElementById("filterSearch");
    var buttons = document.querySelectorAll("[data-filter]");
    var cards = document.querySelectorAll(".filterable");
    var empty = document.getElementById("noResults");
    if (!cards.length) return;

    var activeTag = "all";

    function refresh() {
      var q = (search && search.value || "").trim().toLowerCase();
      var shown = 0;

      cards.forEach(function (card) {
        var tags = (card.getAttribute("data-tags") || "").toLowerCase();
        var text = card.textContent.toLowerCase();
        var matchTag = activeTag === "all" || tags.indexOf(activeTag) !== -1;
        var matchText = q === "" || text.indexOf(q) !== -1;
        var show = matchTag && matchText;
        card.classList.toggle("is-hidden", !show);
        if (show) shown++;
      });

      // Hide a section heading when every card beneath it is filtered out
      document.querySelectorAll("[data-group]").forEach(function (group) {
        var visible = group.querySelectorAll(".filterable:not(.is-hidden)").length;
        group.classList.toggle("is-hidden", visible === 0);
      });

      if (empty) empty.classList.toggle("is-hidden", shown !== 0);
    }

    if (search) search.addEventListener("input", refresh);

    buttons.forEach(function (btn) {
      btn.addEventListener("click", function () {
        buttons.forEach(function (b) {
          b.classList.remove("active");
          b.setAttribute("aria-pressed", "false");
        });
        btn.classList.add("active");
        btn.setAttribute("aria-pressed", "true");
        activeTag = btn.getAttribute("data-filter").toLowerCase();
        refresh();
      });
    });

    refresh();
  }

  /* ---------------------------------------------------------------------
     Boot
     --------------------------------------------------------------------- */
  initTheme();
  document.addEventListener("DOMContentLoaded", function () {
    buildNav();
    buildToc();
    initFilters();
  });
})();
