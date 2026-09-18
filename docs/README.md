# Documentation Website

A static site documenting the Local LLM Scientific Assistant for a non-specialist
audience — written for community college faculty who need to understand what a
RAG system is actually doing before they rely on it or teach with it.

## Viewing it

**Open it directly.** Double-click `index.html`. Everything works from the file
system: navigation, search, filtering and the theme toggle. No build step, no
server, no installation.

**Or serve it locally**, which some browsers prefer:

```bash
cd docs
python3 -m http.server 8000
# then open http://localhost:8000
```

**Or publish it free on GitHub Pages.** This folder is named `docs/` precisely so
that works with no configuration:

> Repository → **Settings** → **Pages** → Source: *Deploy from a branch* →
> Branch: `main`, Folder: `/docs` → **Save**

The site appears at `https://<username>.github.io/<repository>/` within a minute
or two, which is the simplest way to share it with faculty who will not clone a
repository.

## Pages

| File | Covers |
|---|---|
| `index.html` | What the application is, why retrieval matters, the three answer modes |
| `how-it-works.html` | The six pipeline stages, and how to read the diagnostics panel |
| `controls.html` | Every control: range, default, effect, recommendation |
| `techniques.html` | Each advanced technique: problem, mechanism, cost, when to enable |
| `settings.html` | Recommended model settings, three presets, hardware guidance |
| `glossary.html` | Plain-language definitions, plus five points for a handout |

## How it is built

Plain HTML5 and Bootstrap 5.3.3 from a CDN, with one small stylesheet and one
small script. No framework, no bundler, no dependencies to install or update.

```
docs/
├── index.html … glossary.html    six pages
├── assets/css/site.css           a thin layer over Bootstrap
└── assets/js/site.js             navigation, contents rail, filtering, theme
```

`site.js` does four things, all progressive enhancements — the pages remain
readable if it fails to load:

- **Navigation** is injected from a single list, so adding a page means editing
  one array rather than six files.
- **The contents rail** is generated from each page's own `<h2>`/`<h3>` headings,
  so it cannot drift out of sync with the content.
- **Filtering** drives the search box and category buttons on the reference pages.
- **The theme toggle** follows the operating system by default and remembers an
  explicit choice.

Bootstrap is loaded with Subresource Integrity hashes, so a tampered CDN file is
refused rather than executed.

## Keeping it accurate

The control ranges and defaults in `controls.html` were taken from the
application source, not from memory. If you change a default in `app.py` or
`rag_core/config.py`, update the matching card. The values currently documented:

| Control | Range | Default | Defined in |
|---|---|---|---|
| Temperature | 0.0 – 2.0 | 0.3 | `app.py` |
| Top-K | 1 – 100 | 40 | `app.py` |
| Top-P | 0.0 – 1.0 | 0.9 | `app.py` |
| Context Length | 256 – 32,768 | 8,192 | `app.py` |
| Semantic ↔ Keyword balance | 0.0 – 1.0 | 0.5 | `rag_core/config.py` |
| Passages sent to the model | 1 – 15 | 5 | `rag_core/config.py` |
| Candidates considered | up to 40 | 12 | `rag_core/config.py` |
| Candidates per index | 5 – 50 | 20 | `rag_core/config.py` |

## Editing

Each page is self-contained apart from the shared CSS and JS. To add a page,
create the HTML file (copy an existing one for the head, nav mount and footer)
and add an entry to the `PAGES` array at the top of `assets/js/site.js`.

Give every `<h2>` and `<h3>` an `id` — the contents rail is built from them.
Add `data-toc="Shorter label"` to a heading if its full text is too long for the
rail.
