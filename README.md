# Spyre planning reports

Published with GitHub Pages from the `gh-pages` branch. `.nojekyll` is set, so files are served as-is
(Markdown is not rendered; pages must be HTML).

## Layout

```
index.html              top-level index: one card per effort
<effort-slug>/
  index.html            effort index: one card per page, links back to ../index.html
  *.html                the effort's reports (self-contained; relative links only)
  <subdir>/             optional supporting files, e.g. Markdown drafts (served raw; link to them on GitHub)
```

## Adding an effort

1. Create `<effort-slug>/` with a short kebab-case name (e.g. `model-runner-v2`).
2. Add `<effort-slug>/index.html`, copying the style block from `model-runner-v2/index.html`, with a
   `../index.html` back-link and one card per report.
3. Add a card for the effort to the top-level `index.html`, newest first.
4. Keep pages self-contained (inline CSS/JS) and theme-aware: tokens on `:root`, redefined under
   `prefers-color-scheme: dark` and `[data-theme="dark"]`.
5. Only touch your own effort's directory, plus your one card in the top-level `index.html`.

Never move or rename a published page without leaving a redirect stub at the old path; links to
these pages live in GitHub issues and PRs.
