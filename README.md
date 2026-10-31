# onebom.github.io

Personal website built with Jekyll and al-folio.

## Edit content

- `_pages/about.md`: introduction
- `_pages/projects.md`: projects
- `_pages/cv.md`: CV
- `_bibliography/papers.bib`: publications
- `_data/socials.yml`: contact links
- `_config.yml`: site settings

## Preview

Use the Ruby 3.4.5 version pinned in `.ruby-version` (also used in CI).
With rbenv already installed:

```sh
export PATH="$HOME/.rbenv/shims:$PATH"
ruby -v
bundle install
bundle exec jekyll serve --livereload
```

Open <http://localhost:4000/>. `_pages/about.md` has `permalink: /`, so this is the About page.

If Bundler 4.0.6 cannot be found and `which ruby` shows `/usr/bin/ruby`, the shell
is using macOS system Ruby instead of rbenv. Set the PATH above in your terminal
(or initialize rbenv in your shell startup configuration).

If LiveReload reports `no acceptor (port is in use or requires root privileges)`,
check for an existing server:

```sh
lsof -nP -iTCP:4000 -sTCP:LISTEN
lsof -nP -iTCP:35729 -sTCP:LISTEN
```

Stop the previous server with Ctrl-C in its terminal, or use separate ports:

```sh
bundle exec jekyll serve --livereload --port 4001 --livereload-port 35730
```

Then open <http://localhost:4001/>. Restart Jekyll after changing `_config.yml`.

The agent configuration directories `.agents/`, `.claude/`, and `.codex/` are
excluded from Jekyll watching because their shared skill symlinks otherwise
produce `directory is already being watched` errors.

## Project posts linked from About

Keep posts at `_project/YYMMDD/PostName.md`. `_config.yml` includes `_project`,
so nested Markdown files with front matter are built as pages. For example:

```markdown
---
layout: default
title: "My project"
permalink: /261031/MyProject/
nav: false
---

# My project

Project content here.
```

Add its entry under `publications` or `projects` in `_data/about.yml`:

```yaml
- title: "My project"
  project_path: _project/261031/MyProject.md
  description: "A short description."
  link: "[Code](https://github.com/onebom)"
```

About resolves `project_path` with Jekyll's `link` tag, so the title, publication
thumbnail, and generated **Project page** link use the post's permalink on both
localhost and GitHub Pages. Keep other links in `link`; there is no need to repeat
the project URL there. Paths and filenames are case-sensitive. A missing target
fails the build instead of silently producing a broken link. Adding a file alone
does not add an About entry; add its metadata to `_data/about.yml` as above.

## Deployment

GitHub Actions builds the site and deploys to GitHub Pages on pushes to main. Set the repository Pages source to GitHub Actions.

Theme source: https://github.com/alshedivat/al-folio
