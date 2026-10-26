# OpenBonsai website

A small, static research landing page. The site has no JavaScript, package dependencies, external fonts, or build step.

## Preview

From the repository root:

```bash
python3 -m http.server 8080 --bind 127.0.0.1 --directory web
```

Open <http://127.0.0.1:8080>. The page can also be opened directly as `web/index.html`.

## Files

- `index.html`: page copy, metadata, and GitHub link.
- `styles.css`: responsive layout, type, and keyboard focus styles.
- `assets/openbonsai-mark.svg`: original circular bonsai emblem, also used as the favicon.

The visible pot and compact canopy distinguish the tree as a miniature bonsai. The logo is self-contained vector artwork on a transparent background.

Publish the contents of `web/` with any static host. The GitHub link respects the repository's existing access permissions; this page does not change its visibility.
