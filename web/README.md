# OpenBonsai website

A small, static landing page for the OpenBonsai research preview. It has no JavaScript, package dependencies, external fonts, or build step.

The page states the project's status (research preview), the four-stage method with links to the docs on GitHub, and the non-affiliation line: "Independent project, not affiliated with or endorsed by Prism ML. Bonsai is Prism ML's product." Keep these in step with the repository README when the method or the results change.

## Preview

From the repository root:

```bash
python3 -m http.server 8080 --bind 127.0.0.1 --directory web
```

Open <http://127.0.0.1:8080>. The page can also be opened directly as `web/index.html`.

## Files

- `index.html`: page copy (introduction, "How it works", the background video), metadata, and links to GitHub and the docs.
- `styles.css`: responsive layout, type, the research-preview label, and keyboard focus styles.
- `assets/openbonsai-mark.svg`: original circular bonsai emblem, also used as the favicon.
- `assets/openbonsai-explainer-en.mp4`: four-minute English video, 1080p H.264/AAC (approximately 8.9 MB), shown as "Background: how ScaleQ works". It explains ScaleQ and CAT-Q, which inspired the project; CAT-Q is not part of the current pipeline.
- `assets/openbonsai-explainer-poster.jpg`: video cover shown before playback.
- `assets/openbonsai-explainer-transcript.txt`: accessible text version with chapter timestamps and the paper reference.

The native video player supports inline playback on mobile and loads the MP4 only when the visitor starts it (`preload="none"`). English captions are already burned into the video, so no duplicate subtitle track is enabled. The transcript is available below the player. All media is hosted with the page; no external video service is required.

The visible pot and compact canopy distinguish the tree as a miniature bonsai. The logo is self-contained vector artwork on a transparent background.

Publish the contents of `web/` with any static host. The GitHub link respects the repository's existing access permissions; this page does not change its visibility.
