# Handover: dedicated "Review mode" for least-confident cuts

## What to build (one paragraph)

Add an opt-in, full-width **Review mode** to the movie page that turns the existing
"least-confident cuts" list into a fast, keyboard-driven, split-screen triage tool: the
**list of cuts on the left**, the **video clip of the active cut auto-playing on the right**.
You work top-to-bottom; the top (active) row is highlighted and its clip loads + loops on
the right automatically. Accept / reject with one keypress and it advances to the next cut.
The normal movie page stays as it is — Review mode is a button you enter from the
least-confident section (and leave with Esc / a back button).

The goal is to make human time cheap: glance at the clip + thumbnails, hit A or R, move on.

---

## Orientation: how this app is built

- **Everything is one file:** [`label_server.py`](../label_server.py). It's a stdlib
  `ThreadingHTTPServer`; the entire front end is embedded as Python string constants
  `HOME_PAGE` (the movie library + movie detail SPA) and `PAGE` (the older per-session
  labeller at `/label`). **No build step, no framework, no dependencies** — plain HTML/CSS
  and vanilla JS. Match that style exactly; do not introduce a bundler or a JS framework.
- **Run it:** `python3 label_server.py --port 8000` then open `http://localhost:8000`.
- **The movie page** is a client-side view inside `HOME_PAGE`: opening `?v=<video-path>`
  calls `renderMovie(video)`, which fetches `GET /api/movie?v=<path>` and renders the
  end-result diagonal viz, the Scene-fixes section, and the **Least-confident cuts** box
  (which has a `List | Clusters` tab). Review mode should be reachable from that box.
- **Dark theme.** Reuse the existing CSS classes (`.box`, `.muted`, `.fixrow`, `.badge`,
  `.gbtn`/`.gbtn.on.gacc`/`.gbtn.on.grej`, `.tabs`, `.overlay`/`.pbox`, `.cthumb`, `.rng`).

---

## Data model & endpoints you will reuse (do NOT rebuild these)

A **cut** under review is `{time: float(seconds), conf: float, kind: "hard"|"gradual",
gold: "accept"|"reject"|null}`. `time` is the cut's timestamp and the primary key used
everywhere (gold verdicts, clips, frames). `conf` is the detector's native confidence
(hard cuts are `transnet_peak_prob`, gradual are `transnet_gradual_prob`).

- **`GET /api/weak?v=<video>`** → the review list:
  ```json
  { "total": 2423, "ceiling": 0.85,
    "bands": { "lt04": 21, "b0406": 142, "b0608": 187, "gte08": 2073 },
    "rows": [ { "time": 8417.4174, "conf": 0.351, "kind": "gradual", "gold": null }, ... ] }
  ```
  `rows` are the reviewable weak cuts (conf < `ceiling`) sorted **weakest-first**, each
  still in the end result (accepted) OR already carrying a verdict. `bands` are
  confidence-band counts for context. Canonical (imported) movies have no per-cut
  confidence → `total: 0`.
- **`POST /api/gold`** `{video, time, decision}` where `decision ∈ "accept"|"reject"|"clear"`
  → records one verdict; returns `{accept: n, reject: m}`. A **reject** drops the cut from
  the movie's end result; **accept** confirms it. `clear` un-sets. Verdicts persist in
  `results/gold/<slug>.json` (`{accept:[times], reject:[times]}`).
- **`POST /api/gold_batch`** `{video, times:[...], decision}` → same, many at once.
- **`GET /clip?v=&t=&pre=&post=`** → a transcoded **H.264 mp4** of `[t-pre, t+post]`,
  streamed. Works for any source container/codec (incl. mkv/hevc). Accurate two-stage seek,
  so `clip-time == pre` is exactly the cut. Current single-cut watch uses `pre=2, post=0.5`
  (2.5s: 2s lead-in, cut at 2.0s, 0.5s tail). **Transcoding takes ~1-2s** — see "prefetch".
- **`GET /clipvtt?v=&t=&pre=&post=&mark=<t>&only=1`** → a WebVTT subtitle track. With
  `only=1` it emits ONE caption, `"Cut #<ordinal>"`, timed to **appear exactly at the cut**
  (`clip-time = pre`) and hold through the tail. Use this as the `<track>` for the clip so
  the caption lands on the cut.
- **`GET /frameat?v=&t=`** → a single JPEG (~150px wide) decoded at time `t`. Used for the
  before/after thumbnails (`t-0.15` and `t+0.15`). Lazy-load these.
- Clusters (for reference, probably not needed in Review mode):
  `GET /api/clusters?v=&conf=&gap=&min=`.

### Front-end functions already in `HOME_PAGE` you can call or crib from
- `loadWeak()` — fetches `/api/weak`, stores `window._weakall`, renders. Sets the band line.
- `weakSlide(which)` — filters `window._weakall` by the two-handle conf range + mode
  (`_weakMode` is `'undecided'|'decided'`) and renders rows via `cutRow`.
- `cutRow(x)` — renders one list row: before/after `<img class=cthumb>`, timecode, kind
  badge, conf, and **▶ watch / ✓ accept / ✗ reject / fix →** buttons.
- `setGold(t, decision)` — POSTs `/api/gold`, updates `window._weakall`/cluster state, and
  re-renders. A verdict (accept or reject) moves the cut out of "To review" into "Decided".
- `openClip(t, pre, post, label, note, marks, onlyMark)` / `watchCut(t)` — builds the
  overlay `<video>` + `<track>` player. In Review mode you want the video **inline in the
  right pane**, not the modal overlay, but the URL-building logic is the same.
- `fmtTC(seconds)` — seconds → `h:mm:ss` timecode.

---

## The current least-confident section (what already exists)

Inside `renderMovie`, the processed-movie branch renders a `.box` titled "Least-confident
cuts" with a `List | Clusters` tab. The **List** panel has: a bands line; a **To review /
Decided** toggle (`weakMode`); a **two-handle confidence-range slider** (low handle default
0.5 = the accept threshold, high 0.85) that filters which cuts show; and the scrollable list
of `cutRow`s. Each row has thumbnails + watch + accept/reject + fix. Rejecting/accepting
removes the cut from "To review"; "Decided" lists what you've judged so you can revisit.

Review mode is a **new presentation of this same list + the same actions**, optimised for
speed. Keep the List/Clusters box working as-is; add an entry point into Review mode.

---

## Review mode — UX spec

### Entry / exit
- Add a button in the least-confident List panel header, e.g. **"Review mode →"**. Clicking
  it enters Review mode for the current movie's current review queue (respect the conf-range
  the user has set — pass the low/high bounds in).
- Represent it in the URL so it's shareable/reloadable, e.g. `?v=<video>#review`. Handle it
  in the existing `route()` / hash logic.
- **Esc** or a **"← back to movie"** button returns to the normal movie page.

### Layout (split view, full width)
```
┌───────────────────────────────────────────────────────────────────────┐
│  ← back    Reviewing least-confident cuts · Interstellar (2014)         │
│            37 left · conf window 0.50–0.85 · [12 accepted · 9 rejected] │
├───────────────────────────────┬───────────────────────────────────────┤
│  QUEUE (left, scrolls)         │  ACTIVE CUT (right, sticky)           │
│                                │                                       │
│ ▸ 1:23:45  gradual  0.41  ◀    │   ┌─────────────────────────────┐     │
│   1:29:02  hard     0.44       │   │      <video> auto-play,     │     │
│   1:31:10  gradual  0.47       │   │      loop, "Cut #686"       │     │
│   ...                          │   │      caption at the cut     │     │
│ [thumbs on each row]           │   └─────────────────────────────┘     │
│                                │   before | after thumbnails           │
│                                │   1:23:45 · gradual · conf 0.41        │
│                                │   [✓ Accept (A)] [✗ Reject (R)]        │
│                                │   [↻ replay (Space)]  [skip ↓ (J)]     │
└───────────────────────────────┴───────────────────────────────────────┘
```
- **Left pane:** the queue (the `/api/weak` rows in the chosen conf window, weakest-first),
  scrollable, with the **active row highlighted**. Clicking a row makes it active. Keep the
  compact thumbnails on rows if they help; the big review happens on the right.
- **Right pane (sticky):** the **active cut's clip auto-playing and looping**, its
  before/after thumbnails, its metadata (timecode, kind, conf, cut #), and big
  **Accept / Reject** buttons. Show which verdict (if any) it currently has.

### Core interaction
- The **top of the queue is active by default** and its clip loads immediately.
- Moving the active selection (arrow/J-K or click) **auto-loads + autoplays the clip** for
  that cut on the right. Muted autoplay + `loop` (browser autoplay policy needs muted; give
  an unmute control). Use `/clip?...&pre=2&post=0.5` + `/clipvtt?...&mark=<t>&only=1`.
- **Accept / Reject** records the verdict (`POST /api/gold`) and **auto-advances** to the
  next unreviewed cut (so you can rip through the list: look, R, look, A, …). The judged cut
  leaves the queue (it's now "decided").
- Keep the queue and the underlying `window._weakall` in sync with the normal view, so
  verdicts made here show up in "Decided" and update the movie's end-result count.

### Keyboard shortcuts (make these first-class — this is the point)
- `A` accept, `R` reject (both advance), `U`/`Backspace` undo last / clear current verdict,
  `J`/`↓` next, `K`/`↑` previous, `Space` replay the clip, `Esc` exit.
- **Guard shortcuts** so they don't fire while a text input/slider is focused.

### Performance: prefetch the next clips
Transcoding a clip takes ~1-2s, which is too slow to feel snappy if it starts only when a
row becomes active. **Prefetch**: when a cut becomes active, kick off the `/clip` (and
`/clipvtt`, `/frameat`) requests for the **next 1-2 cuts** in the queue (e.g. hidden
`<video preload>` or `fetch` into cache) so they're ready when you advance. The server is
threaded, so concurrent transcodes are fine; keep it to ~2 ahead to avoid a stampede.

### Progress & context
- Header shows: cuts left in the queue, the conf window, and running accepted/rejected
  counts. Optionally a small progress bar. When the queue empties, show a "done — N reviewed"
  state with a way back.

---

## Constraints

- **Edit only `label_server.py`.** Keep it one self-contained file. If you add endpoints,
  they should be small and follow the existing handlers; but you likely need **no new
  endpoints** — everything is already there.
- **Do not break** the existing movie library, movie detail page, the List/Clusters box,
  the two-handle slider, scene fixes, diagonal viz, gold verdicts, or the `/label` labeller.
  Review mode is additive and opt-in.
- Reuse existing endpoints and CSS. Don't duplicate `cutRow`, `setGold`, `fmtTC`,
  `openClip`'s URL logic, etc. — factor/share rather than copy.
- Keep verdict state consistent between Review mode and the normal List/Decided views
  (both read the same `window._weakall` / gold state).

---

## Verification — REQUIRED, do not skip

1. `python3 -c "import ast; ast.parse(open('label_server.py').read())"` after each edit, and
   lint the embedded JS: extract the `HOME_PAGE` `<script>` blocks to a temp file and run
   `node --check`.
2. Run your **own** test server on a **spare port** (e.g. `--port 8011`) so you don't disturb
   an instance on 8000. Kill only your own python process when restarting.
3. Good test movie (processed, ~480 reviewable weak cuts, real gradual/hard mix):
   `data/movies/Interstellar.2014.720p.BluRay.H264.AAC-RARBG/Interstellar.2014.720p.BluRay.H264.AAC-RARBG.mp4`
   (URL-encode it for query params).
4. **Actually look at it** with headless Chrome and iterate until it renders and behaves:
   ```
   rm -rf /tmp/rev_prof; timeout 90 google-chrome --headless=new --no-sandbox --disable-gpu \
     --disable-dev-shm-usage --disable-software-rasterizer --user-data-dir=/tmp/rev_prof \
     --hide-scrollbars --screenshot=/tmp/rev.png --window-size=1400,900 \
     --virtual-time-budget=14000 "http://localhost:8011/?v=<urlencoded>#review"
   ```
   Then read the PNG. Notes: the SPA fetches async, so give it virtual-time; `--headless=new`
   captures full page height (crop with PIL to inspect regions). **Headless Chrome will not
   reliably show `<video>` playback or `<track>` captions in a screenshot** (it catches the
   clip mid-buffer) — verify the clip/caption endpoints with `curl` + `ffprobe` instead, and
   verify the *layout/queue/active-row* visually.
5. Verify the loop end-to-end with curl: accept a cut via `/api/gold`, confirm it leaves
   `/api/weak` `rows` and the movie's `/api/movie` `n_cuts` drops; `clear` restores it. Clean
   up any test verdicts you create (don't leave gold in `results/gold/` — but NOTE the file
   may contain the **user's real verdicts**; only remove verdicts you added).

### Gotchas already hit this project (save yourself the time)
- **Inline SVG via innerHTML:** quote all attributes and self-close with `></tag>`. An
  unquoted `class=x/>` swallows the slash, leaves the tag open, and nothing renders.
- **Responsive SVG:** give it `aspect-ratio` + width, not just `height:auto` (which collapses
  to 0). And don't use `preserveAspectRatio=none` with `height:auto`.
- **Single-threaded server blocks the UI** — the server is already `ThreadingHTTPServer`;
  keep it that way (prefetching several clips relies on it).
- **Clip/caption timing is already correct**: accurate two-stage seek, caption at
  `clip-time == pre` via `only=1`. Reuse `pre=2, post=0.5` unless you have a reason to change.

---

## Open decisions (make a sensible call, note what you chose)

- **Conf window in Review mode:** inherit the two-handle slider's low/high from the List
  panel, or expose the slider again inside Review mode? (Recommend: inherit, and allow
  changing it without leaving.)
- **Undecided only, or include decided?** The queue should be the *undecided* cuts by
  default; consider a toggle to also step through "Decided" to double-check past verdicts.
- **Autoplay audio:** default muted (autoplay policy) with an unmute toggle, or start paused
  with a play button? (Recommend muted autoplay + loop; audio rarely helps judge a cut.)
- **Clip length:** the current 2s+0.5s works; a tighter/looser option per user taste is a
  one-number change.

---

## Deliverable

Working Review mode in `label_server.py`, reachable from the least-confident section, with
the split view + keyboard flow + auto-playing looped clip + prefetch. Leave the rest of the
app untouched and working. Summarise what you built, the entry point, the keyboard map, any
decisions from the list above, and confirm you verified it (layout via screenshot; the
review→verdict→advance loop via curl).
