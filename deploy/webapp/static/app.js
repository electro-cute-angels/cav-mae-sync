"use strict";

/* Corpus Audio–Visual Search — front-end
   Talks to the FastAPI backend (server.py). Three columns:
     I  · Query   — direction, strategy, filters, upload
     II · Segments — browse & pick a query segment
     III· Match    — query frame-strip (+ localization heatmap toggle) and
                     the retrieval results grid. */

const state = {
  meta: null,
  direction: "audio2video",     // audio2video | video2audio
  strategy: "diagonal_mean",
  excludeParent: true,
  // segment browser
  filters: { q: "", series: "", media_type: "", silent: "" },
  segItems: [],
  segTotal: 0,
  segOffset: 0,
  segLimit: 60,
  // hierarchical tree browser: series → film → segment (lazy-loaded)
  tree: { expSeries: {}, expFilms: {}, films: {}, segs: {} },
  // selection / results
  querySeg: null,               // provenance object of selected query
  results: null,
  heatOn: false,
  selFrame: 8,
  perRow: 5,
  uploadMode: false,
  thLo: 0.5,                    // active-region playback thresholds (0..1)
  thHi: 1.0,
  activeRegionOpen: false,      // active-region audio panel collapsed by default
  collapsed: { query: false, segments: false, results: false },
};

const $ = (id) => document.getElementById(id);
const fmt = (x, d = 3) =>
  (x === null || x === undefined || Number.isNaN(x)) ? "–" : Number(x).toFixed(d);
const hms = (s) => {
  if (s === null || s === undefined) return "–";
  const m = Math.floor(s / 60), ss = Math.floor(s % 60);
  return `${m}:${String(ss).padStart(2, "0")}`;
};

function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined) continue;
    if (k === "class") n.className = v;
    else if (k === "html") n.innerHTML = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v);
  }
  for (const kid of kids) if (kid !== null && kid !== undefined) n.append(kid);
  return n;
}

const api = (p) => fetch(p).then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); });

/* ---------------- init ---------------- */
async function init() {
  document.querySelectorAll(".col-head").forEach((h) => {
    h.addEventListener("click", (e) => {
      if (e.target.closest(".col-tools")) return;
      const key = h.dataset.toggle;
      state.collapsed[key] = !state.collapsed[key];
      applyCollapsed();
    });
  });
  document.documentElement.style.setProperty("--per-row", state.perRow);

  try {
    state.meta = await api("/api/meta");
  } catch (e) {
    $("body-query").innerHTML = `<p class="placeholder">Backend not reachable. Start server.py.</p>`;
    return;
  }
  $("topmeta").textContent = `${state.meta.n_segments.toLocaleString()} segments`;
  renderQueryControls();
  renderResultsTools();
  loadSegments(true);
  renderResults();
}

function applyCollapsed() {
  for (const key of ["query", "segments", "results"])
    $(`col-${key}`).classList.toggle("collapsed", state.collapsed[key]);
}

/* ---------------- column I · query controls ---------------- */
function renderQueryControls() {
  const body = $("body-query");
  body.innerHTML = "";

  // direction
  const dirBox = el("div", { class: "ctl" },
    el("div", { class: "section-label" }, "direction"),
    el("div", { class: "seg2" },
      el("button", {
        class: state.direction === "audio2video" ? "on" : "",
        onclick: () => { state.direction = "audio2video"; onDirectionChange(); },
      }, "sound → image"),
      el("button", {
        class: state.direction === "video2audio" ? "on" : "",
        onclick: () => { state.direction = "video2audio"; onDirectionChange(); },
      }, "image → sound"),
    ),
  );
  body.append(dirBox);

  // strategy
  const stratSel = el("select", { class: "wide",
    onchange: (e) => { state.strategy = e.target.value; if (state.querySeg) runRetrieval(); } });
  for (const s of state.meta.strategies) {
    const o = el("option", { value: s }, s.replace("_", " "));
    if (s === state.strategy) o.setAttribute("selected", "");
    stratSel.append(o);
  }
  body.append(el("div", { class: "ctl" },
    el("label", { class: "fld" }, el("span", { class: "lbl" }, "strategy"), stratSel),
    el("div", { class: "upnote" },
      "fast = mean-pooled clip cosine. diagonal_* re-ranks on the 16 time-aligned frames (true sync)."),
    el("label", { class: "chk", style: "margin-top:10px",
      onclick: (e) => { if (e.target.tagName !== "INPUT") return; } },
      (() => {
        const c = el("input", { type: "checkbox" });
        c.checked = state.excludeParent;
        c.addEventListener("change", () => { state.excludeParent = c.checked; if (state.querySeg) runRetrieval(); });
        return c;
      })(),
      "exclude same source film"),
  ));

  // filters
  const seriesSel = el("select", { class: "wide",
    onchange: (e) => { state.filters.series = e.target.value; loadSegments(true); } },
    el("option", { value: "" }, "all series"));
  for (const s of state.meta.series) {
    const o = el("option", { value: s.id }, `${s.id} · ${s.title} (${s.count})`);
    if (s.id === state.filters.series) o.setAttribute("selected", "");
    seriesSel.append(o);
  }
  const typeSel = el("select", { class: "wide",
    onchange: (e) => { state.filters.media_type = e.target.value; loadSegments(true); } },
    el("option", { value: "" }, "all types"),
    el("option", { value: "cartoon" }, "cartoon"),
    el("option", { value: "film" }, "film"));
  const silentSel = el("select", { class: "wide",
    onchange: (e) => { state.filters.silent = e.target.value; loadSegments(true); } },
    el("option", { value: "" }, "sound + silent"),
    el("option", { value: "no" }, "has sound"),
    el("option", { value: "yes" }, "silent only"));
  body.append(el("div", { class: "ctl" },
    el("div", { class: "section-label" }, "filter segments"),
    el("label", { class: "fld" }, el("span", { class: "lbl" }, "series"), seriesSel),
    el("label", { class: "fld" }, el("span", { class: "lbl" }, "media type"), typeSel),
    el("label", { class: "fld" }, el("span", { class: "lbl" }, "audio"), silentSel),
  ));

  // upload
  body.append(uploadCtl());
}

function onDirectionChange() {
  renderQueryControls();
  if (state.querySeg) runRetrieval();
}

function uploadCtl() {
  const wrap = el("div", { class: "ctl" }, el("div", { class: "section-label" }, "or upload your own"));
  const kind = () => state.direction === "audio2video" ? "audio" : "image";
  const note = el("div", { class: "upnote" });
  const setNote = () => note.textContent = state.direction === "audio2video"
    ? "drop a SOUND (wav) → find frames that match it"
    : "drop an IMAGE → find sounds that match it";
  setNote();

  const input = el("input", { type: "file", style: "display:none",
    accept: kind() === "audio" ? "audio/*,.wav" : "image/*" });
  const drop = el("div", { class: "drop", onclick: () => input.click() },
    kind() === "audio" ? "drop / choose a .wav" : "drop / choose an image");
  input.addEventListener("change", () => { if (input.files[0]) uploadQuery(input.files[0], kind()); });
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.style.background = "#fff"; });
  drop.addEventListener("dragleave", () => { drop.style.background = ""; });
  drop.addEventListener("drop", (e) => {
    e.preventDefault(); drop.style.background = "";
    if (e.dataTransfer.files[0]) uploadQuery(e.dataTransfer.files[0], kind());
  });
  wrap.append(drop, input, note);
  return wrap;
}

/* ---------------- column II · segments ---------------- */
function renderSegmentsShell() {
  const body = $("body-segments");
  body.innerHTML = "";
  const search = el("input", { placeholder: "search title / id…", value: state.filters.q });
  let t;
  search.addEventListener("input", (e) => {
    clearTimeout(t);
    t = setTimeout(() => { state.filters.q = e.target.value; loadSegments(true); }, 220);
  });
  body.append(el("div", { class: "filterbar" }, search));

  if (state.filters.q) {                          // flat search results
    const table = el("div", { class: "seg-table", id: "seg-table" });
    table.append(el("div", { class: "seg-hrow" },
      el("span", {}, "title / segment"),
      el("span", { class: "num" }, "year"),
      el("span", { class: "num" }, "start"),
      el("span", {}, ""),
    ));
    body.append(table);
  } else {                                        // hierarchical tree
    body.append(el("div", { class: "seg-tree", id: "seg-tree" }));
  }
}

async function loadSegments(reset) {
  if (reset) {
    state.segOffset = 0; state.segItems = [];
    state.tree = { expSeries: {}, expFilms: {}, films: {}, segs: {} };
    renderSegmentsShell();
  }
  if (state.filters.q) return loadFlat();
  renderTree();
}

async function loadFlat() {
  const f = state.filters;
  const qs = new URLSearchParams({
    q: f.q, series: f.series, media_type: f.media_type, silent: f.silent,
    offset: state.segOffset, limit: state.segLimit,
  });
  const data = await api("/api/segments?" + qs.toString());
  state.segTotal = data.total;
  state.segItems.push(...data.items);
  renderSegRows(data.items);
}

function renderSegRows(items) {
  const table = $("seg-table");
  if (!table) return;
  table.querySelectorAll(".loadmore").forEach((b) => b.remove());
  for (const it of items) table.append(segRow(it, false));
  if (state.segItems.length < state.segTotal) {
    table.append(el("button", {
      class: "loadmore",
      onclick: () => { state.segOffset += state.segLimit; loadFlat(); },
    }, `load more — ${state.segItems.length} / ${state.segTotal}`));
  }
}

function segRow(it, nested) {
  const active = state.querySeg && state.querySeg.video_id === it.seg_id;
  return el("div", {
    class: "seg-row" + (nested ? " nested" : "") + (active ? " active" : ""),
    onclick: () => selectSegment(it.seg_id),
  },
    el("span", {},
      el("span", { class: "nm" }, el("span", { class: active ? "hl" : "" },
        nested ? `segment ${(it.segment_index ?? 0)}` : (it.title || it.seg_id))),
      el("span", { class: "sid" }, nested ? it.seg_id : `${it.series_id} · ${it.seg_id}`)),
    el("span", { class: "num" }, it.year ?? "–"),
    el("span", { class: "num" }, hms(it.start_sec)),
    el("span", { class: "tag" + (it.silent_source ? " silent" : "") },
      it.silent_source ? "sil" : (it.media_type === "film" ? "film" : "cart")),
  );
}

/* -------- tree: series → film → segment -------- */
function renderTree() {
  const root = $("seg-tree");
  if (!root) return;
  root.innerHTML = "";
  const seriesList = state.meta.series.filter(
    (s) => !state.filters.series || s.id === state.filters.series);

  for (const s of seriesList) {
    const openS = !!state.tree.expSeries[s.id];
    root.append(el("div", {
      class: "tree-series" + (openS ? " open" : ""),
      onclick: () => toggleSeries(s.id),
    },
      el("span", { class: "tw" }, openS ? "▾" : "▸"),
      el("span", { class: "tt" }, s.title),
      el("span", { class: "tc" }, `${s.id} · ${s.count.toLocaleString()}`)));

    if (!openS) continue;
    const films = state.tree.films[s.id];
    if (films === "loading" || films === undefined) {
      root.append(el("div", { class: "tree-loading" }, "loading films…"));
      continue;
    }
    for (const f of films) {
      const openF = !!state.tree.expFilms[f.parent_id];
      root.append(el("div", {
        class: "tree-film" + (openF ? " open" : ""),
        onclick: () => toggleFilm(s.id, f.parent_id),
      },
        el("span", { class: "tw" }, openF ? "▾" : "▸"),
        el("span", { class: "tt" }, f.title || f.parent_id),
        el("span", { class: "tc" },
          `${f.year ?? "?"} · ${f.count} · ${f.media_type === "film" ? "film" : "cart"}`)));

      if (!openF) continue;
      const segs = state.tree.segs[f.parent_id];
      if (segs === "loading" || segs === undefined) {
        root.append(el("div", { class: "tree-loading seg" }, "loading segments…"));
        continue;
      }
      for (const it of segs) root.append(segRow(it, true));
    }
  }
}

async function toggleSeries(id) {
  const open = !state.tree.expSeries[id];
  state.tree.expSeries[id] = open;
  if (open && state.tree.films[id] === undefined) {
    state.tree.films[id] = "loading";
    renderTree();
    const f = state.filters;
    const qs = new URLSearchParams({ series: id, media_type: f.media_type, silent: f.silent });
    try {
      const data = await api("/api/films?" + qs.toString());
      state.tree.films[id] = data.films;
    } catch (e) { state.tree.films[id] = []; }
  }
  renderTree();
}

async function toggleFilm(seriesId, pid) {
  const open = !state.tree.expFilms[pid];
  state.tree.expFilms[pid] = open;
  if (open && state.tree.segs[pid] === undefined) {
    state.tree.segs[pid] = "loading";
    renderTree();
    const f = state.filters;
    const qs = new URLSearchParams({ parent: pid, media_type: f.media_type, silent: f.silent });
    try {
      const data = await api("/api/segments?" + qs.toString());
      state.tree.segs[pid] = data.items;
    } catch (e) { state.tree.segs[pid] = []; }
  }
  renderTree();
}

/* ---------------- selection + retrieval ---------------- */
async function selectSegment(segId) {
  state.uploadMode = false;
  state.querySeg = await api("/api/segment/" + encodeURIComponent(segId));
  state.selFrame = 8;
  // refresh active highlight in the list
  if (state.filters.q) {
    document.querySelectorAll(".seg-row").forEach((r) => r.classList.remove("active"));
  } else if ($("seg-tree")) {
    renderTree();
  }
  renderCrumbs();
  runRetrieval();
}

async function runRetrieval() {
  if (!state.querySeg) return;
  renderResults(true);
  const qs = new URLSearchParams({
    seg_id: state.querySeg.video_id,
    direction: state.direction,
    strategy: state.strategy,
    top_k: 48,
    exclude_parent: state.excludeParent,
  });
  try {
    state.results = await api("/api/retrieve?" + qs.toString());
  } catch (e) {
    state.results = { error: String(e) };
  }
  renderResults();
}

async function uploadQuery(file, kind) {
  state.uploadMode = true;
  state.querySeg = null;
  renderResults(true, `encoding uploaded ${kind}…`);
  const fd = new FormData();
  fd.append("kind", kind);
  fd.append("strategy", "fast");
  fd.append("top_k", "48");
  fd.append("file", file);
  try {
    const r = await fetch("/api/retrieve_upload", { method: "POST", body: fd });
    if (!r.ok) throw new Error(await r.text());
    state.results = await r.json();
    state.results.upload = { kind, name: file.name };
  } catch (e) {
    state.results = { error: String(e) };
  }
  renderCrumbs();
  renderResults();
}

/* ---------------- column III · results tools ---------------- */
function renderResultsTools() {
  const tools = $("results-tools");
  tools.innerHTML = "";
  const cnt = el("span", { class: "perrow-val" }, String(state.perRow));
  const range = el("input", { type: "range", min: "2", max: "8", step: "1",
    value: String(state.perRow), class: "perrow-range" });
  range.addEventListener("input", (e) => {
    state.perRow = parseInt(e.target.value, 10);
    document.documentElement.style.setProperty("--per-row", state.perRow);
    cnt.textContent = state.perRow;
  });
  const sw = el("span", {
    class: "switch" + (state.heatOn ? " on" : ""),
    onclick: () => {
      state.heatOn = !state.heatOn;
      document.body.classList.toggle("heat-on", state.heatOn);
      sw.classList.toggle("on", state.heatOn);
      if (state.heatOn && state.querySeg) loadHeatmaps();
    },
  }, el("span", { class: "track" }), el("span", { class: "lbl" }, "activation maps"));
  tools.append(el("span", { class: "perrow" }, range, cnt), sw);
}

/* ---------------- column III · render ---------------- */
function renderResults(busy, busyMsg) {
  const body = $("body-results");
  body.innerHTML = "";
  if (busy) {
    body.append(queryBlock());
    body.append(el("div", { class: "busy" }, busyMsg || "searching the corpus…"));
    return;
  }
  if (!state.querySeg && !state.uploadMode) {
    body.append(el("p", { class: "placeholder" },
      "select a segment in column II, or upload a sound / image in column I."));
    return;
  }
  body.append(queryBlock());

  if (!state.results) return;
  if (state.results.error) {
    body.append(el("div", { class: "busy" }, "error: " + state.results.error));
    return;
  }

  const res = state.results.results || [];
  const targetLabel = (state.results.direction === "audio2video") ? "matching frames" : "matching sounds";
  const table = el("div", { class: "match-table" });
  table.append(el("div", { class: "sec-row" },
    el("span", {}, `results — ${targetLabel}`),
    el("span", { class: "sec-note" },
      `${res.length} shown · strategy ${state.results.strategy}`)));
  const grid = el("div", { class: "grid-cards" });
  res.forEach((r, i) => grid.append(resultCard(r, i, state.results.direction)));
  table.append(grid);
  body.append(table);
}

function queryBlock() {
  const wrap = el("div", {});
  wrap.append(el("div", { class: "cluster-head" },
    el("h2", {}, el("span", {}, "query "),
      el("span", { class: "hl" }, state.uploadMode
        ? `uploaded ${state.results?.upload?.kind || ""}`
        : (state.querySeg ? (state.querySeg.title || state.querySeg.video_id) : "…")))));

  if (state.uploadMode) {
    const u = state.results?.upload;
    if (u) {
      const t = el("div", { class: "match-table" });
      t.append(el("div", { class: "meta-row" },
        el("span", { class: "mk" }, "file"), el("span", { class: "mv" }, u.name)));
      t.append(el("div", { class: "meta-row" },
        el("span", { class: "mk" }, "direction"),
        el("span", { class: "mv" }, u.kind === "image" ? "image → sound" : "sound → image")));
      wrap.append(t);
    }
    return wrap;
  }

  const p = state.querySeg;
  if (!p) return wrap;
  const t = el("div", { class: "match-table" });
  const meta = (k, v) => t.append(el("div", { class: "meta-row" },
    el("span", { class: "mk" }, k), el("span", { class: "mv" }, v)));
  meta("series", `${p.series_id} · ${p.series_title}`);
  meta("source", `${p.parent_filename || "–"}`);
  meta("time", `${hms(p.start_sec)} – ${hms(p.end_sec)} (${p.year ?? "?"})`);
  const audible = (p.audible !== undefined) ? p.audible : !p.silent_source;
  meta("audio", p.silent_source ? "silent (synthesised)"
      : (audible ? "has sound" : "no audible track (silent-era source)"));

  // frame strip
  t.append(el("div", { class: "sec-row" },
    el("span", {}, "query frames · 10 s"),
    el("span", { class: "sec-note" },
      state.heatOn ? "heatmap = where the sound localises" : "toggle spatial activation map ↗")));
  const strip = el("div", { class: "strip" });
  for (let fi = 0; fi < 16; fi++) {
    const fr = el("div", {
      class: "fr" + (fi === state.selFrame ? " sel" : ""),
      onclick: () => { state.selFrame = fi; renderResults(); },
    },
      el("img", { src: `/media/frame/${p.video_id}/${fi}`, loading: "lazy" }),
      el("img", { class: "heat", "data-fi": fi, loading: "lazy" }),
      el("span", { class: "fi" }, fi));
    strip.append(fr);
  }
  t.append(strip);

  // spectrogram + audio activation map (mirrors the visual heatmap)
  if (audible) {
    t.append(el("div", { class: "sec-row" },
      el("span", {}, `query sound · spectrogram · 10 s`),
      el("span", { class: "sec-note" },
        state.heatOn ? "heatmap = which sound (freq × time) grounds to the selected frame"
                     : "box = the 4 s window frame " + state.selFrame + " encodes · play to follow ↗")));
    const win = frameWindow(state.selFrame);
    const leftPct = win.start / win.len * 100;
    const widthPct = (win.end - win.start) / win.len * 100;
    const spectro = el("div", { class: "spectro" },
      el("img", { class: "spec-base", src: `/api/spectrogram_full/${p.video_id}`, loading: "lazy" }),
      el("img", { class: "spec-heat", "data-fi": state.selFrame,
        style: `left:${leftPct}%;width:${widthPct}%`,
        src: state.heatOn ? `/api/localize_audio/${p.video_id}/${state.selFrame}?alpha=0.55` : "",
        loading: "lazy" }),
      el("div", { class: "spec-window", style: `left:${leftPct}%;width:${widthPct}%` }),
      el("div", { class: "spec-playhead", id: "spec-playhead" }),
      el("span", { class: "spec-ax spec-ax-f" }, "freq"),
      el("span", { class: "spec-ax spec-ax-t" }, "time →"));
    t.append(spectro);
  }

  // audio player (drives playback-sync of frame strip + spectrogram)
  if (audible) {
    const audio = el("audio", { src: `/media/audio/${p.video_id}`, preload: "metadata" });
    audio.addEventListener("timeupdate", () => onAudioTime(audio, p));
    audio.addEventListener("play", () => document.body.classList.add("playing"));
    audio.addEventListener("pause", () => document.body.classList.remove("playing"));
    audio.addEventListener("ended", () => document.body.classList.remove("playing"));
    t.append(audioPlayer(audio));

    // third player: hear only the activation-gated part of this frame's sound
    t.append(activeRegionPlayer(p));
  }
  wrap.append(t);

  if (state.heatOn) setTimeout(loadHeatmaps, 0);
  return wrap;
}

function loadHeatmaps() {
  if (!state.querySeg) return;
  const p = state.querySeg;
  document.querySelectorAll(".strip .heat").forEach((img) => {
    const fi = img.getAttribute("data-fi");
    img.src = `/api/localize/${p.video_id}/${fi}?alpha=0.55`;
  });
  // spectrogram audio-activation map for the selected frame's 4 s window
  const heat = document.querySelector(".spectro .spec-heat");
  if (heat) heat.src = `/api/localize_audio/${p.video_id}/${state.selFrame}?alpha=0.55`;
}

/* map a frame index to its 4 s spectrogram window (mirrors the dataloader).
   returns {start, end, len} in fbank-frame units (len = 1024 ≈ 10.24 s). */
function frameWindow(i, len = 1024, target = 416, numFrames = 16) {
  const pos = Math.round(i * len / numFrames);
  let start = Math.max(0, pos - Math.floor(target / 2));
  let end = start + target;
  if (end > len) { end = len; start = Math.max(0, end - target); }
  return { start, end, len };
}

/* move the 4 s window box + heatmap overlay to frame `fi` */
function moveSpecWindow(p, fi) {
  const win = frameWindow(fi);
  const leftPct = win.start / win.len * 100;
  const widthPct = (win.end - win.start) / win.len * 100;
  const box = document.querySelector(".spectro .spec-window");
  const heat = document.querySelector(".spectro .spec-heat");
  if (box) { box.style.left = leftPct + "%"; box.style.width = widthPct + "%"; }
  if (heat) {
    heat.style.left = leftPct + "%"; heat.style.width = widthPct + "%";
    heat.setAttribute("data-fi", fi);
    if (state.heatOn) heat.src = `/api/localize_audio/${p.video_id}/${fi}?alpha=0.55`;
  }
}

/* playback sync: move the spectrogram playhead + follow the current frame */
function onAudioTime(audio, p) {
  const t = audio.currentTime;
  const specDur = 1024 * 0.01;                 // spectrogram time axis ≈ 10.24 s
  const xFrac = Math.max(0, Math.min(1, t / specDur));
  const ph = document.getElementById("spec-playhead");
  if (ph) ph.style.left = (xFrac * 100) + "%";

  const fi = Math.max(0, Math.min(15, Math.round(t / 0.64)));   // frame i at i·0.64 s
  if (fi !== state.selFrame) {
    state.selFrame = fi;
    document.querySelectorAll(".strip .fr").forEach((elm, idx) =>
      elm.classList.toggle("sel", idx === fi));
    moveSpecWindow(p, fi);
  }
}

/* custom minimal audio player matching the archival aesthetic.
   takes an <audio> element (with its listeners already wired) and wraps it in a
   flat play/pause button + scrub bar + time readout. */
function audioPlayer(audio) {
  const btn = el("button", { class: "ap-btn", title: "play / pause" }, "▶");
  const bar = el("div", { class: "ap-bar" }, el("div", { class: "ap-fill" }));
  const fill = bar.firstChild;
  const time = el("span", { class: "ap-time" }, "0:00");

  btn.addEventListener("click", () => { audio.paused ? audio.play() : audio.pause(); });
  audio.addEventListener("play", () => btn.textContent = "❚❚");
  audio.addEventListener("pause", () => btn.textContent = "▶");
  audio.addEventListener("ended", () => btn.textContent = "▶");
  audio.addEventListener("timeupdate", () => {
    const d = audio.duration || 0;
    fill.style.width = (d ? (audio.currentTime / d * 100) : 0) + "%";
    time.textContent = hms(audio.currentTime);
  });
  const seek = (e) => {
    const r = bar.getBoundingClientRect();
    const f = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
    if (audio.duration) audio.currentTime = f * audio.duration;
  };
  bar.addEventListener("click", seek);

  return el("div", { class: "qaudio" }, el("div", { class: "ap" }, btn, bar, time), audio);
}

/* third player: hear only the sound whose image-grounding activation is in
   [thLo, thHi]. A dual-threshold slider over the jet scale (blue 0 → red 1). */
function activeRegionPlayer(p) {
  const wrap = el("div", {});
  const box = el("div", { class: "threshplay" });
  const caret = el("span", { class: "sec-caret" }, state.activeRegionOpen ? "▾" : "▸");
  const header = el("div", { class: "sec-row sec-toggle",
    onclick: () => {
      state.activeRegionOpen = !state.activeRegionOpen;
      caret.textContent = state.activeRegionOpen ? "▾" : "▸";
      box.classList.toggle("collapsed", !state.activeRegionOpen);
    } },
    el("span", {}, caret, ` active-region audio · frame ${state.selFrame}`),
    el("span", { class: "sec-note" }, "bright = will play · dim = removed"));
  wrap.append(header);

  const audio = el("audio", { preload: "none" });
  const lab = el("span", { class: "thlab" });
  const setLab = () => lab.textContent = `${state.thLo.toFixed(2)} – ${state.thHi.toFixed(2)}`;

  // live preview: the 4 s spectrogram masked to the current threshold band
  const preview = el("img", { class: "thspec-img" });
  const refreshPreview = () => {
    preview.src = `/api/filter_spectrogram/${p.video_id}/${state.selFrame}`
      + `?lo=${state.thLo}&hi=${state.thHi}`;
  };
  let t;
  const debouncedPreview = () => { clearTimeout(t); t = setTimeout(refreshPreview, 60); };

  const loI = el("input", { type: "range", min: "0", max: "1", step: "0.02",
    value: String(state.thLo), class: "thrange" });
  const hiI = el("input", { type: "range", min: "0", max: "1", step: "0.02",
    value: String(state.thHi), class: "thrange" });
  loI.addEventListener("input", (e) => {
    state.thLo = Math.min(parseFloat(e.target.value), state.thHi);
    loI.value = state.thLo; setLab(); debouncedPreview();
  });
  hiI.addEventListener("input", (e) => {
    state.thHi = Math.max(parseFloat(e.target.value), state.thLo);
    hiI.value = state.thHi; setLab(); debouncedPreview();
  });

  const play = el("button", { class: "thbtn", onclick: () => {
    audio.src = `/api/filter_audio/${p.video_id}/${state.selFrame}?lo=${state.thLo}&hi=${state.thHi}`;
    audio.play();
  } }, "▶ play active region");
  setLab();
  refreshPreview();

  box.classList.toggle("collapsed", !state.activeRegionOpen);
  box.append(
    el("div", { class: "thspec" }, preview,
      el("span", { class: "spec-ax spec-ax-f" }, "freq"),
      el("span", { class: "spec-ax spec-ax-t" }, "time →")),
    el("div", { class: "thgrad" }),
    el("label", { class: "throw" }, el("span", { class: "thk" }, "min"), loI),
    el("label", { class: "throw" }, el("span", { class: "thk" }, "max"), hiI),
    el("div", { class: "thctl" }, play, lab),
    audio);
  wrap.append(box);
  return wrap;
}

function resultCard(r, i, direction) {
  const showFrame = (direction === "audio2video");  // matching frames -> show frame
  const rep = r.seg_id;
  const img = el("img", { src: `/media/frame/${rep}/8`, loading: "lazy" });
  const imgwrap = el("div", { class: "imgwrap" }, img,
    el("span", { class: "score" }, fmt(r.score)),
    el("span", { class: "rank" }, "#" + (i + 1)));
  if (!showFrame) {
    // matching sounds -> add a play button
    const audio = el("audio", { src: `/media/audio/${rep}`, preload: "none" });
    const btn = el("button", { class: "play",
      onclick: (e) => { e.stopPropagation(); audio.paused ? audio.play() : audio.pause(); } }, "▶");
    imgwrap.append(btn, audio);
  }
  const card = el("div", {
    class: "card" + (r.silent_source ? " silent" : ""),
    onclick: () => selectSegment(r.seg_id),
    title: "click to make this the query",
  }, imgwrap,
    el("div", { class: "cbody" },
      el("div", { class: "cap" }, r.title || r.seg_id),
      el("div", { class: "det" },
        `${r.series_id} · ${r.year ?? "?"} · ${hms(r.start_sec)}`)));
  return card;
}

/* ---------------- crumbs ---------------- */
function renderCrumbs() {
  const c = $("crumbs");
  if (state.uploadMode) { c.textContent = "upload query"; return; }
  const p = state.querySeg;
  c.textContent = p ? `${p.series_id} / ${p.title || p.video_id}` : "";
}

init();
