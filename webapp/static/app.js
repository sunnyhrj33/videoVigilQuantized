/* VideoVigil dashboard – single page, hash routes:
   #/            upload + past analyses
   #/job/<id>    one analysis: video, activity, people gallery
   #/people      known people, watch list, enrolled staff
   #/help        how it works                                                    */
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const view = $("#view");

const STATUS = {
  KNOWN:   { label: "Recognised",      help: "Matches someone the system knows." },
  UNKNOWN: { label: "Unrecognised",    help: "Doesn't match anyone the system knows." },
  SUSPECT: { label: "Watch list",      help: "Matches a person flagged earlier for moving a bag or laptop." },
  UNCLEAR: { label: "Not enough view", help: "Never seen clearly enough to decide who it is." },
};
const JOB_STATUS = { queued: "Waiting", running: "Analysing", done: "Ready", failed: "Failed" };

const ICON = {
  upload: '<svg viewBox="0 0 24 24"><path d="M12 16V4M6 10l6-6 6 6M4 20h16"/></svg>',
  film: '<svg viewBox="0 0 24 24"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 4v16M17 4v16M3 9h4M3 15h4M17 9h4M17 15h4"/></svg>',
  back: '<svg viewBox="0 0 24 24"><path d="M15 18l-6-6 6-6"/></svg>',
  trash: '<svg viewBox="0 0 24 24"><path d="M4 7h16M10 11v6M14 11v6M5 7l1 13h12l1-13M9 7V4h6v3"/></svg>',
  play: '<svg viewBox="0 0 24 24"><path d="M7 5l12 7-12 7z"/></svg>',
  bag: '<svg viewBox="0 0 24 24"><rect x="4" y="7" width="16" height="13" rx="2"/><path d="M9 7V5a3 3 0 0 1 6 0v2"/></svg>',
  person: '<svg viewBox="0 0 24 24"><circle cx="12" cy="8" r="4"/><path d="M4 21c0-4 4-6 8-6s8 2 8 6"/></svg>',
};

// ── helpers ──────────────────────────────────────────────────────────────────

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function clock(sec) {
  sec = Math.max(0, Math.round(sec || 0));
  const h = Math.floor(sec / 3600), m = Math.floor(sec / 60) % 60, s = sec % 60;
  return (h ? h + ":" + String(m).padStart(2, "0") : m) + ":" + String(s).padStart(2, "0");
}

function duration(sec) {
  sec = Math.round(sec || 0);
  if (sec < 60) return sec + " s";
  const m = Math.floor(sec / 60), s = sec % 60;
  return m + " min" + (s ? " " + s + " s" : "");
}

function when(iso) {
  const d = new Date(iso);
  return isNaN(d) ? "" : d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

// "2023-06-02" → "2 Jun 2023" (calendar day, no time-zone shift)
function day(iso) {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso || "");
  if (!m) return iso || "";
  return new Date(+m[1], m[2] - 1, +m[3]).toLocaleDateString(undefined, { dateStyle: "medium" });
}

function todayIso() {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

// First real YYYY-MM-DD in a filename – only ever offered as a suggestion.
function dateInName(name) {
  for (const m of String(name || "").matchAll(/(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)/g)) {
    const d = new Date(+m[1], m[2] - 1, +m[3]);
    if (d.getFullYear() === +m[1] && d.getMonth() === m[2] - 1 && d.getDate() === +m[3]) return m[0];
  }
  return null;
}

let galleryDays = null;   // [{date, staff, usable, problem}] from /api/galleries
async function loadGalleryDays(force = false) {
  if (galleryDays && !force) return galleryDays;
  try { galleryDays = (await api("/api/galleries")).days; } catch { galleryDays = galleryDays || []; }
  return galleryDays;
}

function sureWord(p) {
  return p >= 0.85 ? "Very sure" : p >= 0.65 ? "Fairly sure" : "Not very sure";
}

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  let body = null;
  try { body = await res.json(); } catch { /* not JSON */ }
  if (!res.ok) throw new Error((body && body.error) || `Request failed (${res.status})`);
  return body;
}

let toastTimer;
function toast(msg, isError = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast" + (isError ? " error" : "");
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), isError ? 6000 : 3500);
}

function chip(status, text) {
  return `<span class="chip ${esc(status)}"><span class="dot"></span>${esc(text ?? STATUS[status]?.label ?? status)}</span>`;
}

// ── modal ────────────────────────────────────────────────────────────────────

const modal = $("#modal");
function openModal(html) {
  $("#modal-body").innerHTML = html;
  modal.hidden = false;
  document.body.style.overflow = "hidden";
}
function closeModal() {
  modal.hidden = true;
  document.body.style.overflow = "";
}
modal.addEventListener("click", e => { if (e.target.closest("[data-close]")) closeModal(); });
document.addEventListener("keydown", e => { if (e.key === "Escape" && !modal.hidden) closeModal(); });

// ── engine status pill ───────────────────────────────────────────────────────

let engineState = "loading";
async function pollStatus() {
  const pill = $("#engine-pill");
  try {
    const s = await api("/api/status");
    engineState = s.engine;
    pill.className = "engine-pill " + s.engine;
    const busy = s.running.length ? " · analysing" + (s.waiting ? `, ${s.waiting} waiting` : "") : "";
    $(".label", pill).textContent =
      s.engine === "ready" ? "Ready" + busy : s.engine === "loading" ? "Starting up AI…" : "AI failed to start";
    pill.title = s.engine === "error" ? s.error : `Detector ${s.detector} · Re-ID ${s.reid}`;
  } catch {
    pill.className = "engine-pill error";
    $(".label", pill).textContent = "Server offline";
  }
}
pollStatus();
setInterval(pollStatus, 4000);

// ── router ───────────────────────────────────────────────────────────────────

let pageTimer = null;
function setPoll(fn, ms) {
  clearInterval(pageTimer);
  pageTimer = fn ? setInterval(fn, ms) : null;
}

function route() {
  setPoll(null);
  closeModal();
  const [, page, arg] = location.hash.replace(/^#/, "").split("/");
  $$(".nav a").forEach(a => a.classList.toggle("active",
    a.dataset.nav === (page === "people" || page === "help" ? page : "home")));
  window.scrollTo(0, 0);
  if (page === "job" && arg) return renderJob(decodeURIComponent(arg));
  if (page === "people") return renderPeople();
  if (page === "help") return renderHelp();
  return renderHome();
}
window.addEventListener("hashchange", route);

// ── home: upload + past analyses ─────────────────────────────────────────────

function renderHome() {
  pendingFile = null;
  view.innerHTML = `
    <div class="page-head">
      <div>
        <h1>Analyse a video</h1>
        <p class="sub">Upload a recording. VideoVigil finds every person in it, works out who they are, and builds a gallery you can browse.</p>
      </div>
    </div>
    <div class="hero">
      <div class="upload-col">
        <label class="drop" id="drop">
          <input type="file" id="file" accept="video/*,.mkv,.avi,.wmv" hidden>
          <div class="big-icon">${ICON.upload}</div>
          <strong>Drop a video here</strong>
          <span class="muted">or <u>browse your computer</u> · MP4, MOV, AVI, MKV, WebM</span>
          <div class="upload-progress" id="upload-progress" hidden>
            <div class="bar"><span></span></div>
            <p class="small muted" id="upload-text" style="margin:8px 0 0"></p>
          </div>
        </label>
        <form class="card upload-form" id="upload-form" hidden novalidate>
          <div class="upload-file">${ICON.film}<strong id="up-name"></strong>
            <span class="small muted" id="up-size"></span>
            <button type="button" class="btn ghost" id="up-cancel" style="margin-left:auto">Cancel</button></div>
          <label class="field">When was this video recorded?
            <select class="input" id="up-day"></select></label>
          <input type="date" class="input" id="up-other" max="${todayIso()}" hidden aria-label="Recording date">
          <div class="small muted" id="up-hint" hidden></div>
          <div class="note" id="up-note" hidden></div>
          <button class="btn primary" type="submit" id="up-go" disabled>Start analysis</button>
        </form>
      </div>
      <div class="card steps">
        <div class="step"><div class="step-num">1</div><div><h3>Upload</h3><p>Any security or phone recording. Choose the day it was recorded, so the staff enrolled that day are recognised. Videos are analysed one at a time, in order.</p></div></div>
        <div class="step"><div class="step-num">2</div><div><h3>We do the work</h3><p>People are detected, followed through the video and compared with everyone the system knows. Bags and laptops are watched too.</p></div></div>
        <div class="step"><div class="step-num">3</div><div><h3>Review the gallery</h3><p>See each person once, with photos and when they were on screen. Tell the system who unrecognised people are, and it will recognise them next time.</p></div></div>
      </div>
    </div>
    <div class="section">
      <div class="section-head"><h2>Your analyses</h2></div>
      <div id="jobs"><div class="card empty">Loading…</div></div>
    </div>`;

  const drop = $("#drop"), input = $("#file");
  input.addEventListener("change", () => { if (input.files[0]) chooseFile(input.files[0]); input.value = ""; });
  ["dragenter", "dragover"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", e => e.dataTransfer.files[0] && chooseFile(e.dataTransfer.files[0]));
  bindUploadForm();

  loadJobs();
  setPoll(loadJobs, 2500);
}

// ── upload step 2: the recording date (required – the file name is only a hint) ──

let pendingFile = null;

async function chooseFile(file) {
  pendingFile = file;
  const days = await loadGalleryDays(true);
  if (pendingFile !== file || !$("#upload-form")) return;
  const usable = days.filter(d => d.usable);
  $("#up-name").textContent = file.name;
  $("#up-name").title = file.name;
  $("#up-size").textContent = `${(file.size / 2 ** 20).toFixed(1)} MB`;
  $("#up-day").innerHTML = `<option value="" selected disabled>Choose the recording date…</option>`
    + (usable.length ? `<optgroup label="Days with enrolled staff">${usable.map(d =>
        `<option value="${esc(d.date)}">${esc(day(d.date))} · ${d.staff} staff</option>`).join("")}</optgroup>` : "")
    + `<option value="other">${usable.length ? "Another date (no staff enrolled)…" : "Pick a date…"}</option>`;
  $("#up-other").value = "";
  $("#up-other").hidden = true;

  const hint = $("#up-hint"), suggested = dateInName(file.name);
  hint.hidden = !suggested;
  if (suggested) {
    hint.innerHTML = `The file name mentions <b>${esc(day(suggested))}</b>. <button type="button" class="link-btn" id="up-use">Use this date</button>`;
    $("#up-use").addEventListener("click", () => setUploadDate(suggested));
  }
  $("#drop").hidden = true;
  $("#upload-form").hidden = false;
  updateUploadForm();
  $("#up-day").focus();
}

function setUploadDate(iso) {
  const sel = $("#up-day");
  if ([...sel.options].some(o => o.value === iso)) {
    sel.value = iso;
    $("#up-other").hidden = true;
  } else {
    sel.value = "other";
    $("#up-other").hidden = false;
    $("#up-other").value = iso;
  }
  updateUploadForm();
}

function uploadDate() {
  const v = $("#up-day").value;
  return v === "other" ? $("#up-other").value : v;
}

function updateUploadForm() {
  const iso = uploadDate(), note = $("#up-note"), go = $("#up-go");
  const info = (galleryDays || []).find(d => d.date === iso);
  const suggested = pendingFile && dateInName(pendingFile.name);
  let text = "", warn = false, ok = !!iso;
  if (!iso) {
    text = $("#up-day").value === "other" ? "Pick the day the video was recorded." : "";
  } else if (iso > todayIso()) {
    text = "The recording date can't be in the future.";
    warn = true; ok = false;
  } else if (info && info.usable) {
    text = `The ${info.staff} staff enrolled on ${day(iso)} will be recognised, plus anyone you've named.`;
  } else {
    text = `${info && info.problem ? info.problem : `No staff are enrolled for ${day(iso)}`}. `
      + "Everyone will show as unrecognised, except people you've named.";
    warn = true;
  }
  if (ok && suggested && suggested !== iso) {
    text += ` The file name mentions ${day(suggested)}, so check that the date is right.`;
    warn = true;
  }
  note.textContent = text;
  note.className = "note" + (warn ? " warn" : "");
  note.hidden = !text;
  go.disabled = !ok;
}

function bindUploadForm() {
  const form = $("#upload-form");
  $("#up-day").addEventListener("change", () => {
    const other = $("#up-day").value === "other";
    $("#up-other").hidden = !other;
    if (other) $("#up-other").focus();
    updateUploadForm();
  });
  $("#up-other").addEventListener("input", updateUploadForm);
  $("#up-cancel").addEventListener("click", resetUploadForm);
  form.addEventListener("submit", e => {
    e.preventDefault();
    const iso = uploadDate();
    if (!pendingFile || !iso || $("#up-go").disabled) return;
    upload(pendingFile, iso);
  });
}

function resetUploadForm() {
  pendingFile = null;
  if (!$("#upload-form")) return;
  $("#upload-form").hidden = true;
  $("#drop").hidden = false;
}

function upload(file, videoDate) {
  const drop = $("#drop"), box = $("#upload-progress"), bar = $("#upload-progress .bar span"), text = $("#upload-text");
  $("#upload-form").hidden = true;
  drop.hidden = false;
  drop.classList.add("busy");
  box.hidden = false;
  const form = new FormData();
  form.append("video", file);
  form.append("video_date", videoDate);
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/jobs");
  xhr.upload.onprogress = e => {
    if (!e.lengthComputable) return;
    const pct = Math.round((e.loaded / e.total) * 100);
    bar.style.width = pct + "%";
    text.textContent = `Uploading ${file.name} – ${pct}%`;
  };
  xhr.onload = () => {
    let body = {};
    try { body = JSON.parse(xhr.responseText); } catch { /* ignore */ }
    if (xhr.status === 201) {
      toast("Uploaded – analysis started");
      pendingFile = null;
      location.hash = "#/job/" + body.id;
    } else {
      toast(body.error || "Upload failed", true);
      uploadFailed();
    }
  };
  xhr.onerror = () => { toast("Upload failed – is the server running?", true); uploadFailed(); };
  // Keep the chosen file and date so the user can fix the problem and try again.
  const uploadFailed = () => {
    drop.classList.remove("busy");
    box.hidden = true;
    bar.style.width = "0";
    if (pendingFile && $("#upload-form")) { drop.hidden = true; $("#upload-form").hidden = false; }
  };
  text.textContent = `Uploading ${file.name}…`;
  xhr.send(form);
}

async function loadJobs() {
  const box = $("#jobs");
  if (!box) return;
  let jobs;
  try { jobs = await api("/api/jobs"); } catch (e) { box.innerHTML = `<div class="card empty">${esc(e.message)}</div>`; return; }
  if (!jobs.length) {
    box.innerHTML = `<div class="card empty">No videos yet – upload one above to get started.</div>`;
    return;
  }
  box.innerHTML = `<div class="jobs">${jobs.map(jobCard).join("")}</div>`;
}

function jobCard(j) {
  const s = j.summary;
  let meta;
  if (j.status === "done" && s) {
    meta = [chip("KNOWN", `${s.known} recognised`), chip("UNKNOWN", `${s.unknown} unrecognised`)];
    if (s.suspects) meta.push(chip("SUSPECT", `${s.suspects} on watch list`));
    if (s.alerts) meta.push(chip("alert", `${s.alerts} alert${s.alerts > 1 ? "s" : ""}`));
    meta = meta.join("");
  } else if (j.status === "running") {
    meta = `<div style="flex:1"><div class="bar"><span style="width:${Math.round((j.progress || 0) * 100)}%"></span></div>
            <div class="small muted" style="margin-top:6px">${esc(j.stage)} · ${Math.round((j.progress || 0) * 100)}%</div></div>`;
  } else {
    meta = chip(j.status, j.status === "failed" ? "Failed" : JOB_STATUS[j.status]);
  }
  const poster = j.poster
    ? `<img src="/media/jobs/${esc(j.id)}/${esc(j.poster)}" alt="" loading="lazy">`
    : `<div class="placeholder">${ICON.film}</div>`;
  return `<a class="card job" href="#/job/${esc(j.id)}">
      <div class="job-poster">${poster}</div>
      <div class="job-body">
        <div class="job-name" title="${esc(j.name)}">${esc(j.name)}</div>
        <div class="small muted">${j.video_date ? "Recorded " + esc(day(j.video_date)) + " · " : ""}${when(j.created)}${s ? " · " + duration(s.duration) + " video" : ""}</div>
        <div class="job-meta">${meta}</div>
      </div></a>`;
}

// ── job page ─────────────────────────────────────────────────────────────────

let job = null;           // current job (done)
let peopleFilter = "all";
let eventFilter = "all";

async function renderJob(id) {
  view.innerHTML = `<div class="card empty">Loading…</div>`;
  let first = true;
  const tick = async () => {
    let j;
    try { j = await api("/api/jobs/" + encodeURIComponent(id)); } catch (e) {
      setPoll(null);
      view.innerHTML = `<a class="back" href="#/">${ICON.back} All analyses</a><div class="card empty">${esc(e.message)}</div>`;
      return;
    }
    if (j.status === "done") {
      setPoll(null);
      job = j;
      peopleFilter = eventFilter = "all";
      drawJob();
    } else {
      drawProgress(j, first);
    }
    first = false;
  };
  await tick();
  if (job === null || job.id !== id || job.status !== "done") setPoll(tick, 1500);
}

// Which staff gallery a job uses, and anything the user should double-check.
function galleryLine(j) {
  const g = j.gallery;
  const parts = [j.video_date ? `Recorded ${esc(day(j.video_date))}` : ""];
  if (g) {
    parts.push(g.staff ? `${g.staff} staff enrolled that day` : "no staff gallery");
    if (g.named) parts.push(`${g.named} ${g.named > 1 ? "people" : "person"} you named`);
  }
  return parts.filter(Boolean).join(" · ");
}

function galleryNotes(j) {
  const notes = [];
  if (j.gallery && j.gallery.warning) {
    notes.push(`${j.gallery.warning}${j.gallery.staff ? "" : ". Everyone was treated as unrecognised, except people you named."}`);
  }
  if (j.video_date && j.filename_date && j.filename_date !== j.video_date) {
    notes.push(`You chose ${day(j.video_date)} but the file name mentions ${day(j.filename_date)}. If the date is wrong, delete this analysis and upload it again with the right date.`);
  }
  return notes.length ? `<div class="page-notes">${notes.map(n => `<div class="note warn">${esc(n)}</div>`).join("")}</div>` : "";
}

function jobHeader(j, extra = "") {
  const g = galleryLine(j);
  return `<a class="back" href="#/">${ICON.back} All analyses</a>
    <div class="page-head">
      <div><h1>${esc(j.name)}</h1>
        <p class="sub">Uploaded ${when(j.created)} · ${j.size_mb} MB${extra}</p>
        ${g ? `<p class="sub small" style="margin-top:2px">${g}</p>` : ""}</div>
      <button class="btn danger" id="delete-job" ${j.status === "running" ? "disabled title='Wait for the analysis to finish'" : ""}>${ICON.trash} Delete</button>
    </div>${galleryNotes(j)}`;
}

function bindDelete(j) {
  $("#delete-job")?.addEventListener("click", async () => {
    if (!confirm(`Delete the analysis of “${j.name}”? The uploaded video and its results are removed.`)) return;
    try {
      await api("/api/jobs/" + encodeURIComponent(j.id), { method: "DELETE" });
      toast("Analysis deleted");
      location.hash = "#/";
    } catch (e) { toast(e.message, true); }
  });
}

function drawProgress(j, first) {
  const pct = Math.round((j.progress || 0) * 100);
  if (j.status === "failed") {
    view.innerHTML = jobHeader(j) + `<div class="card processing">
        <h2>We couldn't analyse this video</h2>
        <div class="note error">${esc(j.error || "Unknown error")}</div>
        <p class="muted" style="margin:0">Try another file, or convert it to MP4 first.</p></div>`;
    bindDelete(j);
    setPoll(null);
    return;
  }
  const waiting = j.status === "queued";
  const stage = waiting
    ? (engineState === "loading" ? "Starting up the AI models – this takes a few seconds the first time." : "Waiting for the video ahead of it to finish.")
    : j.stage;
  const html = `<div class="card processing">
      <h2>${waiting ? "In line" : "Analysing your video"}</h2>
      <div class="bar ${waiting ? "indeterminate" : ""}"><span style="${waiting ? "" : `width:${pct}%`}"></span></div>
      <div class="processing-row"><span>${esc(stage)}</span>
        <span>${waiting ? "" : pct + "%"}${j.eta ? " · about " + duration(j.eta) + " left" : ""}</span></div>
      <p class="muted small" style="margin:0">You can leave this page – the analysis keeps running and shows up under <a href="#/">Your analyses</a>.</p>
    </div>`;
  const headKey = JSON.stringify([j.status, j.gallery]);   // gallery is known once analysis starts
  if (first || !$(".processing") || headKey !== drawProgress.headKey) {
    drawProgress.headKey = headKey;
    view.innerHTML = jobHeader(j) + html;
    bindDelete(j);
  } else {
    $(".processing").outerHTML = html;
  }
}

function drawJob() {
  const j = job, s = j.summary;
  view.innerHTML = jobHeader(j, ` · ${duration(s.duration)} video · analysed in ${duration(s.processing_s)}`) + `
    <div class="stats">
      <div class="card stat"><div class="num">${s.people}</div><div class="lbl">People found</div></div>
      <div class="card stat KNOWN"><div class="num">${s.known}</div><div class="lbl">Recognised</div></div>
      <div class="card stat UNKNOWN"><div class="num">${s.unknown}</div><div class="lbl">Unrecognised</div></div>
      <div class="card stat ${s.suspects ? "alert" : ""}"><div class="num">${s.suspects}</div><div class="lbl">On watch list</div></div>
      <div class="card stat ${s.alerts ? "alert" : ""}"><div class="num">${s.alerts}</div><div class="lbl">Alerts</div></div>
    </div>
    <div class="player-row">
      <div class="card player">
        <video id="video" controls preload="metadata" poster="/media/jobs/${esc(j.id)}/${esc(j.poster)}"
               src="/media/jobs/${esc(j.id)}/${esc(j.video_url)}"></video>
        <div class="legend">
          <span><i style="border-color:var(--known)"></i>Recognised</span>
          <span><i style="border-color:var(--unknown)"></i>Unrecognised</span>
          <span><i style="border-color:var(--alert)"></i>Watch list / alert</span>
          <span><i style="border-color:var(--unclear)"></i>Still identifying</span>
          <span><i style="border-color:#00dce6"></i>Bag / laptop</span>
        </div>
      </div>
      <div class="card timeline">
        <div class="timeline-head"><h2>Activity</h2>
          <div class="seg-toggle" id="event-filter">
            <button data-f="all" class="on">Everything</button><button data-f="alerts">Alerts</button>
          </div></div>
        <div class="timeline-list" id="events"></div>
      </div>
    </div>
    ${enrollCard(j)}
    <div class="section">
      <div class="section-head">
        <div><h2>People in this video</h2><p class="sub small">Each card is one person. Click a card for photos, when they were on screen, and to tell the system who they are.</p></div>
        <div class="seg-toggle" id="people-filter"></div>
      </div>
      <div id="people"></div>
    </div>`;
  bindDelete(j);
  $("#enroll-btn")?.addEventListener("click", openEnroll);
  $("#event-filter").addEventListener("click", e => {
    const b = e.target.closest("button");
    if (!b) return;
    eventFilter = b.dataset.f;
    $$("#event-filter button").forEach(x => x.classList.toggle("on", x === b));
    drawEvents();
  });
  drawEvents();
  drawPeople();
}

// ── staff enrollment from a morning check-in video ─────────────────────────

function enrollCard(j) {
  if (!j.video_date) return "";
  const done = j.enrolled
    ? `<div class="note warn" style="margin-top:10px">Used on ${esc(when(j.enrolled.at))} to enroll
        ${j.enrolled.labels.length} staff for ${esc(day(j.enrolled.date))} (${esc(j.enrolled.labels.join(", "))}).
        Don't use this same footage to test recognition – it would only be recognising its own enrollment photos.</div>`
    : "";
  return `<div class="card enroll-cta">
      <div class="enroll-cta-row">
        <div><h2>Morning check-in video?</h2>
          <p class="sub small" style="margin:4px 0 0">Build the staff gallery for ${esc(day(j.video_date))} from the people in this
          video. Staff are then recognised in every video recorded that day.</p></div>
        <button class="btn" id="enroll-btn">${ICON.person} Build staff gallery…</button>
      </div>${done}</div>`;
}

async function openEnroll() {
  let pv;
  try { pv = await api(`/api/jobs/${encodeURIComponent(job.id)}/enroll`); }
  catch (e) { toast(e.message, true); return; }
  const ex = pv.existing;
  const canMerge = !ex || ex.can_merge;
  const existing = ex ? `
    <div class="note ${canMerge ? "" : "warn"}">
      ${ex.unreadable ? `The current staff gallery for ${esc(day(pv.date))} can't be read, so it can only be replaced.`
        : `${esc(day(pv.date))} already has ${ex.staff.length} staff (${esc(ex.staff.join(", ") || "none")}),
           from ${esc(ex.built_by)}.${canMerge ? "" : ` It was built with another Re-ID model (${esc(ex.model)}), so it can only be replaced.`}`}
      <div class="enroll-modes">
        <label><input type="radio" name="enroll-mode" value="merge" ${canMerge ? "checked" : "disabled"}>
          Add or update these people, and keep the others</label>
        <label><input type="radio" name="enroll-mode" value="replace" ${canMerge ? "" : "checked"}>
          Replace the whole day with only these people</label>
      </div>
      <div class="small muted">The current gallery is backed up before anything changes.</div>
    </div>` : `<div class="note">No staff are enrolled for ${esc(day(pv.date))} yet – this creates the day's gallery.</div>`;
  const eligible = pv.people.filter(p => !p.problem);
  const rows = pv.people.map(p => `
    <div class="enroll-row ${p.problem ? "off" : ""}">
      <div class="enroll-photo">${p.photo ? `<img src="/media/jobs/${esc(job.id)}/${esc(p.photo)}" alt="">` : ""}</div>
      <div><div><b>${esc(p.name)}</b> ${chip(p.status)}</div>
        <div class="small muted">${p.problem ? esc("Can't be enrolled: " + p.problem) : `${p.samples} clear views`}</div>
        <div class="small enroll-hint" data-hint="${esc(p.id)}"></div></div>
      <input class="input" list="staff-labels" data-person="${esc(p.id)}" maxlength="40" autocomplete="off"
        placeholder="${p.problem ? "—" : "not enrolled"}" ${p.problem ? "disabled" : ""}
        data-suggest="${esc(p.suggest || "")}" aria-label="Staff label for ${esc(p.name)}">
    </div>`).join("");
  openModal(`
    <h2 id="modal-title">Build the staff gallery for ${esc(day(pv.date))}</h2>
    <p class="muted" style="margin-top:-6px">Type each employee's staff label, such as <b>emp_04</b> or their name.
      People you leave blank aren't enrolled. For each person the ${pv.samples} clearest, most varied views are averaged into their
      gallery entry for this day. Use footage from the start of the day: people change clothes between days.</p>
    ${existing}
    <datalist id="staff-labels">${pv.labels.map(l => `<option value="${esc(l)}">`).join("")}</datalist>
    <form id="enroll-form" class="enroll-form">
      <div class="enroll-list">${rows || '<div class="note">No people were found in this video.</div>'}</div>
      <div class="form-error" id="enroll-error" hidden></div>
      <div class="enroll-actions">
        ${eligible.some(p => p.suggest) ? '<button type="button" class="btn ghost" id="enroll-fill">Fill in recognised staff</button>' : ""}
        <button class="btn primary" type="submit" ${eligible.length ? "" : "disabled"}>Build gallery</button>
      </div>
    </form>`);

  const inputs = $$("#enroll-form input[data-person]");
  const hint = () => {
    const mode = $("input[name=enroll-mode]:checked")?.value || "merge";
    const counts = {};
    inputs.forEach(i => { const v = i.value.trim().toLowerCase(); if (v) counts[v] = (counts[v] || 0) + 1; });
    inputs.forEach(i => {
      const v = i.value.trim(), box = $(`[data-hint="${i.dataset.person}"]`);
      let t = "";
      if (v && counts[v.toLowerCase()] > 1) t = "Same label as another card: they'll be combined if they look alike.";
      else if (v && ex && mode === "merge" && ex.staff.some(s => s.toLowerCase() === v.toLowerCase()))
        t = `Replaces the current ${v} for this day.`;
      box.textContent = t;
    });
  };
  inputs.forEach(i => i.addEventListener("input", hint));
  $$("input[name=enroll-mode]").forEach(r => r.addEventListener("change", hint));
  $("#enroll-fill")?.addEventListener("click", () => {
    inputs.forEach(i => { if (!i.disabled && !i.value && i.dataset.suggest) i.value = i.dataset.suggest; });
    hint();
  });
  $("#enroll-form").addEventListener("submit", async e => {
    e.preventDefault();
    const err = $("#enroll-error"), btn = e.target.querySelector("button[type=submit]");
    const assignments = inputs.filter(i => i.value.trim()).map(i => ({ person: i.dataset.person, label: i.value.trim() }));
    const mode = $("input[name=enroll-mode]:checked")?.value || "merge";
    err.hidden = true;
    if (!assignments.length) { err.textContent = "Give at least one person a staff label."; err.hidden = false; return; }
    if (ex && mode === "replace" && !confirm(`Replace the staff gallery for ${day(pv.date)}? Staff not in this video `
        + `will no longer be recognised in videos from that day (a backup is kept).`)) return;
    btn.disabled = true;
    try {
      const s = await api(`/api/jobs/${encodeURIComponent(job.id)}/enroll`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ assignments, mode }),
      });
      galleryDays = null;
      job = await api("/api/jobs/" + encodeURIComponent(job.id));
      const list = (xs) => xs.length ? esc(xs.join(", ")) : "none";
      $("#modal-body").innerHTML = `
        <h2 id="modal-title">Staff gallery saved for ${esc(day(s.date))}</h2>
        <div class="facts">
          <div class="fact"><div class="k">Added</div><div class="v">${list(s.added)}</div></div>
          <div class="fact"><div class="k">Updated</div><div class="v">${list(s.replaced)}</div></div>
          <div class="fact"><div class="k">Kept</div><div class="v">${list(s.kept)}</div></div>
          ${s.removed.length ? `<div class="fact"><div class="k">Removed</div><div class="v">${list(s.removed)}</div></div>` : ""}
        </div>
        <p class="muted">${s.staff} staff are now enrolled for this day. Videos from ${esc(day(s.date))} that you upload or
          analyse from now on recognise them.${s.backup ? " The previous gallery was backed up." : ""}</p>
        <button class="btn primary" data-close>Done</button>`;
      drawJob();
      toast(`Staff gallery for ${day(s.date)} saved`);
    } catch (ex2) {
      err.textContent = ex2.message;
      err.hidden = false;
      btn.disabled = false;
    }
  });
}

function seek(t) {
  const v = $("#video");
  if (!v) return;
  closeModal();
  v.currentTime = Math.max(0, t - 0.5);
  v.play().catch(() => {});
  v.scrollIntoView({ behavior: "smooth", block: "center" });
}

function drawEvents() {
  const evs = job.events.filter(e => eventFilter === "all" || e.level !== "person");
  $("#events").innerHTML = evs.length
    ? evs.map((e, i) => `<button class="event ev-${esc(e.level)}" data-i="${i}">
        <time>${clock(e.t)}</time><span class="ev-dot"></span><span class="ev-text">${esc(e.text)}</span></button>`).join("")
    : `<div class="empty small">${eventFilter === "all" ? "Nothing happened in this video." : "No alerts – nothing suspicious happened."}</div>`;
  $$("#events .event").forEach(b => b.addEventListener("click", () => seek(evs[+b.dataset.i].t)));
}

function drawPeople() {
  const counts = { all: job.people.length };
  job.people.forEach(p => (counts[p.status] = (counts[p.status] || 0) + 1));
  const filters = [["all", "All"], ["KNOWN", "Recognised"], ["UNKNOWN", "Unrecognised"], ["SUSPECT", "Watch list"], ["UNCLEAR", "Not enough view"]]
    .filter(([k]) => k === "all" || counts[k]);
  $("#people-filter").innerHTML = filters.map(([k, l]) =>
    `<button data-f="${k}" class="${peopleFilter === k ? "on" : ""}">${l} ${counts[k]}</button>`).join("");
  $$("#people-filter button").forEach(b => b.addEventListener("click", () => { peopleFilter = b.dataset.f; drawPeople(); }));

  const list = job.people.filter(p => peopleFilter === "all" || p.status === peopleFilter);
  $("#people").innerHTML = list.length
    ? `<div class="people">${list.map(personCard).join("")}</div>`
    : `<div class="card empty">No people were found in this video.</div>`;
  $$("#people .person").forEach(c => c.addEventListener("click", () => openPerson(c.dataset.id)));
}

function personCard(p) {
  const photo = p.photos[0] ? `<img src="/media/jobs/${esc(job.id)}/${esc(p.photos[0])}" alt="${esc(p.name)}" loading="lazy">` : "";
  const sure = p.confidence != null && (p.status === "KNOWN" || p.status === "UNKNOWN")
    ? `<div class="sure" title="How sure the system is"><div class="bar"><span style="width:${Math.round(p.confidence * 100)}%"></span></div>${sureWord(p.confidence)}</div>`
    : "";
  const flags = [p.carrying ? `<span class="chip" title="Carried a bag or laptop">${ICON.bag} carried item</span>` : "",
                 p.intrusion ? `<span class="chip alert">restricted area</span>` : "",
                 p.named_by_you ? `<span class="chip">named by you</span>` : ""].join("");
  return `<button class="card person ${esc(p.status)}" data-id="${esc(p.id)}">
      <div class="person-photo">${photo}</div>
      <div class="person-body">
        <div class="person-name"><span>${esc(p.name)}</span></div>
        ${chip(p.status)}
        <div class="person-facts">On screen ${duration(p.visible_s)} · from ${clock(p.first_seen)}</div>
        ${sure}
        ${flags ? `<div class="flags">${flags}</div>` : ""}
      </div></button>`;
}

function openPerson(pid) {
  const p = job.people.find(x => x.id === pid);
  if (!p) return;
  const st = STATUS[p.status];
  const photos = p.photos.map(f => `<img src="/media/jobs/${esc(job.id)}/${esc(f)}" alt="">`).join("");
  const segs = p.segments.map(([a, b]) =>
    `<button class="btn" data-t="${a}">${ICON.play} ${clock(a)} – ${clock(b)}</button>`).join("");
  let remember = "";
  if (p.status !== "KNOWN") {
    remember = p.can_remember
      ? `<div class="remember">
          <div><h3>Do you know this person?</h3>
            <p class="muted small" style="margin:4px 0 0">Give them a name and VideoVigil will recognise them in the videos you upload from now on.</p></div>
          <form id="remember-form">
            <input class="input" name="name" placeholder="Their name, e.g. Sara Khan" maxlength="40" required autocomplete="off">
            <button class="btn primary" type="submit">Remember this person</button>
          </form>
          <div class="form-error" id="remember-error" hidden></div>
        </div>`
      : `<div class="note">We never got a clear enough look at this person to remember them.</div>`;
  }
  const certainty = p.confidence != null ? `${sureWord(p.confidence)} (${Math.round(p.confidence * 100)}%)` : "—";
  openModal(`
    <div class="detail-head"><h2 id="modal-title">${esc(p.name)}</h2>${chip(p.status)}
      ${p.named_by_you ? '<span class="chip">named by you</span>' : ""}</div>
    <p class="muted" style="margin:-8px 0 16px">${esc(st.help)}</p>
    <div class="photo-strip">${photos || '<div class="note">No photos</div>'}</div>
    <div class="facts">
      <div class="fact"><div class="k">On screen for</div><div class="v">${duration(p.visible_s)}</div></div>
      <div class="fact"><div class="k">First / last seen</div><div class="v">${clock(p.first_seen)} – ${clock(p.last_seen)}</div></div>
      <div class="fact"><div class="k">How sure</div><div class="v">${certainty}</div></div>
      <div class="fact"><div class="k">Carried a bag / laptop</div><div class="v">${p.carrying ? "Yes" : "No"}</div></div>
    </div>
    <h3>When they were on screen</h3>
    <p class="muted small" style="margin:2px 0 0">Click to jump the video there.</p>
    <div class="appearances">${segs}</div>
    ${remember}`);
  $$(".appearances .btn").forEach(b => b.addEventListener("click", () => seek(+b.dataset.t)));
  $("#remember-form")?.addEventListener("submit", async e => {
    e.preventDefault();
    const btn = e.target.querySelector("button"), err = $("#remember-error");
    btn.disabled = true;
    err.hidden = true;
    try {
      const name = e.target.name.value.trim();
      const updated = await api(`/api/jobs/${encodeURIComponent(job.id)}/people/${encodeURIComponent(pid)}/remember`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }),
      });
      job = await api("/api/jobs/" + encodeURIComponent(job.id));
      closeModal();
      drawJob();
      toast(`Saved – ${updated.name} will be recognised in future videos`);
    } catch (ex) {
      err.textContent = ex.message;
      err.hidden = false;
      btn.disabled = false;
    }
  });
}

// ── known people ─────────────────────────────────────────────────────────────

let staffDay = null;      // day shown under "Enrolled staff" (null = default gallery)

async function renderPeople() {
  view.innerHTML = `<div class="page-head"><div><h1>Known people</h1>
      <p class="sub">Everyone VideoVigil can recognise. Add people from any analysed video – open a person and choose “Remember this person”.</p></div></div>
    <div id="known"><div class="card empty">Loading…</div></div>`;
  let data, days;
  try {
    [data, days] = await Promise.all([
      api("/api/known" + (staffDay ? "?day=" + encodeURIComponent(staffDay) : "")), loadGalleryDays(true)]);
  } catch (e) { $("#known").innerHTML = `<div class="card empty">${esc(e.message)}</div>`; return; }
  const shownDay = data.staff_day;

  const photos = (base, list, query = "") => `<div class="photos">${list.slice(0, 3).map(f =>
    `<img src="${base}/${encodeURIComponent(f)}${query}" alt="" loading="lazy">`).join("") || ""}</div>`;

  const yours = data.yours.length
    ? `<div class="known-grid">${data.yours.map(p => `<div class="card known-card">
        ${photos("/media/known/" + encodeURIComponent(p.name), p.photos)}
        <div class="row"><div><h3>${esc(p.name)}</h3><div class="small muted">Added ${when(p.added)}</div></div>
          <button class="icon-btn" data-forget="${esc(p.name)}" title="Forget ${esc(p.name)}" aria-label="Forget ${esc(p.name)}">${ICON.trash}</button></div>
        <div class="small muted">From “${esc(p.source)}”</div></div>`).join("")}</div>`
    : `<div class="card empty">You haven't named anyone yet.</div>`;

  const watch = data.watchlist.length
    ? `<div class="known-grid">${data.watchlist.map(s => `<div class="card known-card">
        ${photos("/media/watchlist/" + encodeURIComponent(s.id), s.photos)}
        <div class="row"><div><h3>${esc(s.name)}</h3><div class="small muted">${esc(reasonText(s.reason))}</div></div>
          <button class="icon-btn" data-unwatch="${esc(s.id)}" title="Remove from watch list" aria-label="Remove ${esc(s.name)} from watch list">${ICON.trash}</button></div>
        <div class="small muted">First seen in “${esc(s.source)}”${s.sightings ? ` · seen ${s.sightings} more time${s.sightings > 1 ? "s" : ""}` : ""}</div></div>`).join("")}</div>`
    : `<div class="card empty">Nobody is on the watch list.</div>`;

  const staffQuery = shownDay ? "?day=" + encodeURIComponent(shownDay) : "";
  const staff = data.staff.length
    ? `<div class="known-grid">${data.staff.map(p => `<div class="card known-card">
        ${photos("/media/staff/" + encodeURIComponent(p.label), p.photos, staffQuery)}
        <h3>${esc(p.name)}</h3></div>`).join("")}</div>`
    : `<div class="card empty">${shownDay ? `No staff enrolled on ${esc(day(shownDay))}.` : "No staff enrolled."}</div>`;
  const dayPicker = days.length
    ? `<label class="staff-day small muted">Day
        <select class="input" id="staff-day">${days.map(d => `<option value="${esc(d.date)}" ${d.date === shownDay ? "selected" : ""}>
          ${esc(day(d.date))} · ${d.staff} staff${d.usable ? "" : " (can't be used)"}</option>`).join("")}</select></label>`
    : "";

  $("#known").innerHTML = `
    <div class="section" style="margin-top:0"><div class="section-head"><h2>Named by you · ${data.yours.length}</h2></div>${yours}</div>
    <div class="section"><div class="section-head"><div><h2>Watch list · ${data.watchlist.length}</h2>
      <p class="sub small">Added automatically when an unrecognised person moves or lingers at a bag or laptop. They're flagged in red if they show up again.</p></div></div>${watch}</div>
    <div class="section"><div class="section-head"><div><h2>Enrolled staff · ${data.staff.length}</h2>
      <p class="sub small">Set up by your administrator, once per day, because people wear different clothes on different days. Each video is checked against the staff enrolled on the recording date you choose when uploading.</p></div>
      ${dayPicker}</div>${staff}</div>`;
  $("#staff-day")?.addEventListener("change", e => { staffDay = e.target.value; renderPeople(); });

  $$("[data-forget]").forEach(b => b.addEventListener("click", async () => {
    const name = b.dataset.forget;
    if (!confirm(`Forget ${name}? They will no longer be recognised in new videos.`)) return;
    try { await api("/api/known/" + encodeURIComponent(name), { method: "DELETE" }); toast(`${name} forgotten`); renderPeople(); }
    catch (e) { toast(e.message, true); }
  }));
  $$("[data-unwatch]").forEach(b => b.addEventListener("click", async () => {
    if (!confirm("Remove this person from the watch list?")) return;
    try { await api("/api/watchlist/" + encodeURIComponent(b.dataset.unwatch), { method: "DELETE" }); toast("Removed from the watch list"); renderPeople(); }
    catch (e) { toast(e.message, true); }
  }));
}

function reasonText(r) {
  return { displacement: "Moved a bag or laptop", unknown_interaction: "Lingered at a bag or laptop", manual: "Flagged by an operator" }[r] || r;
}

// ── help ─────────────────────────────────────────────────────────────────────

function renderHelp() {
  view.innerHTML = `<div class="page-head"><div><h1>How it works</h1>
      <p class="sub">What VideoVigil does with your video, in plain words.</p></div></div>
    <div class="help">
      <div class="card"><h2>1 · Finding people</h2>
        <p>Every frame is scanned for people, bags, backpacks and laptops. Each person is followed as they move, so one walk across the room counts as one person.</p></div>
      <div class="card"><h2>2 · Recognising them</h2>
        <p>Several clear views of each person are compared with everyone the system knows. It only says “Recognised” once those views agree – a single lucky look isn't enough.</p>
        <p>Staff are compared with photos taken on the day the video was recorded, because people wear different clothes on different days. That's why you're asked for the recording date when you upload. If no staff were enrolled that day, everyone shows as unrecognised, except people you've named.</p>
        <ul><li><b>Recognised</b> – matches a known person.</li><li><b>Unrecognised</b> – doesn't match anyone known.</li>
        <li><b>Not enough view</b> – too small, blurry or brief to decide.</li></ul></div>
      <div class="card"><h2>3 · Building the gallery</h2>
        <p>If someone leaves and comes back, or is briefly hidden, the pieces are joined back into one person using their appearance. Each card shows their best photos and when they were on screen.</p></div>
      <div class="card"><h2>4 · Watching bags & laptops</h2>
        <p>Items that stay put are remembered. If an unrecognised person moves one or hangs around it, an alert is raised and they're added to the <a href="#/people">watch list</a>, so they're flagged if they appear in a later video.</p></div>
      <div class="card"><h2>Teaching it new people</h2>
        <p>Open any unrecognised person and give them a name. From then on, videos you upload will recognise them. Recognition is based on appearance (mainly clothing), so it works best for videos recorded on the same day.</p></div>
      <div class="card"><h2>Good to know</h2>
        <ul><li>Analysis runs on this computer; videos aren't sent anywhere.</li>
        <li>It takes a little longer than the video itself – a 1-minute clip takes about 1½ minutes.</li>
        <li>Nothing here is a final judgement – always check the video before acting on an alert.</li></ul></div>
    </div>`;
}

route();
