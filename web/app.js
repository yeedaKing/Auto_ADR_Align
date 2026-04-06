// web/app.js
// Simple UI for Auto-ADR Align API.
//
// Assumes same-origin API. If UI is hosted elsewhere, set API_BASE accordingly.
const API_BASE = ""; // e.g. "http://localhost:8000"

const el = (id) => document.getElementById(id);

const form = el("alignForm");
const guideFile = el("guideFile");
const adrFile = el("adrFile");
const guideFileStatus = el("guideFileStatus");
const adrFileStatus = el("adrFileStatus");
const renderToggle = el("renderToggle");
const qcToggle = el("qcToggle");
const fadeMs = el("fadeMs");

const band = el("band");
const stepPenalty = el("stepPenalty");
const anchorEvery = el("anchorEvery");
const simplifyEps = el("simplifyEps");
const slopeMin = el("slopeMin");
const slopeMax = el("slopeMax");

const renderCostMax = el("renderCostMax");
const qcOnRenderSkip = el("qcOnRenderSkip");
const writeSummary = el("writeSummary");

const qcCostPercentile = el("qcCostPercentile");
const segmentGuide = el("segmentGuide");
const segMinSilence = el("segMinSilence");
const segRelDb = el("segRelDb");
const writePath = el("writePath");

const submitBtn = el("submitBtn");
const resetBtn = el("resetBtn");
const clearBtn = el("clearBtn");
const stopPollBtn = el("stopPollBtn");

const formError = el("formError");

const jobIdEl = el("jobId");
const jobStatusEl = el("jobStatus");
const jobUpdatedEl = el("jobUpdated");
const jobErrorEl = el("jobError");

const artifactsEl = el("artifacts");
const audioPreviewEl = el("audioPreview");
const summaryEl = el("summary");

let pollTimer = null;
let currentJobId = null;

function updateFileStatus(inputEl, statusEl, label) {
  const file = inputEl.files?.[0];
  if (!file) {
    statusEl.textContent = "No file selected";
    statusEl.classList.add("empty");
    statusEl.classList.remove("ok");
    return;
  }

  statusEl.textContent = `${label}: ${file.name} (${bytesToHuman(file.size)})`;
  statusEl.classList.remove("empty");
  statusEl.classList.add("ok");
}

function showError(targetEl, msg) {
  targetEl.textContent = msg;
  targetEl.classList.remove("hidden");
}

function clearError(targetEl) {
  targetEl.textContent = "";
  targetEl.classList.add("hidden");
}

function setStatus(jobId, status, updated) {
  jobIdEl.textContent = jobId || "—";
  jobStatusEl.textContent = status || "—";
  jobUpdatedEl.textContent = updated || "—";
}

function bytesToHuman(n) {
  if (n === null || n === undefined) return "";
  const units = ["B", "KB", "MB", "GB"];
  let x = n;
  let i = 0;
  while (x >= 1024 && i < units.length - 1) {
    x /= 1024;
    i++;
  }
  return `${x.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
  stopPollBtn.disabled = true;
}

function startPolling(jobId) {
  stopPolling();
  stopPollBtn.disabled = false;
  pollTimer = setInterval(() => pollJob(jobId), 1000);
}

stopPollBtn.addEventListener("click", () => stopPolling());

guideFile.addEventListener("change", () => {
  updateFileStatus(guideFile, guideFileStatus, "Guide loaded");
});

adrFile.addEventListener("change", () => {
  updateFileStatus(adrFile, adrFileStatus, "ADR loaded");
});

resetBtn.addEventListener("click", () => {
  form.reset();
  clearError(formError);

  guideFileStatus.textContent = "No file selected";
  guideFileStatus.classList.add("empty");
  guideFileStatus.classList.remove("ok");

  adrFileStatus.textContent = "No file selected";
  adrFileStatus.classList.add("empty");
  adrFileStatus.classList.remove("ok");
});

clearBtn.addEventListener("click", () => {
  stopPolling();
  currentJobId = null;
  setStatus(null, null, null);
  clearError(jobErrorEl);
  artifactsEl.innerHTML = "No artifacts yet.";
  artifactsEl.classList.add("empty");
  audioPreviewEl.innerHTML = "No audio available yet.";
  audioPreviewEl.classList.add("empty");
  summaryEl.textContent = "No summary yet.";
  summaryEl.classList.add("empty");
});

async function submitJob() {
  clearError(formError);
  clearError(jobErrorEl);

  if (!guideFile.files?.[0] || !adrFile.files?.[0]) {
    showError(formError, "Please choose both Guide and ADR files.");
    return;
  }

  submitBtn.disabled = true;
  submitBtn.textContent = "Submitting…";

  try {
    const fd = new FormData();
    fd.append("guide", guideFile.files[0]);
    fd.append("adr", adrFile.files[0]);

    // main options
    fd.append("render", String(renderToggle.checked));
    fd.append("qc", String(qcToggle.checked));
    fd.append("fade_ms", String(Number(fadeMs.value || 20)));

    // dtw
    fd.append("band", String(Number(band.value || 0.15)));
    fd.append("step_penalty", String(Number(stepPenalty.value || 0.001)));
    fd.append("anchor_every", String(Number(anchorEvery.value || 5)));
    fd.append("simplify_eps", String(Number(simplifyEps.value || 0.02)));
    fd.append("slope_min", String(Number(slopeMin.value || 0.0)));
    fd.append("slope_max", String(Number(slopeMax.value || 3.0)));

    // features
    fd.append("frame_ms", String(25.0));
    fd.append("hop_ms", String(10.0));
    fd.append("n_mfcc", String(20));
    fd.append("n_mels", String(40));

    // qc
    fd.append("qc_cost_percentile", String(Number(qcCostPercentile.value || 95)));

    // misc
    fd.append("write_path", String(writePath.checked));
    fd.append("segment_guide", String(segmentGuide.checked));
    fd.append("seg_min_silence", String(Number(segMinSilence.value || 0.35)));
    fd.append("seg_rel_db", String(Number(segRelDb.value || 20)));

    // guardrails
    fd.append("render_cost_max", String(Number(renderCostMax.value || 0.25)));
    fd.append("qc_on_render_skip", String(qcOnRenderSkip.checked));
    fd.append("write_summary", String(writeSummary.checked));
    fd.append("plateau_warn_s", String(2.0));

    const resp = await fetch(`${API_BASE}/align`, {
      method: "POST",
      body: fd,
    });

    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`HTTP ${resp.status}: ${text}`);
    }

    const data = await resp.json();
    currentJobId = data.job_id;

    setStatus(data.job_id, data.status, "—");
    artifactsEl.innerHTML = "No artifacts yet.";
    artifactsEl.classList.add("empty");
    audioPreviewEl.innerHTML = "No audio available yet.";
    audioPreviewEl.classList.add("empty");
    summaryEl.textContent = "No summary yet.";
    summaryEl.classList.add("empty");

    startPolling(data.job_id);
    await pollJob(data.job_id);
  } catch (e) {
    showError(formError, e.message || String(e));
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Submit Job";
  }
}

form.addEventListener("submit", (ev) => {
  ev.preventDefault();
  submitJob();
});

async function pollJob(jobId) {
  try {
    const resp = await fetch(`${API_BASE}/jobs/${jobId}`);
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`HTTP ${resp.status}: ${text}`);
    }
    const job = await resp.json();

    setStatus(job.job_id, job.status, job.updated_at || "—");

    if (job.status === "ERROR") {
      stopPolling();
      showError(jobErrorEl, job.error || "Job failed.");
    } else {
      clearError(jobErrorEl);
    }

    renderArtifacts(job.job_id, job.artifacts || []);
    renderAudioPreview(job.job_id, job.artifacts || []);
    await tryFetchSummary(job.job_id, job.artifacts || []);

    if (job.status === "DONE" || job.status === "ERROR") {
      stopPolling();
    }
  } catch (e) {
    showError(jobErrorEl, e.message || String(e));
  }
}

function renderArtifacts(jobId, artifacts) {
  if (!artifacts || artifacts.length === 0) {
    artifactsEl.innerHTML = "No artifacts yet.";
    artifactsEl.classList.add("empty");
    return;
  }
  artifactsEl.classList.remove("empty");
  artifactsEl.innerHTML = "";

  for (const a of artifacts) {
    const row = document.createElement("div");
    row.className = "artifactRow";

    const left = document.createElement("div");
    left.className = "artifactName";
    left.textContent = a.name;

    const right = document.createElement("div");
    right.className = "artifactMeta";

    const size = document.createElement("span");
    size.textContent = a.size_bytes ? bytesToHuman(a.size_bytes) : "";

    const link = document.createElement("a");
    link.href = a.url;
    link.textContent = "download";
    link.target = "_blank";
    link.rel = "noopener noreferrer";

    right.appendChild(size);
    right.appendChild(link);

    row.appendChild(left);
    row.appendChild(right);
    artifactsEl.appendChild(row);
  }
}

function findArtifact(artifacts, name) {
  return artifacts.find((a) => a.name === name);
}

function renderAudioPreview(jobId, artifacts) {
  const wav = findArtifact(artifacts, "adr_conformed.wav");
  if (!wav) {
    audioPreviewEl.innerHTML = "No audio available yet.";
    audioPreviewEl.classList.add("empty");
    return;
  }

  audioPreviewEl.classList.remove("empty");
  audioPreviewEl.innerHTML = "";

  const card = document.createElement("div");
  card.className = "audioCard";

  const title = document.createElement("div");
  title.className = "title";
  title.textContent = "adr_conformed.wav";

  const audio = document.createElement("audio");
  audio.controls = true;
  audio.src = wav.url;

  const dl = document.createElement("div");
  dl.style.marginTop = "8px";
  const link = document.createElement("a");
  link.href = wav.url;
  link.textContent = "download wav";
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  dl.appendChild(link);

  card.appendChild(title);
  card.appendChild(audio);
  card.appendChild(dl);
  audioPreviewEl.appendChild(card);
}

async function tryFetchSummary(jobId, artifacts) {
  const sum = findArtifact(artifacts, "run_summary.json");
  if (!sum) {
    summaryEl.textContent = "No summary yet.";
    summaryEl.classList.add("empty");
    return;
  }

  try {
    const resp = await fetch(sum.url);
    if (!resp.ok) return;
    const json = await resp.json();
    summaryEl.classList.remove("empty");
    summaryEl.textContent = JSON.stringify(json, null, 2);
  } catch {
    // ignore parse/temporary errors
  }
}