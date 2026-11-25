// Voice Digits dashboard. Plain JavaScript, no build step.
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"];
const charts = {};
const pct = (x, d = 1) => (x == null ? "–" : (x * 100).toFixed(d) + "%");
const fmt = (x, d = 0) => (x == null ? "–" : Number(x).toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d }));

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  const body = res.headers.get("content-type")?.includes("json") ? await res.json() : await res.text();
  if (!res.ok) throw new Error(body.detail || body || res.statusText);
  return body;
}

function duration(s) {
  if (s == null) return "–";
  s = Math.floor(s);
  const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60);
  if (d) return `${d}d ${h}h`;
  if (h) return `${h}h ${m}m`;
  if (m) return `${m}m ${s % 60}s`;
  return `${s}s`;
}

// ------------------------------------------------------------------ charts
function baseOptions(extra = {}) {
  const grid = css("--grid"), text = css("--text-secondary");
  return {
    responsive: true, maintainAspectRatio: false, animation: { duration: 250 },
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: { position: "top", align: "start", labels: { color: text, boxWidth: 12, boxHeight: 12, useBorderRadius: true, borderRadius: 3 } },
      tooltip: { backgroundColor: css("--surface-2"), titleColor: css("--text-primary"), bodyColor: css("--text-primary"),
                 borderColor: css("--border"), borderWidth: 1, padding: 10, boxPadding: 4 },
    },
    scales: {
      x: { grid: { display: false }, ticks: { color: text }, border: { color: grid } },
      y: { grid: { color: grid }, ticks: { color: text }, border: { display: false } },
    },
    ...extra,
  };
}

function draw(id, config) {
  if (charts[id]) charts[id].destroy();
  charts[id] = new Chart(document.getElementById(id), config);
  return charts[id];
}

const bar = (label, data, color, more = {}) => ({
  label, data, backgroundColor: color, borderRadius: 4, borderSkipped: false,
  maxBarThickness: 34, categoryPercentage: 0.7, barPercentage: 0.9, ...more,
});
const line = (label, data, color, more = {}) => ({
  label, data, borderColor: color, backgroundColor: color, borderWidth: 2,
  pointRadius: 0, pointHoverRadius: 4, tension: 0.25, ...more,
});

// ------------------------------------------------------------------ tabs
function showTab(name) {
  $$(".tabs button").forEach((b) => b.setAttribute("aria-selected", b.dataset.tab === name));
  $$(".tab").forEach((t) => t.classList.toggle("active", t.id === name));
  location.hash = name;
  ({ insights: loadInsights, retrain: refreshRetrain, load: loadLoadTests, overview: refreshOverview }[name] || (() => {}))();
}
$$(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
$$(".subtabs button").forEach((b) => b.addEventListener("click", () => {
  $$(".subtabs button").forEach((x) => x.setAttribute("aria-selected", x === b));
  $$(".sub").forEach((s) => s.classList.toggle("active", s.id === b.dataset.sub));
}));

// ------------------------------------------------------------------ overview + uptime
const pings = [];
let lastHealth = null;

async function ping() {
  const t0 = performance.now();
  try {
    const h = await api("/health");
    const ms = performance.now() - t0;
    lastHealth = h;
    pings.push({ t: new Date(), ms });
    $("#status").className = "status up";
    $("#status-text").textContent = `Online · up ${duration(h.uptime_seconds)}`;
    $("#t-uptime").textContent = duration(h.uptime_seconds);
    $("#t-started").textContent = "since " + new Date(h.started_at).toLocaleString();
    $("#t-version").textContent = "v" + h.model_version;
    if (h.engine === "lite" && !$("#r-explain").dataset.lite) {
      // Light engine (small servers): the network stays fixed, only its last layer is retrained.
      $("#r-explain").dataset.lite = "1";
      $("#r-explain").innerHTML = "This server runs the <strong>light engine</strong>: the trained network stays fixed and only its last layer learns. <strong>Retrain</strong> starts from the live model's last layer; <strong>Retrain last layer from zero</strong> starts that layer over. Either way, the new model only goes live if it is not worse on the test set.";
      $("#r-train").textContent = "Retrain last layer from zero";
    }
    $("#t-model-age").textContent = "loaded " + duration(h.model_uptime_seconds) + " ago";
    $("#t-ping").textContent = fmt(ms) + " ms";
    $("#t-served").textContent = "served by " + h.served_by;
  } catch (e) {
    pings.push({ t: new Date(), ms: null });
    $("#status").className = "status down";
    $("#status-text").textContent = "Offline";
  }
  while (pings.length > 60) pings.shift();
  if ($("#overview").classList.contains("active")) drawPings();
}

function drawPings() {
  const c = charts["c-ping"];
  const labels = pings.map((p) => p.t.toLocaleTimeString());
  const data = pings.map((p) => p.ms);
  if (c) { c.data.labels = labels; c.data.datasets[0].data = data; c.update("none"); return; }
  draw("c-ping", { type: "line", data: { labels, datasets: [line("Response time (ms)", data, css("--series-1"), { spanGaps: false })] },
    options: baseOptions({ plugins: { ...baseOptions().plugins, legend: { display: false } },
      scales: { ...baseOptions().scales, x: { ...baseOptions().scales.x, ticks: { color: css("--text-secondary"), maxTicksLimit: 6 } },
                y: { ...baseOptions().scales.y, beginAtZero: true, title: { display: true, text: "ms", color: css("--text-secondary") } } } }) });
}

async function refreshOverview() {
  try {
    const m = await api("/model");
    const t = m.info.test_metrics || {};
    $("#t-acc").textContent = pct(t.accuracy);
    $("#t-f1").textContent = t.f1_macro != null ? `F1 ${t.f1_macro.toFixed(3)} on ${t.n} clips` : "";
  } catch (e) { /* shown by status */ }
  try {
    const s = await api("/stats");
    $("#t-preds").textContent = fmt(s.summary.n);
    $("#t-latency").textContent = s.summary.avg_ms != null ? `average ${fmt(s.summary.avg_ms)} ms` : "none yet";
    $("#latency-empty").hidden = s.recent.length > 0;
    const labels = s.recent.map((r) => new Date(r.at).toLocaleTimeString());
    draw("c-latency", { type: "line",
      data: { labels, datasets: [line("Prediction latency (ms)", s.recent.map((r) => r.latency_ms), css("--series-1"), { pointRadius: 2 })] },
      options: baseOptions({ plugins: { ...baseOptions().plugins, legend: { display: false },
        tooltip: { ...baseOptions().plugins.tooltip, callbacks: { afterLabel: (ctx) => {
          const r = s.recent[ctx.dataIndex]; return `digit ${r.digit} (${pct(r.confidence, 0)}) · ${r.served_by}`; } } } },
        scales: { ...baseOptions().scales, x: { ...baseOptions().scales.x, ticks: { color: css("--text-secondary"), maxTicksLimit: 6 } },
                  y: { ...baseOptions().scales.y, beginAtZero: true, title: { display: true, text: "ms", color: css("--text-secondary") } } } }) });
  } catch (e) { /* ignore */ }
  drawPings();
}

// ------------------------------------------------------------------ predict
let lastBlob = null;

function showPrediction(r, trueLabel) {
  $("#p-result").hidden = false;
  $("#p-digit").textContent = r.digit;
  $("#p-word").textContent = `“${r.word}” · ${pct(r.confidence)} sure`;
  const truth = trueLabel != null ? ` · true digit ${trueLabel} ${Number(trueLabel) === r.digit ? "✓" : "✗"}` : "";
  $("#p-meta").textContent = `model v${r.model_version} · ${fmt(r.latency_ms)} ms · ${r.served_by}${truth}`;
  const blue = css("--series-1"), soft = css("--muted-mark");
  draw("c-proba", { type: "bar",
    data: { labels: WORDS.map((w, i) => `${i} ${w}`),
            datasets: [bar("Probability", r.probabilities, r.probabilities.map((_, i) => (i === r.digit ? blue : soft)))] },
    options: baseOptions({ plugins: { ...baseOptions().plugins, legend: { display: false },
      tooltip: { ...baseOptions().plugins.tooltip, callbacks: { label: (c) => pct(c.raw, 2) } } },
      scales: { ...baseOptions().scales, y: { ...baseOptions().scales.y, min: 0, max: 1, ticks: { color: css("--text-secondary"), callback: (v) => pct(v, 0) } } } }) });
}

async function predictBlob(blob, name, trueLabel) {
  $("#p-error").textContent = "";
  lastBlob = { blob, name };
  const audio = $("#p-audio");
  audio.src = URL.createObjectURL(blob);
  audio.hidden = false;
  const fd = new FormData();
  fd.append("file", blob, name);
  try {
    const r = await api("/predict", { method: "POST", body: fd });
    showPrediction(r, trueLabel);
    $("#p-save").hidden = false;
    $("#p-save-label").value = trueLabel ?? r.digit;
  } catch (e) {
    $("#p-result").hidden = true;
    $("#p-error").textContent = "Could not predict: " + e.message;
  }
}

$("#p-file").addEventListener("change", (e) => {
  const f = e.target.files[0];
  if (f) predictBlob(f, f.name, (f.name.match(/^(\d)[_\- .]/) || [])[1]);
  e.target.value = "";
});

$("#p-sample").addEventListener("click", async () => {
  const res = await fetch("/sample?dataset=" + (Math.random() < 0.5 ? "test" : "new_voice"));
  const label = res.headers.get("X-True-Label");
  const name = res.headers.get("X-File-Name") || "sample.wav";
  predictBlob(await res.blob(), name, label);
});

WORDS.forEach((w, i) => {
  $("#p-save-label").add(new Option(`${i} (${w})`, i));
  $("#u-label").add(new Option(`all are ${i} (${w})`, i));
});

$("#p-save-btn").addEventListener("click", async () => {
  if (!lastBlob) return;
  const label = $("#p-save-label").value;
  const fd = new FormData();
  fd.append("files", lastBlob.blob, `${label}_mic_${Date.now()}.wav`);
  fd.append("label", label);
  fd.append("speaker", "microphone");
  try {
    const r = await api("/upload", { method: "POST", body: fd });
    $("#p-error").textContent = r.saved ? `Saved for retraining as digit ${label}.` : (r.duplicates ? "Already saved." : "Not saved: " + (r.rejected[0]?.reason || ""));
  } catch (e) { $("#p-error").textContent = e.message; }
});

// Microphone: record raw samples and encode a 16-bit mono WAV in the browser.
// The stream stays open during a session so the browser asks only once.
let micStream = null;
async function openMic() {
  if (!micStream) micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  return micStream;
}
function closeMic() {
  if (micStream) micStream.getTracks().forEach((t) => t.stop());
  micStream = null;
}
async function recordClip(ms = 2000) {
  const stream = await openMic();
  const ctx = new AudioContext();
  const src = ctx.createMediaStreamSource(stream);
  const proc = ctx.createScriptProcessor(4096, 1, 1);
  const chunks = [];
  proc.onaudioprocess = (e) => chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  src.connect(proc); proc.connect(ctx.destination);
  await new Promise((r) => setTimeout(r, ms));
  proc.disconnect(); src.disconnect();
  const rate = ctx.sampleRate; await ctx.close();
  const samples = new Float32Array(chunks.reduce((n, c) => n + c.length, 0));
  let o = 0; chunks.forEach((c) => { samples.set(c, o); o += c.length; });
  return encodeWav(samples, rate);
}
const MIC_HELP = "Microphone not available (it needs https or localhost, and your permission).";

$("#p-record").addEventListener("click", async () => {
  const btn = $("#p-record");
  btn.classList.add("recording"); btn.textContent = "● Say one digit now…"; btn.disabled = true;
  try {
    const blob = await recordClip(2000);
    predictBlob(blob, `mic_${Date.now()}.wav`);
  } catch (e) { $("#p-error").textContent = MIC_HELP; }
  closeMic();
  btn.classList.remove("recording"); btn.textContent = "● Record 2 s"; btn.disabled = false;
});

// ------------------------------------------------------------------ my voice
const speakerName = () => ($("#v-name").value.trim().toLowerCase().replace(/[^a-z0-9-]/g, "") || "me").slice(0, 20);
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const shuffled = (a) => { a = [...a]; for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };

function showPrompt(box, digit, state, live = false) {
  box.hidden = false;
  box.classList.toggle("live", live);
  box.querySelector(".say").innerHTML = digit == null ? "" : `${WORDS[digit]}<small>${digit}</small>`;
  box.querySelector(".state").textContent = state;
}

async function recordDigit(box, digit, i, n) {
  showPrompt(box, digit, `${i + 1} of ${n} · get ready…`);
  await wait(900);
  showPrompt(box, digit, "● Speak now", true);
  const blob = await recordClip(2000);
  showPrompt(box, digit, "Got it");
  return blob;
}

async function saveClip(blob, digit, name) {
  const fd = new FormData();
  fd.append("files", blob, `${digit}_${name}_${Date.now()}.wav`);
  fd.append("label", String(digit));
  fd.append("speaker", name);
  return api("/upload", { method: "POST", body: fd });
}

function setVoiceButtons(disabled) { $$("#v-test, #v-teach").forEach((b) => (b.disabled = disabled)); }

$("#v-test").addEventListener("click", async () => {
  const box = $("#v-test-prompt"), name = speakerName(), save = $("#v-save").checked;
  const digits = shuffled([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const results = [];
  setVoiceButtons(true);
  $("#v-score").hidden = true; $("#v-chips").innerHTML = "";
  try {
    await openMic();
    for (let i = 0; i < digits.length; i++) {
      const d = digits[i];
      const blob = await recordDigit(box, d, i, digits.length);
      const fd = new FormData(); fd.append("file", blob, `${d}_${name}_test.wav`);
      let r;
      try { r = await api("/predict", { method: "POST", body: fd }); }
      catch (e) { r = { digit: null, confidence: 0, word: "nothing heard" }; }
      results.push({ d, r });
      if (save) saveClip(blob, d, name).catch(() => {});
      const ok = r.digit === d;
      $("#v-chips").insertAdjacentHTML("beforeend", `<span class="chip ${ok ? "ok" : "bad"}" title="you said ${WORDS[d]}, it heard ${r.word}">${ok ? "✓" : "✗"} ${d}${ok ? "" : ` → ${r.digit ?? "?"}`} · ${pct(r.confidence, 0)}</span>`);
    }
  } catch (e) { showPrompt(box, null, MIC_HELP); }
  closeMic();
  setVoiceButtons(false);
  if (results.length) {
    const right = results.filter((x) => x.r.digit === x.d);
    const conf = right.length ? right.reduce((a, x) => a + x.r.confidence, 0) / right.length : 0;
    $("#v-score").hidden = false;
    $("#v-score").textContent = `${right.length} of ${results.length} right · average confidence when right: ${pct(conf, 0)}`;
    showPrompt(box, null, right.length >= 9 ? "The model knows your voice well." : "Teach it your voice (right), then test again.");
  }
});

$("#v-teach").addEventListener("click", async () => {
  const box = $("#v-teach-prompt"), name = speakerName(), rounds = Number($("#v-rounds").value);
  const digits = [];
  for (let k = 0; k < rounds; k++) digits.push(...shuffled([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]));
  let saved = 0, last = null;
  setVoiceButtons(true);
  $("#v-retrain").hidden = true;
  $("#v-teach-progress").hidden = false; $("#v-teach-bar").style.width = "0%";
  try {
    await openMic();
    for (let i = 0; i < digits.length; i++) {
      const blob = await recordDigit(box, digits[i], i, digits.length);
      try { last = await saveClip(blob, digits[i], name); saved += last.saved; }
      catch (e) { /* skip a clip the server rejected */ }
      $("#v-teach-bar").style.width = `${((i + 1) / digits.length) * 100}%`;
      $("#v-teach-status").textContent = `${saved} clips saved as "${name}"`;
    }
  } catch (e) { showPrompt(box, null, MIC_HELP); }
  closeMic();
  setVoiceButtons(false);
  showPrompt(box, null, `Done: ${saved} clips saved.`);
  if (last && last.trigger) {
    const t = last.trigger;
    $("#v-teach-status").textContent = `${saved} clips saved as "${name}". ` +
      (t.fired ? `Retraining recommended: ${t.reasons.join("; ")}.` : `${t.pending} recordings are waiting for a retrain.`);
  }
  $("#v-retrain").hidden = saved === 0;
});

$("#v-retrain").addEventListener("click", async () => {
  showTab("retrain");
  await startRun("retrain");
});

function encodeWav(samples, rate) {
  const buf = new ArrayBuffer(44 + samples.length * 2), v = new DataView(buf);
  const str = (off, s) => [...s].forEach((ch, i) => v.setUint8(off + i, ch.charCodeAt(0)));
  str(0, "RIFF"); v.setUint32(4, 36 + samples.length * 2, true); str(8, "WAVE"); str(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  str(36, "data"); v.setUint32(40, samples.length * 2, true);
  samples.forEach((s, i) => v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, s)) * 0x7fff, true));
  return new Blob([buf], { type: "audio/wav" });
}

// ------------------------------------------------------------------ insights
let insights = null;

function rangeChart(id, labels, stats, unit, colors, legendNames, digits = 2) {
  // Floating bar = middle half of the recordings (25th to 75th percentile); dot = median.
  const fmtv = (v) => Number(v).toFixed(digits);
  const ranges = stats.map((s) => [s.q1, s.q3]);
  const datasets = [];
  legendNames.forEach((name, k) => {
    datasets.push(bar(name, ranges.map((r, i) => (colors[i] === k ? r : null)), colors.map((c) => [css("--series-1"), css("--series-2"), css("--series-3")][c]),
      { grouped: false, maxBarThickness: 30, order: 1 }));
  });
  // order 0 is drawn on top of the bars (order 1)
  datasets.push({ type: "line", label: "Median", data: stats.map((s) => s.median), showLine: false, order: 0,
    pointStyle: "line", pointRadius: 13, pointHoverRadius: 14, borderWidth: 3, borderColor: css("--text-primary"), backgroundColor: css("--text-primary") });
  draw(id, { type: "bar", data: { labels, datasets },
    options: baseOptions({
      plugins: { ...baseOptions().plugins,
        tooltip: { ...baseOptions().plugins.tooltip, mode: "index", filter: (i) => i.raw != null,
          callbacks: { label: (c) => (Array.isArray(c.raw) ? `middle half: ${fmtv(c.raw[0])}–${fmtv(c.raw[1])} ${unit}` : `median: ${fmtv(c.raw)} ${unit}`) } } },
      scales: { ...baseOptions().scales, y: { ...baseOptions().scales.y, title: { display: true, text: unit, color: css("--text-secondary") } } } }) });
}

async function loadInsights() {
  if (insights) return;
  try { insights = await api("/insights"); } catch (e) { $("#s-duration").textContent = e.message; return; }
  const I = insights, labels = WORDS.map((w, i) => `${i} ${w}`);
  const med = (f) => I.by_digit[f].map((s) => s.median);
  const order = (f) => med(f).map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).map((x) => x[1]);

  // 1 duration
  rangeChart("c-duration", labels, I.by_digit.duration_s, "seconds", Array(10).fill(0), ["Spoken length"]);
  const d = order("duration_s"), dm = med("duration_s");
  const cap = (w) => w[0].toUpperCase() + w.slice(1);
  const durLo = Math.min(...I.by_digit.duration_s.map((s) => s.q1)), durHi = Math.max(...I.by_digit.duration_s.map((s) => s.q3));
  $("#s-duration").innerHTML = `<h3>${cap(WORDS[d[0]])} and ${WORDS[d[1]]} take longest</h3>
    <p>After cutting the silence, <strong>${WORDS[d[0]]}</strong> (${dm[d[0]].toFixed(2)} s) and <strong>${WORDS[d[1]]}</strong> (${dm[d[1]].toFixed(2)} s) are the longest words, and <strong>${WORDS[d[9]]}</strong> (${dm[d[9]].toFixed(2)} s) the shortest.</p>
    <p>The bars overlap a lot, so length alone can't tell the digits apart. The middle half of every digit lasts between ${durLo.toFixed(2)} and ${durHi.toFixed(2)} seconds, so a <strong>1-second window</strong> holds a whole word. For the few longer recordings, the app keeps the loudest second.</p>`;

  // 2 zero-crossing rate
  rangeChart("c-zcr", labels, I.by_digit.zcr, "zero-crossing rate", Array(10).fill(0), ["Zero-crossing rate"], 3);
  const z = order("zcr"), zm = med("zcr");
  $("#s-zcr").innerHTML = `<h3>“Six” hisses</h3>
    <p>The zero-crossing rate counts how often the wave crosses zero. Hissing sounds like <em>s</em> and <em>f</em> are noise, and noise crosses zero all the time.</p>
    <p><strong>${WORDS[z[0]]}</strong> (${zm[z[0]].toFixed(2)}) stands far above the rest because it has two <em>s</em> sounds. Next come <strong>${WORDS[z[1]]}</strong>, <strong>${WORDS[z[2]]}</strong> and <strong>${WORDS[z[3]]}</strong>. Voiced words like <strong>${WORDS[z[9]]}</strong> and <strong>${WORDS[z[8]]}</strong> sit lowest.</p>
    <p>This is why the spectrogram works: hiss shows up as energy in the high frequency bands, a pattern the CNN can see.</p>`;

  // 3 pitch by voice: FSDD speakers, Speech Commands as one group, the held-out voice
  const SC = "speech commands";
  const spk = Object.keys(I.by_speaker.pitch_hz).sort((a, b) => (a === SC) - (b === SC));
  const newVoice = spk.filter((s) => (I.speakers[s] || {}).new_voice);
  const fsdd = spk.filter((s) => s !== SC && !newVoice.includes(s));
  const order3 = [...fsdd, ...(spk.includes(SC) ? [SC] : []), ...newVoice];
  rangeChart("c-pitch", order3, order3.map((s) => I.by_speaker.pitch_hz[s]), "pitch (Hz)",
    order3.map((s) => (newVoice.includes(s) ? 1 : s === SC ? 2 : 0)),
    ["FSDD speakers (training)", "New voice (never trained on)", "Speech Commands, thousands of voices (training)"], 0);
  const pm = Object.fromEntries(spk.map((s) => [s, I.by_speaker.pitch_hz[s].median]));
  const lo = Math.min(...fsdd.map((s) => pm[s])), hi = Math.max(...fsdd.map((s) => pm[s]));
  const scq = I.by_speaker.pitch_hz[SC];
  const rmsQuiet = fsdd.filter((s) => I.by_speaker.rms_db[s].median < -38);
  $("#s-pitch").innerHTML = `<h3>Six voices are not enough</h3>
    <p>FSDD's five training speakers sit between <strong>${lo.toFixed(0)}</strong> and <strong>${hi.toFixed(0)} Hz</strong>, and the held-out <strong>${newVoice.join(", ")}</strong> is above all of them at <strong>${newVoice.map((s) => pm[s].toFixed(0)).join(", ")} Hz</strong>. A model that has heard only those five struggles with anyone else.</p>` +
    (scq ? `<p>The Speech Commands recordings come from thousands of people: the middle half of them spans <strong>${scq.q1.toFixed(0)}–${scq.q3.toFixed(0)} Hz</strong>, and 90% fall between ${scq.p5.toFixed(0)} and ${scq.p95.toFixed(0)} Hz. Training on that range is what lets the model handle new voices. Retraining on uploads covers the rest.</p>` : "") +
    `<p>Loudness varies too: ${rmsQuiet.join(" and ")} recorded about 15 to 20 dB quieter than the others, so every clip is scaled to the same peak before the model sees it.</p>`;

  // 4 mean spectrogram
  const picker = $("#spec-picker");
  picker.innerHTML = "";
  WORDS.forEach((w, i) => {
    const b = document.createElement("button");
    b.className = "btn small"; b.textContent = `${i} ${w}`;
    b.onclick = () => { $$("#spec-picker .btn").forEach((x) => x.classList.remove("primary")); b.classList.add("primary"); drawSpec(i); };
    picker.appendChild(b);
  });
  picker.children[6].click();
  $("#s-spec").innerHTML = `<h3>What the model actually sees</h3>
    <p>Each picture is the average log-mel spectrogram of 60 training recordings of one digit. Time runs left to right, low frequencies at the bottom, and darker means louder.</p>
    <p>Compare <strong>6 six</strong> (dark high bands at both ends: the two <em>s</em> sounds) with <strong>0 zero</strong> (energy in the low bands for longer). Every digit has its own shape, and the CNN learns to recognise these shapes.</p>`;

  // 5 class balance
  const sets = [["train", "Training", "--series-1"], ["test", "Test", "--series-2"], ["new_voice", "New voice", "--series-3"], ["uploads", "Uploaded", "--series-4"]];
  draw("c-counts", { type: "bar",
    data: { labels, datasets: sets.map(([k, name, c]) => bar(name, I.counts[k] || Array(10).fill(0), css(c), { borderColor: css("--surface-1"), borderWidth: { top: 2 } })) },
    options: baseOptions({ scales: { x: { ...baseOptions().scales.x, stacked: true }, y: { ...baseOptions().scales.y, stacked: true, title: { display: true, text: "recordings", color: css("--text-secondary") } } } }) });
  const up = (I.counts.uploads || []).reduce((a, b) => a + b, 0);
  $("#s-counts").innerHTML = `<h3>Every digit has the same number of recordings</h3>
    <p>Per digit: 625 training recordings (225 from FSDD's five speakers, 400 from different Speech Commands speakers) and 125 test recordings (25 FSDD, 100 from Speech Commands speakers never used in training), plus 50 from the held-out voice. The classes are balanced, so plain accuracy is a fair score, and no class weights are needed.</p>
    <p>Uploaded so far: <strong>${up}</strong> recordings.</p>`;
}

function drawSpec(d) {
  const m = insights.mean_spectrogram[d], rows = m.length, cols = m[0].length;
  const cv = $("#c-spec"), ctx = cv.getContext("2d");
  cv.width = cols * 10; cv.height = rows * 5;
  const a = hex(css("--seq-light")), b = hex(css("--seq-dark"));
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    const t = Math.max(0, Math.min(1, (m[r][c] - 0.25) / 0.65));
    ctx.fillStyle = `rgb(${a.map((x, i) => Math.round(x + (b[i] - x) * t)).join(",")})`;
    ctx.fillRect(c * 10, (rows - 1 - r) * 5, 10, 5);
  }
}
const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));

// ------------------------------------------------------------------ upload + retrain
$("#u-files").addEventListener("change", (e) => {
  const n = e.target.files.length;
  $("#u-chosen").textContent = n ? `${n} file${n > 1 ? "s" : ""} chosen (${(([...e.target.files].reduce((a, f) => a + f.size, 0)) / 1e6).toFixed(1)} MB)` : "No files chosen";
});

$("#u-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const files = $("#u-files").files;
  if (!files.length) { $("#u-result").textContent = "Choose some files first."; return; }
  const fd = new FormData();
  [...files].forEach((f) => fd.append("files", f, f.webkitRelativePath || f.name));
  fd.append("label", $("#u-label").value);
  fd.append("speaker", $("#u-speaker").value);
  $("#u-submit").disabled = true; $("#u-submit").textContent = "Uploading and checking…";
  try {
    const r = await api("/upload", { method: "POST", body: fd });
    const rej = r.rejected.length ? ` ${r.rejected.length} rejected (${r.rejected.slice(0, 3).map((x) => `${x.file}: ${x.reason}`).join("; ")}${r.rejected.length > 3 ? "…" : ""}).` : "";
    const dup = r.duplicates ? ` ${r.duplicates} already uploaded.` : "";
    const acc = r.batch_accuracy != null ? ` The live model got ${pct(r.batch_accuracy)} of them right.` : "";
    $("#u-result").textContent = `Saved ${r.saved} recording${r.saved === 1 ? "" : "s"}.${dup}${rej}${acc}`;
    showTrigger(r.trigger);
    if (r.auto_retrain_run) { $("#r-status").textContent = `The trigger started retraining run #${r.auto_retrain_run}.`; pollRun(); }
    insights = null;
    $("#u-files").value = ""; $("#u-chosen").textContent = "No files chosen";
    loadUploadTotals();
  } catch (err) { $("#u-result").textContent = "Upload failed: " + err.message; }
  $("#u-submit").disabled = false; $("#u-submit").textContent = "Upload";
});

function showTrigger(t) {
  const box = $("#trigger-box");
  box.className = "trigger " + (t.fired ? "fired" : "calm");
  $("#trigger-state").innerHTML = t.fired ? "⚠ Retraining recommended" : "✓ No retraining needed";
  $("#trigger-reasons").innerHTML = (t.fired ? t.reasons : [`${t.pending} new recording${t.pending === 1 ? "" : "s"} waiting` + (t.accuracy_on_pending != null ? `; live model gets ${pct(t.accuracy_on_pending)} of them right` : "")]).map((r) => `<li>${r}</li>`).join("");
  $("#trigger-rules").textContent = `Fires when ${t.rules.min_new_samples}+ new recordings are waiting, or when the live model gets under ${pct(t.rules.accuracy_below, 0)} of ${t.rules.min_for_accuracy}+ new recordings right. Automatic retraining is ${t.auto_retrain ? "on" : "off"}: press Retrain now.`;
}

async function loadUploadTotals() {
  const u = await api("/uploads");
  const t = u.totals;
  $("#upload-totals").innerHTML = `<span>Uploaded in total</span><span>${fmt(t.total)}</span><span>Waiting for a retrain</span><span>${fmt(t.waiting)}</span><span>Kept aside to check new models</span><span>${fmt(t.holdout)}</span>`;
}

async function refreshRetrain() {
  try { showTrigger(await api("/trigger")); } catch (e) { $("#trigger-state").textContent = e.message; }
  loadUploadTotals();
  loadRuns();
}

async function startRun(mode) {
  const fd = new FormData(); fd.append("mode", mode);
  try {
    const r = await api("/retrain", { method: "POST", body: fd });
    $("#r-status").textContent = `Run #${r.run_id} started.`;
    pollRun();
  } catch (e) { $("#r-status").textContent = e.message; }
}
$("#r-retrain").addEventListener("click", () => startRun("retrain"));
$("#r-train").addEventListener("click", () => startRun("train"));

let polling = false;
async function pollRun() {
  if (polling) return;
  polling = true;
  $$("#r-retrain, #r-train").forEach((b) => (b.disabled = true));
  $("#r-progress").hidden = false;
  while (true) {
    const s = await loadRuns();
    const r = s.latest;
    if (!r || r.status !== "running") break;
    const p = r.progress || {};
    const frac = p.epochs ? p.epoch / p.epochs : 0.03;
    $("#r-bar").style.width = Math.max(3, frac * 100) + "%";
    $("#r-status").textContent = `Run #${r.id}: ${p.stage}` + (p.epoch ? ` · epoch ${p.epoch} of up to ${p.epochs} · validation accuracy ${pct(p.val_accuracy)}` : "…");
    await new Promise((res) => setTimeout(res, 2000));
  }
  $("#r-progress").hidden = true;
  $$("#r-retrain, #r-train").forEach((b) => (b.disabled = false));
  polling = false;
  refreshRetrain();
  refreshOverview();
}

const STATUS = { running: ["run", "◷ running"], done: ["good", "✓ live"], kept_old: ["warn", "– kept old"], failed: ["bad", "✗ failed"] };
async function loadRuns() {
  const s = await api("/retrain/status");
  $("#runs-table tbody").innerHTML = s.runs.map((r) => {
    const [cls, text] = STATUS[r.status] || ["", r.status];
    const ba = (a, b) => (a == null && b == null ? "–" : `${pct(a)} → ${pct(b)}`);
    return `<tr><td>${r.id}</td><td>${r.mode === "train" ? "from scratch" : "fine-tune"}</td><td><span class="pill ${cls}">${text}</span></td>
      <td>${r.n_new ?? "–"}</td><td>${ba(r.test_acc_before, r.test_acc_after)}</td><td>${ba(r.new_acc_before, r.new_acc_after)}</td><td>${r.message || ""}</td></tr>`;
  }).join("") || `<tr><td colspan="7" class="muted">No training runs yet.</td></tr>`;
  const latest = s.latest;
  if (latest && latest.status !== "running") $("#r-status").textContent = `Last run #${latest.id}: ${latest.message || latest.status}`;
  const done = s.runs.find((r) => r.metrics && r.metrics.history);
  let hist = done ? done.metrics.history : null, title = done ? `run #${done.id}` : "";
  if (!hist) { const m = await api("/model"); hist = m.info.history; title = "first model (notebook)"; }
  if (hist) {
    const ep = hist.accuracy.map((_, i) => i + 1);
    draw("c-history", { type: "line", data: { labels: ep, datasets: [
      line("Training accuracy", hist.accuracy, css("--series-1"), { pointRadius: 2 }),
      line("Validation accuracy", hist.val_accuracy, css("--series-2"), { pointRadius: 2 })] },
      options: baseOptions({ plugins: { ...baseOptions().plugins, title: { display: true, text: title, color: css("--text-secondary"), align: "end", padding: 0 } },
        scales: { x: { ...baseOptions().scales.x, title: { display: true, text: "epoch", color: css("--text-secondary") } },
                  y: { ...baseOptions().scales.y, max: 1, ticks: { color: css("--text-secondary"), callback: (v) => pct(v, 0) } } } }) });
  }
  return s;
}

// ------------------------------------------------------------------ evaluate
$("#e-run").addEventListener("click", async () => {
  const btn = $("#e-run"); btn.disabled = true; btn.textContent = "Evaluating…";
  try {
    const r = await api("/evaluate?dataset=" + $("#e-dataset").value, { method: "POST" });
    $("#e-meta").textContent = `model v${r.model_version} · ${r.n} clips · ${r.seconds} s · ${r.served_by}`;
    const tiles = [["Accuracy", pct(r.accuracy)], ["Precision", pct(r.precision_macro)], ["Recall", pct(r.recall_macro)],
                   ["F1 score", r.f1_macro.toFixed(3)], ["Log loss", r.log_loss.toFixed(3)], ["ROC AUC", r.roc_auc_ovr ? r.roc_auc_ovr.toFixed(4) : "–"]];
    $("#e-tiles").innerHTML = tiles.map(([k, v]) => `<div class="tile"><span class="label">${k}</span><span class="value">${v}</span><span class="sub">${k.startsWith("Prec") || k.startsWith("Rec") || k.startsWith("F1") ? "macro average" : ""}</span></div>`).join("");
    drawCM(r.confusion_matrix);
    const pc = WORDS.map((_, i) => r.per_class[String(i)]);
    draw("c-perclass", { type: "bar", data: { labels: WORDS.map((w, i) => `${i}`), datasets: [
      bar("Precision", pc.map((c) => c.precision), css("--series-1")),
      bar("Recall", pc.map((c) => c.recall), css("--series-2")),
      bar("F1", pc.map((c) => c.f1), css("--series-3"))] },
      options: baseOptions({ plugins: { ...baseOptions().plugins, tooltip: { ...baseOptions().plugins.tooltip, callbacks: { title: (c) => WORDS[c[0].dataIndex], label: (c) => `${c.dataset.label}: ${pct(c.raw)}` } } },
        scales: { ...baseOptions().scales, y: { ...baseOptions().scales.y, min: 0, max: 1, ticks: { color: css("--text-secondary"), callback: (v) => pct(v, 0) } } } }) });
  } catch (e) { $("#e-meta").textContent = e.message; }
  btn.disabled = false; btn.textContent = "Evaluate live model";
});

function drawCM(cm) {
  const max = Math.max(...cm.flat()), a = hex(css("--seq-light")), b = hex(css("--seq-dark"));
  const darkMode = matchMedia("(prefers-color-scheme: dark)").matches;
  let html = `<div></div>` + WORDS.map((_, i) => `<div class="h">${i}</div>`).join("");
  cm.forEach((row, i) => {
    html += `<div class="h">${i}</div>`;
    row.forEach((v, j) => {
      const t = max ? Math.sqrt(v / max) : 0;  // square root so small mistakes still show
      const bg = v ? `rgb(${a.map((x, k) => Math.round(x + (b[k] - x) * t)).join(",")})` : "transparent";
      const ink = !v ? "var(--text-muted)" : t > 0.55 ? (darkMode ? "#0b0b0b" : "#ffffff") : "var(--text-primary)";
      html += `<div title="true ${WORDS[i]}, predicted ${WORDS[j]}: ${v}" style="background:${bg};color:${ink};${i === j ? "font-weight:700" : ""}">${v || "·"}</div>`;
    });
  });
  $("#e-cm").innerHTML = html;
}

// ------------------------------------------------------------------ load tests
async function loadLoadTests() {
  const rows = await api("/load-tests");
  if (!rows.length) { $("#load-note").textContent = "No load test results yet. Run scripts/run_load_tests.ps1."; return; }
  $("#load-table tbody").innerHTML = rows.map((r) => `<tr><td>${r.containers}</td><td>${r.users}</td><td>${fmt(r.requests)}</td><td>${fmt(r.failures)}</td><td>${fmt(r.rps, 1)}</td><td>${fmt(r.median_ms)}</td><td>${fmt(r.p95_ms)}</td><td>${fmt(r.avg_ms)}</td></tr>`).join("");
  const users = [...new Set(rows.map((r) => r.users))].sort((a, b) => a - b);
  const counts = [...new Set(rows.map((r) => r.containers))].sort((a, b) => a - b);
  const colors = ["--series-1", "--series-2", "--series-3", "--series-4"];
  const names = { median_ms: "Median response time (ms)", p95_ms: "95th percentile (ms)", rps: "Requests per second" };
  const drawLoad = (metric) => {
    $$("#load-metric .btn").forEach((b) => b.classList.toggle("primary", b.dataset.metric === metric));
    draw("c-load", { type: "bar", data: { labels: users.map((u) => `${u} users`), datasets: counts.map((c, k) =>
      bar(`${c} container${c > 1 ? "s" : ""}`, users.map((u) => (rows.find((r) => r.users === u && r.containers === c) || {})[metric] ?? null), css(colors[k % 4]))) },
      options: baseOptions({ plugins: { ...baseOptions().plugins, tooltip: { ...baseOptions().plugins.tooltip, callbacks: { label: (c) => `${c.dataset.label}: ${fmt(c.raw, metric === "rps" ? 1 : 0)}` } } },
        scales: { ...baseOptions().scales, y: { ...baseOptions().scales.y, beginAtZero: true, title: { display: true, text: names[metric], color: css("--text-secondary") } } } }) });
  };
  $$("#load-metric .btn").forEach((b) => (b.onclick = () => drawLoad(b.dataset.metric)));
  drawLoad("median_ms");
  $("#load-note").textContent = rows[0].note || "";
}

// ------------------------------------------------------------------ start
ping();
setInterval(ping, 5000);
setInterval(() => { if ($("#overview").classList.contains("active")) refreshOverview(); }, 15000);
showTab((location.hash || "#overview").slice(1));
