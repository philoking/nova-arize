// Nova Voice — "Constellation" PWA client.
// Flow: record → /api/stt → transcript → /api/chat (SSE) → /api/tts → play.
// The capability graph lights the Nova → model → provider → tool path as each
// real tool call streams back from /api/chat, so you watch which capability fires.

const els = {
  // chrome
  historyBtn: document.getElementById("historyBtn"),
  newConvoBtn: document.getElementById("newConvoBtn"),
  healthDiamond: document.getElementById("healthDiamond"),
  novaStatus: document.getElementById("novaStatus"),
  clock: document.getElementById("clock"),
  timerPill: document.getElementById("timerPill"),
  timerRing: document.getElementById("timerRing"),
  timerPillTime: document.getElementById("timerPillTime"),
  timerPillCount: document.getElementById("timerPillCount"),
  devicesBtn: document.getElementById("devicesBtn"),
  devicesBadge: document.getElementById("devicesBadge"),
  notifBtn: document.getElementById("notifBtn"),
  notifBadge: document.getElementById("notifBadge"),
  settingsBtn: document.getElementById("settingsBtn"),
  dashboardBtn: document.getElementById("dashboardBtn"),
  dashboard: document.getElementById("dashboard"),
  dashboardBack: document.getElementById("dashboardBack"),
  dashboardBody: document.getElementById("dashboardBody"),
  dashStatus: document.getElementById("dashStatus"),
  dashEditBtn: document.getElementById("dashEditBtn"),
  dashDone: document.getElementById("dashDone"),
  dashAddGroup: document.getElementById("dashAddGroup"),
  dashReset: document.getElementById("dashReset"),
  stageHud: document.getElementById("stageHud"),
  // thread + command bar
  transcript: document.getElementById("transcript"),
  hint: document.getElementById("hint"),
  commandForm: document.getElementById("commandForm"),
  textInput: document.getElementById("textInput"),
  micBtn: document.getElementById("micBtn"),
  executeBtn: document.getElementById("executeBtn"),
  // graph
  stage: document.getElementById("stage"),
  graphFocus: document.getElementById("graphFocus"),
  graphEdges: document.getElementById("graphEdges"),
  graphNodes: document.getElementById("graphNodes"),
  graphLabels: document.getElementById("graphLabels"),
  focusLabel: document.getElementById("focusLabel"),
  // overlays
  backdrop: document.getElementById("backdrop"),
  convos: document.getElementById("convos"),
  convosClose: document.getElementById("convosClose"),
  newChatBtn: document.getElementById("newChatBtn"),
  convoList: document.getElementById("convoList"),
  convoEmpty: document.getElementById("convoEmpty"),
  timersPop: document.getElementById("timersPop"),
  taskList: document.getElementById("taskList"),
  taskEmpty: document.getElementById("taskEmpty"),
  devicesPop: document.getElementById("devicesPop"),
  satList: document.getElementById("satList"),
  satEmpty: document.getElementById("satEmpty"),
  satCount: document.getElementById("satCount"),
  notifPop: document.getElementById("notifPop"),
  notifList: document.getElementById("notifList"),
  notifEmpty: document.getElementById("notifEmpty"),
  toast: document.getElementById("toast"),
  player: document.getElementById("player"),
  // camera modal (#68)
  cameraModal: document.getElementById("cameraModal"),
  cameraTitle: document.getElementById("cameraTitle"),
  cameraVideo: document.getElementById("cameraVideo"),
  cameraStatus: document.getElementById("cameraStatus"),
  cameraMute: document.getElementById("cameraMute"),
  cameraClose: document.getElementById("cameraClose"),
  // manual PDF viewer modal (#63)
  pdfModal: document.getElementById("pdfModal"),
  pdfFrame: document.getElementById("pdfFrame"),
  pdfTitle: document.getElementById("pdfTitle"),
  pdfOpen: document.getElementById("pdfOpen"),
  pdfClose: document.getElementById("pdfClose"),
  // settings
  settings: document.getElementById("settings"),
  autoEscalate: document.getElementById("autoEscalate"),
  manualAddForm: document.getElementById("manualAddForm"),
  manualFile: document.getElementById("manualFile"),
  manualUpload: document.getElementById("manualUpload"),
  manualTools: document.getElementById("manualTools"),
  manualSearch: document.getElementById("manualSearch"),
  manualSort: document.getElementById("manualSort"),
  manualCount: document.getElementById("manualCount"),
  manualList: document.getElementById("manualList"),
  manualEmpty: document.getElementById("manualEmpty"),
  manualDisabled: document.getElementById("manualDisabled"),
  manualStatus: document.getElementById("manualStatus"),
  settingsBack: document.getElementById("settingsBack"),
  voiceSetting: document.getElementById("voiceSetting"),
  modelSetting: document.getElementById("modelSetting"),
  voicePromptSetting: document.getElementById("voicePromptSetting"),
  voicePromptSave: document.getElementById("voicePromptSave"),
  voicePromptReset: document.getElementById("voicePromptReset"),
  textPromptSetting: document.getElementById("textPromptSetting"),
  textPromptSave: document.getElementById("textPromptSave"),
  textPromptReset: document.getElementById("textPromptReset"),
  settingsStatus: document.getElementById("settingsStatus"),
  notifyEnabled: document.getElementById("notifyEnabled"),
  notifyAddress: document.getElementById("notifyAddress"),
  notifyTopic: document.getElementById("notifyTopic"),
  notifySave: document.getElementById("notifySave"),
  notifyTest: document.getElementById("notifyTest"),
  haRefresh: document.getElementById("haRefresh"),
  haDeviceList: document.getElementById("haDeviceList"),
  haDeviceEmpty: document.getElementById("haDeviceEmpty"),
  memList: document.getElementById("memList"),
  memEmpty: document.getElementById("memEmpty"),
  memStatus: document.getElementById("memStatus"),
  memAddForm: document.getElementById("memAddForm"),
  memText: document.getElementById("memText"),
  memTier: document.getElementById("memTier"),
  logView: document.getElementById("logView"),
  logEmpty: document.getElementById("logEmpty"),
  logLevel: document.getElementById("logLevel"),
  logRefresh: document.getElementById("logRefresh"),
  logClear: document.getElementById("logClear"),
  logOlder: document.getElementById("logOlder"),
  logHideHealth: document.getElementById("logHideHealth"),
  logRetention: document.getElementById("logRetention"),
};

// Conversation turns sent to the LLM: [{role, content}, ...] (render-only).
const history = [];
let conversationId = null;
let recorder = null;
let chunks = [];
let recording = false;
let busy = false;

let MODEL = "qwen3:8b";                 // effective model, from /api/capabilities
let escalationEnabled = false;          // Claude escalation configured (from /api/config)
let metricsEnabled = false;             // live homelab metrics available (drives the constellation HUD)
// The two cloud escalation tiers (from /api/config): `fast` is the reactive brain
// (a struggle → Haiku), `deep` is the upfront brain (research/writing → Opus).
let ESCALATION_MODELS = { fast: "claude-haiku-4-5", deep: "claude-opus-4-8" };
let healthOk = false;
const providerHealth = {};              // provider name → reachable (from /api/health)

// Timers/alarms state (see the timers popover + top-bar pill).
let timers = [];
let clockOffset = 0;
const firedAlerted = new Set();
const timerMaxRemaining = {};           // id → largest remaining seen (for the pill ring)
let audioCtx = null;
let ringingAlarms = [];
let alarmTimer = null;

// Satellites (devices popover).
let lastSatSig = "";
let satOnline = new Set();
let satInitialized = false;
let satEditing = null;   // id of the satellite whose name is being edited (poll won't rebuild the list while set)

// Notifications tray — a live feed of real events observed this session.
let notifications = [];
let notifUnread = 0;

// Overlay bookkeeping.
let openKind = null;
let toastTimer = null;

// ── Boot ─────────────────────────────────────────────────────────────────
init();

async function init() {
  // Wire up every interaction FIRST, synchronously, before any await below.
  // Module scripts are deferred, so the DOM is fully parsed here and the handlers
  // can attach immediately. This must never be gated behind the boot fetches
  // (config/capabilities/health): a slow or hung upstream would otherwise leave
  // the whole UI unclickable until those resolve — dead clicks on open.
  els.micBtn.addEventListener("click", onMicClick);
  els.commandForm.addEventListener("submit", onTextSubmit);
  els.transcript.addEventListener("scroll", updateThreadFade, { passive: true });
  els.textInput.addEventListener("input", autoGrowInput);
  els.textInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); els.commandForm.requestSubmit(); }
  });
  // Top-bar entry points.
  els.historyBtn.addEventListener("click", () => toggleOverlay("convos"));
  els.newConvoBtn.addEventListener("click", newChat);   // start a fresh chat without opening HISTORY (#69)
  els.timerPill.addEventListener("click", () => toggleOverlay("timers"));
  els.devicesBtn.addEventListener("click", () => toggleOverlay("devices"));
  els.notifBtn.addEventListener("click", () => toggleOverlay("notif"));
  els.settingsBtn.addEventListener("click", openSettings);
  els.settingsBack.addEventListener("click", closeSettings);
  els.dashboardBtn.addEventListener("click", openDashboard);
  els.dashboardBack.addEventListener("click", closeDashboard);
  // Hash routing: #dashboard is the source of truth for the dashboard view, so a
  // pasted/bookmarked link opens it and browser back/forward toggle it (#link). The
  // initial reconcile happens after the first await below — running it here (before
  // init's first await) would touch the module-level dashboard `let`s while they're
  // still in their temporal dead zone.
  window.addEventListener("hashchange", syncRoute);
  els.dashEditBtn.addEventListener("click", () => setDashEdit(true));
  els.dashDone.addEventListener("click", () => setDashEdit(false));
  els.dashAddGroup.addEventListener("click", dashAddGroup);
  els.dashReset.addEventListener("click", dashReset);
  els.convosClose.addEventListener("click", closeOverlays);
  els.newChatBtn.addEventListener("click", newChat);
  els.backdrop.addEventListener("click", closeOverlays);
  document.querySelectorAll(".pop-close[data-close]").forEach((b) =>
    b.addEventListener("click", closeOverlays));
  // Camera modal (#68): close via ✕ / backdrop, toggle audio (starts muted so it
  // can autoplay), and close on Escape ahead of the popovers/settings.
  els.cameraClose.addEventListener("click", closeCamera);
  els.cameraModal.querySelector("[data-close-cam]").addEventListener("click", closeCamera);
  els.cameraMute.addEventListener("click", () => {
    const v = els.cameraVideo;
    v.muted = !v.muted;
    els.cameraMute.textContent = v.muted ? "🔇" : "🔊";
    els.cameraMute.setAttribute("aria-label", v.muted ? "Unmute" : "Mute");
    if (!v.muted) v.play().catch(() => {});
  });
  // Manual PDF viewer (#63): close via ✕ / backdrop.
  els.pdfClose.addEventListener("click", closeManual);
  els.pdfModal.querySelector("[data-close-pdf]").addEventListener("click", closeManual);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      if (!els.pdfModal.hidden) closeManual();
      else if (!els.cameraModal.hidden) closeCamera();
      else if (openKind) closeOverlays();
      else if (!els.dashboard.hidden) closeDashboard();
      else if (!els.settings.hidden) closeSettings();
    }
  });

  // Settings wiring (unchanged behaviour).
  els.voiceSetting.addEventListener("change", saveVoice);
  els.modelSetting.addEventListener("change", saveModel);
  els.voicePromptSave.addEventListener("click", () => savePrompt("voice_prompt", els.voicePromptSetting.value));
  els.voicePromptReset.addEventListener("click", () => resetPrompt("voice_prompt"));
  els.textPromptSave.addEventListener("click", () => savePrompt("text_prompt", els.textPromptSetting.value));
  els.textPromptReset.addEventListener("click", () => resetPrompt("text_prompt"));
  els.notifySave.addEventListener("click", saveNotifications);
  els.notifyTest.addEventListener("click", testNotification);
  els.haRefresh.addEventListener("click", () => loadHaDevices(true));
  els.logRefresh.addEventListener("click", loadLog);
  els.logClear.addEventListener("click", clearLog);
  els.logOlder.addEventListener("click", loadOlderLog);
  els.logLevel.addEventListener("change", loadLog);
  els.logHideHealth.addEventListener("change", loadLog);
  document.querySelectorAll(".settings-tab").forEach((btn) =>
    btn.addEventListener("click", () => switchSettingsTab(btn.dataset.tab)));
  els.memAddForm.addEventListener("submit", addMemory);
  els.autoEscalate.addEventListener("change", saveAutoEscalate);
  els.manualAddForm.addEventListener("submit", uploadManual);
  // Library toolbar: filter/sort re-render from cache (no refetch). Sort sticks.
  els.manualSearch.addEventListener("input", applyManualView);
  els.manualSort.addEventListener("change", () => {
    try { localStorage.setItem("manualSort", els.manualSort.value); } catch {}
    applyManualView();
  });
  try {
    const saved = localStorage.getItem("manualSort");
    if (saved) els.manualSort.value = saved;
  } catch {}

  if (!navigator.mediaDevices?.getUserMedia) {
    showToast("Mic needs HTTPS or localhost", "error");
    els.micBtn.disabled = true;
  }
  if ("serviceWorker" in navigator) navigator.serviceWorker.register("/sw.js").catch(() => {});

  // Now load bootstrap data and start the polling loops. These awaits run after
  // the UI is already interactive, so a slow/hung backend never freezes clicks.
  await loadConfig();
  await loadCapabilities();     // builds the capability graph
  syncRoute();                  // honour a deep link (e.g. #dashboard) now that the
                                // graph is built and the dashboard `let`s are live
  await refreshHealth();
  setInterval(refreshHealth, 20000);
  setNovaStatus("idle");
  tickClock();
  setInterval(tickClock, 1000);
  loadTimers();
  setInterval(tickTimers, 1000);
  setInterval(loadTimers, 5000);
  loadSatellites();
  setInterval(loadSatellites, 4000);
  if (metricsEnabled) { loadMetrics(); setInterval(loadMetrics, 12000); }
}

async function loadConfig() {
  try {
    const cfg = await (await fetch("/api/config")).json();
    if (cfg.model) MODEL = cfg.model;
    escalationEnabled = !!cfg.escalation;
    metricsEnabled = !!cfg.metrics;
    if (cfg.escalation_models) ESCALATION_MODELS = cfg.escalation_models;
  } catch { showToast("Couldn't load config", "error"); }
}

// ── Nova activity status (drives the top-bar status text + diamond) ────────
const NOVA_STATUS = {
  idle:      { label: "STANDING BY", color: "var(--muted)" },
  listening: { label: "LISTENING…",  color: "#ff4d5e" },
  thinking:  { label: "PROCESSING",  color: "#ffc857" },
  speaking:  { label: "RESPONDING",  color: "#66b8ff" },
};
let novaState = "idle";
function setNovaStatus(state) {
  novaState = state;
  const s = NOVA_STATUS[state] || NOVA_STATUS.idle;
  if (state === "idle" && !healthOk) {
    els.novaStatus.textContent = "OFFLINE";
    els.novaStatus.style.color = "#ff4d5e";
    return;
  }
  els.novaStatus.textContent = s.label;
  els.novaStatus.style.color = s.color;
}

async function refreshHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    healthOk = !!data.ok;
    for (const [k, v] of Object.entries(data.services || {})) providerHealth[k] = v;
    els.healthDiamond.classList.toggle("ok", healthOk);
    els.healthDiamond.classList.toggle("bad", !healthOk);
    els.healthDiamond.title = "stt:" + data.services.stt + " llm:" + data.services.llm + " tts:" + data.services.tts;
  } catch {
    healthOk = false;
    els.healthDiamond.classList.remove("ok");
    els.healthDiamond.classList.add("bad");
    els.healthDiamond.title = "services unreachable";
  }
  if (novaState === "idle") setNovaStatus("idle");
  renderGraph();
}

// ── Capability graph ───────────────────────────────────────────────────────
// The graph is authored in an 820×780 user space. Layout (positions/colours) is a
// deterministic design constant; the *contents* (which providers exist, whether
// they're enabled, CORE/GATED, and their tool leaves) come from /api/capabilities.
const C = { x: 410, y: 372 };
// Provider ring — angle°, radius, hue, display label. Iterated in THIS order so
// the seeded leaf scatter is reproducible (matches the design reference).
const GRAPH_LAYOUT = {
  timers:   { ang: -92, rad: 205, color: "#7fa8cc", label: "timers" },
  math:     { ang: -50, rad: 200, color: "#79b3ac", label: "math" },
  memory:   { ang: -12, rad: 215, color: "#9a92bd", label: "memory" },
  home:     { ang:  30, rad: 225, color: "#84b39c", label: "home_assistant" },
  web:      { ang:  68, rad: 210, color: "#c090a0", label: "web_search" },
  notes:    { ang: 108, rad: 220, color: "#c0a878", label: "notes" },
  lists:    { ang: 150, rad: 228, color: "#83aab6", label: "lists" },
  tasks:    { ang: 190, rad: 208, color: "#909cc4", label: "tasks" },
  projects: { ang: 220, rad: 222, color: "#b394c0", label: "projects" },
  manuals:  { ang: 243, rad: 256, color: "#9bb0c0", label: "manuals" },
  cameras:  { ang: 300, rad: 218, color: "#79b8c4", label: "cameras" },
};
const GRAPH_ORDER = ["timers", "math", "memory", "home", "web", "notes", "lists", "tasks", "projects", "manuals", "cameras"];

let graph = null;                       // { nodes, edges, providerIds }
const fnToCluster = {};                 // bare tool name → provider/skill id
const graphState = { route: null, ambient: null };
// Respect the OS "reduce motion" setting — skip the idle constellation twinkle.
const REDUCE_MOTION = !!(window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches);
let windTimer = null;                   // pending zoom-back-out after a turn
const ROUTE_HOLD_MS = 5000;             // hold the last tool lit + focused, then zoom out
const MODEL_COLOR = "#8fd0ff";          // the model node's hue for the local model
const CLAUDE_DEEP_COLOR = "#c9a2ff";    // Opus — the deep/upfront escalation brain (matches the ✦ deep badge)
const CLAUDE_FAST_COLOR = "#e6a8d8";    // Haiku — the fast/cheap reactive escalation brain
const SVGNS = "http://www.w3.org/2000/svg";

function hexA(hex, a) {
  const h = hex.replace("#", "");
  return `rgba(${parseInt(h.slice(0, 2), 16)},${parseInt(h.slice(2, 4), 16)},${parseInt(h.slice(4, 6), 16)},${a})`;
}

async function loadCapabilities() {
  let caps;
  try {
    caps = await (await fetch("/api/capabilities")).json();
  } catch {
    caps = { model: MODEL, skills: [] };
  }
  if (caps.model) MODEL = caps.model;
  buildGraph(caps);
}

function buildGraph(caps) {
  const bySkill = {};
  for (const s of caps.skills || []) bySkill[s.name] = s;

  // Same LCG + seed as the design reference, so leaf positions are reproducible.
  let rng = 20260712;
  const rand = () => { rng = (rng * 1103515245 + 12345) & 0x7fffffff; return rng / 0x7fffffff; };

  const nodes = {};
  nodes.nova = { id: "nova", x: C.x, y: C.y, r: 24, tier: "nova", label: "NOVA", color: "#ffcf4a", live: true };
  nodes.model = { id: "model", x: C.x + 58, y: C.y - 96, r: 14, tier: "model", label: MODEL, color: MODEL_COLOR, live: true };
  // Claude — the off-LAN escalation brains, one tier up from the local model. Two
  // lanes: a FAST/cheap node (reactive — the local model struggled) and a DEEP node
  // (upfront — research/writing). Both dashed (`cloud`) and muted until a turn
  // lights the one it used; `live` reflects whether escalation is configured at all.
  const shortModel = (id) => (id || "").replace(/^claude-/, "").split("-")[0] || "claude";
  nodes.claudeFast = { id: "claudeFast", x: C.x + 22, y: C.y - 166, r: 11, tier: "model",
                       label: shortModel(ESCALATION_MODELS.fast), color: CLAUDE_FAST_COLOR,
                       cloud: true, live: escalationEnabled };
  nodes.claudeDeep = { id: "claudeDeep", x: C.x + 96, y: C.y - 166, r: 12, tier: "model",
                       label: shortModel(ESCALATION_MODELS.deep), color: CLAUDE_DEEP_COLOR,
                       cloud: true, live: escalationEnabled };
  const edges = [{ a: "nova", b: "model" },
                 { a: "model", b: "claudeFast" }, { a: "model", b: "claudeDeep" }];
  const providerIds = [];

  for (const id of GRAPH_ORDER) {
    const lay = GRAPH_LAYOUT[id];
    const sk = bySkill[id];
    if (!lay || !sk) continue;
    const a = (lay.ang * Math.PI) / 180;
    const px = C.x + lay.rad * Math.cos(a);
    const py = C.y + lay.rad * Math.sin(a);
    const live = sk.enabled && (sk.kind === "CORE" || providerHealth[sk.provider] !== false);
    providerIds.push(id);
    nodes[id] = { id, x: px, y: py, r: 12, tier: "provider", kind: sk.kind, label: lay.label,
                  color: lay.color, provider: sk.provider, live };
    edges.push({ a: "model", b: id });
    // Fan the tool leaves evenly on the provider's outward side (away from the
    // hub), over a wider arc for bigger clusters, and alternate them between two
    // radii — so neighbouring leaves and their labels sit at different angles AND
    // distances instead of stacking on top of each other.
    const fns = sk.functions || [];
    const spread = Math.min(2.1, 0.6 + fns.length * 0.3);      // total arc (rad)
    fns.forEach((fn, i) => {
      const t = fns.length > 1 ? (i / (fns.length - 1) - 0.5) : 0;  // -0.5 … 0.5
      const la = a + t * spread;
      const lr = 54 + (i % 2) * 34;                            // two rings: 54 / 88
      const lx = px + lr * Math.cos(la);
      const ly = py + lr * Math.sin(la);
      const fid = id + ":" + fn;
      // A slow, staggered idle twinkle so the tool leaves shimmer like distant
      // stars at rest. Duration + a negative delay (start mid-phase) come from the
      // seeded RNG, so each leaf drifts out of sync with its neighbours but stays
      // stable across renders. Applied only at rest (see renderGraph).
      nodes[fid] = { id: fid, x: lx, y: ly, r: 5.5, tier: "fn", kind: sk.kind, label: fn,
                     color: lay.color, provider: sk.provider, live,
                     tw: (4.5 + rand() * 3).toFixed(2), twd: (-rand() * 7).toFixed(2) };
      edges.push({ a: id, b: fid });
      if (!(fn in fnToCluster)) fnToCluster[fn] = id;   // first cluster wins; capture disambiguated below
    });
  }

  // Idle-float parameters: each node gently drifts on its own slow elliptical
  // orbit (independent x/y frequency + phase → organic, non-circular motion),
  // driven by the seeded RNG so it's stable. Amplitude grows outward — the hub
  // stays anchored, the outer tool-leaves wander most (a subtle parallax). See
  // the idleFloat() loop, which only runs at rest.
  for (const id in nodes) {
    const n = nodes[id];
    const amp = id === "nova" ? 2.5
      : n.tier === "fn" ? 9.0
      : n.tier === "provider" ? 6.0
      : 4.0;                                   // model / claude
    n.dAx = amp * (0.75 + rand() * 0.5);
    n.dAy = amp * (0.75 + rand() * 0.5);
    n.dWx = 0.26 + rand() * 0.4;               // rad/s — periods ~9–24s
    n.dWy = 0.26 + rand() * 0.4;
    n.dPx = rand() * Math.PI * 2;
    n.dPy = rand() * Math.PI * 2;
  }

  graph = { nodes, edges, providerIds };
  renderGraphDom();
  els.stage.setAttribute("aria-hidden", "false");
  startIdleFloat();
}

// Create the SVG circles/paths + HTML labels once; renderGraph() only mutates styles.
function renderGraphDom() {
  els.graphEdges.innerHTML = "";
  els.graphNodes.innerHTML = "";
  els.graphLabels.innerHTML = "";

  for (const e of graph.edges) {
    const n1 = graph.nodes[e.a], n2 = graph.nodes[e.b];
    const mx = (n1.x + n2.x) / 2, my = (n1.y + n2.y) / 2;
    const dx = n2.x - n1.x, dy = n2.y - n1.y;
    const cx = mx - dy * 0.13, cy = my + dx * 0.13;
    const path = document.createElementNS(SVGNS, "path");
    path.setAttribute("d", `M${Math.round(n1.x)} ${Math.round(n1.y)} Q${Math.round(cx)} ${Math.round(cy)} ${Math.round(n2.x)} ${Math.round(n2.y)}`);
    path.setAttribute("class", "edge-path");
    e.el = path;
    els.graphEdges.appendChild(path);
  }

  for (const id of Object.keys(graph.nodes)) {
    const n = graph.nodes[id];
    // Each node is a hollow glass "ring" plus a bright inner "core" point.
    const ring = document.createElementNS(SVGNS, "circle");
    ring.setAttribute("cx", Math.round(n.x));
    ring.setAttribute("cy", Math.round(n.y));
    ring.setAttribute("r", n.r);
    ring.setAttribute("fill", "none");
    ring.setAttribute("class", "node-ring" + (n.tier === "provider" ? " clickable" : ""));
    const core = document.createElementNS(SVGNS, "circle");
    core.setAttribute("cx", Math.round(n.x));
    core.setAttribute("cy", Math.round(n.y));
    core.setAttribute("class", "node-core" + (n.tier === "provider" ? " clickable" : ""));
    if (n.tier === "provider") {
      ring.addEventListener("click", () => inspectProvider(id));
      core.addEventListener("click", () => inspectProvider(id));
    }
    n.ring = ring; n.core = core;
    els.graphNodes.appendChild(ring);
    els.graphNodes.appendChild(core);

    const label = document.createElement("div");
    label.className = "node-label";
    label.textContent = n.label;
    label.style.left = ((n.x / 820) * 100).toFixed(2) + "%";
    n.labelEl = label;
    els.graphLabels.appendChild(label);
  }
  renderGraph();
}

function renderGraph() {
  if (!graph) return;
  const route = graphState.route;
  const path = route ? route.path : null;
  const focusId = route ? route.focus : null;
  const amb = route ? null : graphState.ambient;
  const inPath = (id) => path && path.indexOf(id) !== -1;

  // When a route is lit, inactive clusters ease radially outward from the hub so
  // the active path keeps the centre and stays readable — the whole system stays
  // in view (context), it just makes room. `PUSH` is how far out they drift.
  const PUSH = 1.2;
  const hub = graph.nodes.nova;
  const cx0 = hub ? hub.x : 410, cy0 = hub ? hub.y : 390;

  for (const id of Object.keys(graph.nodes)) {
    const n = graph.nodes[id];
    const on = inPath(id);
    const ambientOn = amb && id === amb;
    const dimmed = route && !on;
    // Displaced position for this render: active/at-rest nodes hold their spot,
    // inactive nodes on a live route push outward. Edges read the same _dx/_dy.
    const ndx = (route && !on) ? cx0 + (n.x - cx0) * PUSH : n.x;
    const ndy = (route && !on) ? cy0 + (n.y - cy0) * PUSH : n.y;
    n._dx = ndx; n._dy = ndy;
    const gated = n.kind === "GATED";
    let r = n.r, op = 1, filter = "none", sw = 1.3, anim = "none", dash = "none";
    let ringFill, stroke, coreFill, coreScale = 0.34, coreOp = 1;
    let labelFill, labelOp, fs, labelWeight = 500;

    // Idle base by tier — a translucent glass ring + a bright energy core, with a
    // soft bloom on the larger nodes. Config-gated providers get a dashed ring.
    if (n.tier === "nova") {
      ringFill = hexA(n.color, 0.16); stroke = "rgba(255,225,150,0.9)"; sw = 1.8;
      coreOp = 0;                                 // no core dot — the NOVA wordmark fills the hub
      filter = "drop-shadow(0 0 8px " + hexA(n.color, 0.55) + ")";
      labelFill = "#ffe9ad"; labelWeight = 700; labelOp = 1; fs = 12;
    } else if (n.tier === "model") {
      ringFill = hexA(n.color, n.cloud ? 0.05 : 0.10); stroke = hexA(n.color, n.cloud ? 0.6 : 0.85); sw = 1.5;
      if (n.cloud) dash = "3.5 3.5";               // off-LAN escalation brain — dashed like a gated node
      coreFill = hexA(n.color, 0.95); coreScale = 0.36;
      filter = "drop-shadow(0 0 6px " + hexA(n.color, 0.4) + ")";
      labelFill = hexA(n.color, 0.95); labelOp = 1; fs = 10.5;
    } else if (n.tier === "provider") {
      ringFill = gated ? hexA(n.color, 0.04) : hexA(n.color, 0.10);
      stroke = hexA(n.color, gated ? 0.6 : 0.82); sw = 1.5;
      dash = gated ? "3.5 3.5" : "none";
      coreFill = hexA(n.color, gated ? 0.55 : 0.9); coreScale = 0.32;
      filter = "drop-shadow(0 0 6px " + hexA(n.color, 0.32) + ")";
      labelFill = hexA(n.color, 0.95); labelOp = 1; fs = 10.5; labelWeight = 600;
    } else { // fn leaf
      ringFill = gated ? hexA(n.color, 0.03) : hexA(n.color, 0.06);
      stroke = hexA(n.color, gated ? 0.4 : 0.55); sw = 1;
      coreFill = hexA(n.color, gated ? 0.45 : 0.72); coreScale = 0.44;
      labelFill = hexA(n.color, 0.62); labelOp = 0.72; fs = 8;
    }

    if (on) {
      // Slightly larger, not dominating — the old 2–3.4× multipliers swamped the
      // view; the active path now reads as the focus by colour + a gentle bump.
      r = n.r * (n.tier === "nova" ? 1.1 : n.tier === "fn" ? 1.8 : n.tier === "model" ? 1.35 : 1.45);
      stroke = "#ffffff"; sw = 2.5; op = 1; dash = "none";
      ringFill = n.tier === "nova" ? hexA(n.color, 0.34) : hexA(n.color, 0.2);
      filter = n.tier === "nova"
        ? "drop-shadow(0 0 12px rgba(255,207,74,0.9))"
        : "drop-shadow(0 0 10px " + hexA(n.color, 0.95) + ")";
      coreFill = n.tier === "nova" ? "#fff3cf" : "#ffe9ad"; coreScale = 0.42;
      labelFill = "#ffe9ad"; labelWeight = 700; labelOp = 1;
      fs = n.tier === "fn" ? 11 : n.tier === "provider" ? 13 : Math.max(fs, 12);
    } else if (ambientOn) {
      r = n.r * 1.2; sw = 1.7; stroke = hexA(n.color, 0.95); dash = "none";
      ringFill = hexA(n.color, 0.16);
      filter = "drop-shadow(0 0 9px " + hexA(n.color, 0.6) + ")";
      coreFill = hexA(n.color, 1); coreScale = 0.4;
      labelOp = 1; labelFill = hexA(n.color, 1);
    } else if (dimmed) {
      // Pushed out of the way but still legible — context, not erased.
      op = 0.5; coreOp = coreOp && 0.5; filter = "none"; labelOp = labelOp ? 0.32 : 0;
    } else if (n.tier === "nova" && !route) {
      anim = "nvNova 3s ease-in-out infinite";
    } else if (!n.live) {
      // Disabled / disconnected gated provider — present but muted.
      op = 0.42; coreOp = coreOp && 0.42; filter = "none"; labelOp = Math.min(labelOp, 0.42);
    }

    // Idle twinkle: a slow, staggered opacity+glow breath on the WHOLE tool leaf
    // (ring and core — the ring is the visible part, so animating only the 2px core
    // read as nothing). Only tool leaves, only at rest, only when live and not being
    // hovered — a lit provider/leaf still signals real activity (#71), so this stays
    // decorative and never touches the hub, model, or provider nodes.
    const twinkle = (n.tier === "fn" && !route && n.live && !ambientOn && !REDUCE_MOTION)
      ? `nvTwinkle ${n.tw}s ease-in-out ${n.twd}s infinite` : null;
    const twGlow = twinkle ? hexA(n.color, 0.9) : null;

    const shift = `translate(${(ndx - n.x).toFixed(1)}px, ${(ndy - n.y).toFixed(1)}px)`;
    const ring = n.ring.style;
    ring.setProperty("r", r + "px");
    n.ring.setAttribute("r", r);            // fallback for engines without CSS `r`
    ring.fill = ringFill; ring.stroke = stroke; ring.strokeWidth = sw + "px";
    ring.strokeDasharray = dash; ring.opacity = op; ring.filter = filter;
    ring.animation = twinkle || anim;
    if (twGlow) ring.setProperty("--tw-glow", twGlow);
    ring.transform = shift;

    const coreR = Math.round(r * coreScale * 10) / 10;
    const core = n.core.style;
    core.setProperty("r", coreR + "px");
    n.core.setAttribute("r", coreR);
    core.fill = coreFill || "none"; core.opacity = coreOp;
    core.transform = shift;
    core.animation = twinkle || "none";
    if (twGlow) core.setProperty("--tw-glow", twGlow);

    const L = n.labelEl.style;
    if (n.tier === "fn") {
      // Push each tool label outward from the hub and anchor it by direction, so
      // labels fan into the empty periphery and read away from the cluster centre
      // instead of stacking above their dots.
      const ux = ndx - cx0, uy = ndy - cy0, um = Math.hypot(ux, uy) || 1;
      const ox = ux / um, oy = uy / um;
      L.left = (((ndx + ox * (r + 9)) / 820) * 100).toFixed(2) + "%";
      L.top = (((ndy + oy * (r + 9)) / 780) * 100).toFixed(2) + "%";
      const hx = ox > 0.35 ? "0%" : ox < -0.35 ? "-100%" : "-50%";
      const vy = oy > 0.35 ? "0%" : oy < -0.35 ? "-100%" : "-50%";
      n._labelAnchor = `translate(${hx}, ${vy})`;
      L.textAlign = ox > 0.35 ? "left" : ox < -0.35 ? "right" : "center";
    } else {
      L.left = ((ndx / 820) * 100).toFixed(2) + "%";
      const labelY = n.tier === "nova" ? ndy : ndy - r - 7;
      L.top = ((labelY / 780) * 100).toFixed(2) + "%";
      n._labelAnchor = n.tier === "nova" ? "translate(-50%, -50%)" : "translate(-50%, -100%)";
      L.textAlign = "center";
    }
    L.transform = n._labelAnchor + ((!route && n._fx) ? ` translate(${n._fx.toFixed(2)}px, ${n._fy.toFixed(2)}px)` : "");
    L.fontSize = fs + "px";
    L.fontWeight = labelWeight;
    L.color = labelFill;
    L.opacity = labelOp;
  }

  for (const e of graph.edges) {
    const na = graph.nodes[e.a], nb = graph.nodes[e.b];
    // Redraw the wire between the (possibly displaced) endpoints so it stays
    // connected as inactive clusters drift out — same curve as renderGraphDom.
    const x1 = na._dx, y1 = na._dy, x2 = nb._dx, y2 = nb._dy;
    const emx = (x1 + x2) / 2, emy = (y1 + y2) / 2, edx = x2 - x1, edy = y2 - y1;
    const qx = emx - edy * 0.13, qy = emy + edx * 0.13;
    e.el.setAttribute("d", `M${x1.toFixed(1)} ${y1.toFixed(1)} Q${qx.toFixed(1)} ${qy.toFixed(1)} ${x2.toFixed(1)} ${y2.toFixed(1)}`);
    const on = inPath(e.a) && inPath(e.b);
    const dimmed = route && !on;
    const isFn = na.tier === "fn" || nb.tier === "fn";
    const s = e.el.style;
    // At rest the whole constellation is faintly wired together (provider spokes a
    // touch brighter than the tool-leaf threads); an active edge lights gold, flows
    // toward its target, and gains a glow; off-path edges recede.
    s.stroke = on ? "#ffd772" : "rgba(150,185,230,0.9)";
    s.strokeWidth = (on ? 2 : isFn ? 0.8 : 1.2) + "px";
    s.opacity = on ? 1 : dimmed ? 0.12 : isFn ? 0.16 : 0.26;
    s.strokeDasharray = on ? "3 5" : "none";
    s.animation = on ? "nvFlow 0.6s linear infinite" : "none";
    s.filter = on ? "drop-shadow(0 0 3px rgba(255,215,114,0.75))" : "none";
  }

  // No camera zoom: the active path holds the centre while inactive clusters push
  // outward (above), so the whole system stays framed. (Previously scaled 1.42
  // toward the focused node, which lunged into a corner and hid the rest.)
  const f = graph.nodes[focusId];
  els.graphFocus.style.transformOrigin = "50% 50%";
  els.graphFocus.style.transform = "scale(1)";

  if (focusId && f) {
    els.focusLabel.textContent = "FOCUS ▸ " + (f.label || focusId).toUpperCase();
    els.focusLabel.classList.add("active");
  } else {
    const provs = graph.providerIds.length;
    const tools = Object.values(graph.nodes).filter((n) => n.tier === "fn").length;
    els.focusLabel.textContent = `CAPABILITY GRAPH · ${provs} PROVIDERS · ${tools} TOOLS`;
    els.focusLabel.classList.remove("active");
  }
}

function setRoute(route) { graphState.route = route; renderGraph(); }
function clearRoute() { graphState.route = null; renderGraph(); }

// Idle float: at rest, gently drift every node along its own slow orbit and reflow
// the wires so the whole constellation feels alive — actual movement, on top of the
// twinkle. Driven by requestAnimationFrame; it only touches the graph when idle (no
// active route, no camera modal, graph visible). A turn's route animation or the OS
// reduce-motion setting freezes it. The 0.5–0.6s CSS transitions on the nodes/edges
// smooth the per-frame updates, and since node transform and edge `d` share the same
// timing, each wire stays attached to its dot.
let floatRAF = null;
function startIdleFloat() {
  if (floatRAF == null && !REDUCE_MOTION) floatRAF = requestAnimationFrame(idleFloat);
}
function idleFloat(ts) {
  floatRAF = requestAnimationFrame(idleFloat);
  if (!graph || graphState.route || !els.cameraModal.hidden) return;   // only at idle rest
  if (els.stage && els.stage.offsetParent === null) return;            // graph hidden (phone)
  const t = ts / 1000;
  for (const id in graph.nodes) {
    const n = graph.nodes[id];
    if (!n.ring) continue;
    const dx = n.dAx * Math.sin(n.dWx * t + n.dPx);
    const dy = n.dAy * Math.sin(n.dWy * t + n.dPy);
    n._fx = dx; n._fy = dy;
    const tf = `translate(${dx.toFixed(2)}px, ${dy.toFixed(2)}px)`;
    n.ring.style.transform = tf;
    n.core.style.transform = tf;
    if (n._labelAnchor) n.labelEl.style.transform = `${n._labelAnchor} translate(${dx.toFixed(2)}px, ${dy.toFixed(2)}px)`;
  }
  for (const e of graph.edges) {
    const a = graph.nodes[e.a], b = graph.nodes[e.b];
    const x1 = a.x + (a._fx || 0), y1 = a.y + (a._fy || 0);
    const x2 = b.x + (b._fx || 0), y2 = b.y + (b._fy || 0);
    const mx = (x1 + x2) / 2, my = (y1 + y2) / 2, ddx = x2 - x1, ddy = y2 - y1;
    const qx = mx - ddy * 0.13, qy = my + ddx * 0.13;
    e.el.setAttribute("d", `M${x1.toFixed(1)} ${y1.toFixed(1)} Q${qx.toFixed(1)} ${qy.toFixed(1)} ${x2.toFixed(1)} ${y2.toFixed(1)}`);
  }
}

// Relabel the local model node when the user switches models in Settings.
function setModelNode(label, color, render = true) {
  if (!graph) { if (render) renderGraph(); return; }
  const m = graph.nodes.model;
  m.label = label; m.color = color; m.labelEl.textContent = label;
  if (render) renderGraph();
}

// Which provider cluster owns a bare tool name (capture is shared → use args).
function resolveCluster(tool, args) {
  if (tool === "capture" && args && args.section === "tasks") return "tasks";
  return fnToCluster[tool] || null;
}

// Clicking a provider node briefly inspects it (a user-initiated highlight). The
// graph is otherwise quiet at rest — only a real tool call lights a provider (#71).
let inspectTimer = null;
function inspectProvider(id) {
  if (graphState.route || !graph.nodes[id]?.live) return;
  graphState.ambient = id;
  renderGraph();
  clearTimeout(inspectTimer);
  inspectTimer = setTimeout(() => { if (graphState.ambient === id) { graphState.ambient = null; renderGraph(); } }, 1600);
}

// ── Recording ────────────────────────────────────────────────────────────
function pickMime() {
  const prefs = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg"];
  return prefs.find((m) => window.MediaRecorder?.isTypeSupported(m)) || "";
}

async function onMicClick() {
  if (busy) return;
  if (recording) { stopRecording(); return; }
  stopPlayback();
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch { showToast("Microphone permission denied", "error"); return; }

  const mimeType = pickMime();
  recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
  chunks = [];
  recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
  recorder.onstop = () => {
    stream.getTracks().forEach((t) => t.stop());
    handleClip(new Blob(chunks, { type: recorder.mimeType || "audio/webm" }));
  };
  recorder.start();
  recording = true;
  els.micBtn.classList.add("recording");
  setNovaStatus("listening");
}

function stopRecording() {
  if (recorder && recording) recorder.stop();
  recording = false;
  els.micBtn.classList.remove("recording");
}

async function handleClip(blob) {
  if (!blob.size) { setNovaStatus("idle"); return; }
  setBusy(true);
  setNovaStatus("thinking");
  try {
    const fd = new FormData();
    fd.append("audio", blob, "clip.webm");
    const res = await fetch("/api/stt", { method: "POST", body: fd });
    if (!res.ok) throw new Error("stt " + res.status);
    const { text } = await res.json();
    const clean = (text || "").trim();
    if (!clean) { showToast("Didn't catch that — try again"); setNovaStatus("idle"); setBusy(false); return; }
    await converse(clean, { spoken: true });
  } catch (err) {
    showToast("Transcription failed: " + err.message, "error");
    setNovaStatus("idle");
    setBusy(false);
  }
}

// ── Text input ───────────────────────────────────────────────────────────
async function onTextSubmit(e) {
  e.preventDefault();
  const text = els.textInput.value.trim();
  if (!text || busy) return;
  els.textInput.value = "";
  autoGrowInput();
  await converse(text, { spoken: false });
}

function autoGrowInput() {
  const el = els.textInput;
  el.style.height = "auto";
  const borders = el.offsetHeight - el.clientHeight;
  el.style.height = Math.min(el.scrollHeight + borders, 120) + "px";
}

// ── Conversation loop ────────────────────────────────────────────────────
async function converse(userText, { spoken = false } = {}) {
  setBusy(true);
  const started = performance.now();
  addMessage("user", userText);
  history.push({ role: "user", content: userText });

  const bubble = addMessage("assistant", "");
  setNovaStatus("thinking");
  let reply = "";
  let firstDelta = true;
  let lastCluster = null;
  let lastLeaf = null;
  let escalatedNode = null;  // null | "claudeFast" | "claudeDeep" once a turn escalates
  let modelBadged = false;
  const routeTimers = [];
  clearTimeout(windTimer);   // cancel a pending zoom-out from the previous turn

  // Build the lit path for a tool (or the bare model tier), threading the exact
  // cloud node in once the turn has escalated so that node + edge light up.
  const routeFor = (cluster, leaf) => {
    // Keep the hub lit so the gold path runs Nova → model → provider → tool from
    // the centre outward (rather than floating detached above a dimmed Nova).
    const p = escalatedNode ? ["nova", "model", escalatedNode] : ["nova", "model"];
    if (cluster) { p.push(cluster); if (leaf) p.push(leaf); }
    return p;
  };
  const relight = () => {
    if (lastCluster) setRoute({ path: routeFor(lastCluster, lastLeaf), focus: lastCluster });
    else setRoute({ path: escalatedNode ? ["nova", "model", escalatedNode] : ["nova", "model"],
                    focus: escalatedNode || "model" });
  };

  // A new command resets the graph: light Nova, then focus the model as it "parses".
  setRoute({ path: ["nova"], focus: null });
  const modelTimer = setTimeout(() => { if (busy && !lastCluster) relight(); }, 350);
  routeTimers.push(modelTimer);

  try {
    await streamSSE(
      "/api/chat",
      { message: userText, conversation_id: conversationId, source: "web", mode: spoken ? "voice" : "text" },
      (delta) => {
        // First reply token: if no tool fired this turn, settle focus on the model
        // (or Claude, if the turn escalated).
        if (firstDelta) { firstDelta = false; if (!lastCluster) relight(); }
        reply += delta;
        bubble.textContent = reply;
        scrollDown();
      },
      (tool) => {
        // Real tool call → light model → provider → the exact tool leaf all at once
        // (a fast turn would clear before a staged leaf step could show), and
        // zoom-focus the provider so the fired capability is clearly visible.
        const cluster = resolveCluster(tool.tool, tool.args);
        if (cluster && graph.nodes[cluster]) {
          clearTimeout(modelTimer);
          lastCluster = cluster;
          const leafId = cluster + ":" + tool.tool;
          lastLeaf = graph.nodes[leafId] ? leafId : null;
          setRoute({ path: routeFor(cluster, lastLeaf), focus: cluster });
        }
        addToolChip(bubble, tool);
        onToolSideEffects(tool);
      },
      (meta) => {
        if (meta.conversation) conversationId = meta.conversation;
        // Manuals Nova answered from → clickable chips that open the PDF at the page.
        if (meta.sources) renderSources(bubble, meta.sources);
        // The local model handed off to a cloud brain (#65). meta.tier says which:
        // "fast" (Haiku, reactive struggle) or "deep" (Opus, upfront) — light that
        // exact node + edge so the route shows which brain drove the tools.
        if (meta.escalated && !escalatedNode) {
          escalatedNode = meta.tier === "deep" ? "claudeDeep" : "claudeFast";
          clearTimeout(modelTimer);
          showToast(meta.tier === "deep" ? "✦ Escalating to Claude (deep)…" : "✦ Escalating to Claude (fast)…");
          relight();
        }
        // Which model produced this reply (sent once near the end) → badge it.
        if (meta.model) {
          setModelBadge(bubble, meta.model);
          modelBadged = true;
          // Safety net: if a Claude model is reported but no escalation event lit a
          // node, infer the tier from the model id (haiku ⇒ fast, else deep).
          if (!escalatedNode && /claude/i.test(meta.model)) {
            escalatedNode = /haiku/i.test(meta.model) ? "claudeFast" : "claudeDeep";
            relight();
          }
        }
      },
    );
  } catch (err) {
    routeTimers.forEach(clearTimeout);
    showToast("Chat failed: " + err.message, "error");
    clearRoute();   // the turn failed — return the graph to its quiet rest state
    setNovaStatus("idle");
    setBusy(false);
    return;
  }

  routeTimers.forEach(clearTimeout);
  const elapsed = ((performance.now() - started) / 1000).toFixed(1);
  bubble.metaEl.textContent = `${fmtClock(new Date())} · ${elapsed}s`;
  reply = reply.trim();
  // The bubble streamed as plain text; format its markdown now the turn is done.
  if (reply) renderMarkdown(bubble, reply);
  // Fall back to badging the local model if the backend sent no model event.
  if (!modelBadged) setModelBadge(bubble, MODEL);
  history.push({ role: "assistant", content: reply });

  if (reply && spoken) await speak(reply, bubble);
  setNovaStatus("idle");
  setBusy(false);
  loadConvoList(true);
  // Hold the last tool/provider lit + zoom-focused so it's readable, then zoom the
  // graph back out to its quiet rest state. The next command cancels this and
  // starts a fresh animation.
  clearTimeout(windTimer);
  windTimer = setTimeout(clearRoute, ROUTE_HOLD_MS);
}

// Side effects of a tool call worth surfacing outside the graph. Tool actions the
// user just performed aren't pushed to the notifications tray — the reply and its
// tool chip already show them; the tray is for async events (fired timers,
// satellite reconnects) you'd otherwise miss.
function onToolSideEffects(tool) {
  const { icon, text } = toolInfo(tool);
  if (/timer|reminder|alarm/.test(tool.tool)) showToast(`${icon} ${text}`);
  // "Pull up the front porch" → open the live feed in a modal (#68).
  if (tool.tool === "show_camera") openCamera(tool.args && tool.args.camera);
}

// ── Camera feed modal (#68) ─────────────────────────────────────────────────
// A show_camera tool event opens a live feed in a centered modal. The video is
// HLS from Frigate's go2rtc, proxied same-origin via /api/cameras — native HLS on
// Safari/iOS, hls.js (vendored, lazily loaded on first use) everywhere else.
let cameraList = null;      // cached [{id,name}] from /api/cameras
let hls = null;             // active hls.js instance, torn down on close
let hlsLibPromise = null;   // memoized lazy-load of the vendored hls.js
let camWatchdog = null;     // interval that catches a frozen picture and rebuilds
let camLastTime = -1;       // last observed video.currentTime (progress check)
let camStalls = 0;          // consecutive watchdog ticks with no progress
let camRetries = 0;         // consecutive stream rebuilds (bounded before giving up)

async function ensureCameraList() {
  if (cameraList) return cameraList;
  try {
    const data = await (await fetch("/api/cameras")).json();
    cameraList = Array.isArray(data.cameras) ? data.cameras : [];
  } catch { cameraList = []; }
  return cameraList;
}

// Map the model's `camera` argument to a known camera: exact id first, then a
// best token-overlap match on name+id, so a spoken "front porch" still resolves.
function resolveCamera(arg, list) {
  const a = String(arg || "").trim().toLowerCase();
  if (!a) return null;
  const exact = list.find((c) => c.id.toLowerCase() === a);
  if (exact) return exact;
  const toks = (s) => new Set(String(s).toLowerCase().split(/[^a-z0-9]+/).filter((w) => w.length > 1));
  const q = toks(a);
  let best = null, bestN = 0;
  for (const c of list) {
    const hay = toks(c.name + " " + c.id);
    let n = 0; q.forEach((t) => { if (hay.has(t)) n++; });
    if (n > bestN) { best = c; bestN = n; }
  }
  return best;
}

function loadHlsLib() {
  if (window.Hls) return Promise.resolve(window.Hls);
  if (hlsLibPromise) return hlsLibPromise;
  hlsLibPromise = new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = "/vendor/hls.light.min.js";
    s.onload = () => resolve(window.Hls);
    s.onerror = () => reject(new Error("player failed to load"));
    document.head.appendChild(s);
  });
  return hlsLibPromise;
}

async function openCamera(arg) {
  const list = await ensureCameraList();
  const cam = resolveCamera(arg, list);
  if (!cam) { showToast("No camera matches that", "error"); return; }
  closeOverlays();
  els.cameraTitle.textContent = cam.name;
  setCameraStatus("");
  const v = els.cameraVideo;
  v.muted = true;
  els.cameraMute.textContent = "🔇";
  els.cameraMute.setAttribute("aria-label", "Unmute");
  els.cameraModal.hidden = false;
  camRetries = 0;              // fresh open: reset the watchdog's give-up counter
  startStream(cam);
}

function startStream(cam) {
  const v = els.cameraVideo;
  const src = `/api/cameras/${encodeURIComponent(cam.id)}/index.m3u8`;
  v.poster = `/api/cameras/${encodeURIComponent(cam.id)}/snapshot.jpg`;
  teardownHls();
  // Native HLS (Safari / iOS) tolerates go2rtc's tiny live window on its own.
  if (v.canPlayType("application/vnd.apple.mpegurl")) {
    v.src = src;
    v.play().catch(() => {});
    startWatchdog(cam);
    return;
  }
  loadHlsLib().then((Hls) => {
    if (els.cameraModal.hidden) return;            // closed while the lib loaded
    if (Hls && Hls.isSupported()) {
      // go2rtc only advertises ~1s of live HLS (2×0.5s segments) and evicts older
      // ones immediately, so hls.js's default 3-segment sync target can never be
      // met and it stalls on Chrome. Pin playback to the live edge and recover
      // aggressively from the dropped-segment errors a tiny window produces.
      hls = new Hls({
        liveDurationInfinity: true,
        liveSyncDurationCount: 1,
        liveMaxLatencyDurationCount: 4,
        backBufferLength: 8,
        maxBufferLength: 8,
        nudgeMaxRetry: 10,
        fragLoadingMaxRetry: 8,
      });
      hls.loadSource(src);
      hls.attachMedia(v);
      hls.on(Hls.Events.MANIFEST_PARSED, () => v.play().catch(() => {}));
      hls.on(Hls.Events.ERROR, (_e, data) => {
        if (!data || !data.fatal) return;          // non-fatal: hls.js self-heals
        // A fatal error on a live feed is almost always an evicted segment or a
        // media glitch — try to recover in place before rebuilding the session.
        if (data.type === Hls.ErrorTypes.NETWORK_ERROR) {
          setCameraStatus("Reconnecting…");
          try { hls.startLoad(); } catch { restartStream(cam); }
        } else if (data.type === Hls.ErrorTypes.MEDIA_ERROR) {
          setCameraStatus("Reconnecting…");
          try { hls.recoverMediaError(); } catch { restartStream(cam); }
        } else {
          restartStream(cam);
        }
      });
      startWatchdog(cam);
    } else {
      v.src = src; v.play().catch(() => {});        // last-ditch
      startWatchdog(cam);
    }
  }).catch(() => setCameraStatus("Couldn't load the video player."));
}

// A live feed should never sit still. If the picture stops advancing for a few
// seconds (a silent hls.js stall, or the go2rtc session dropping us behind its
// window) rebuild the stream rather than leave a frozen frame that reads as a
// static thumbnail. Bounded so a genuinely dead feed surfaces an error.
function startWatchdog(cam) {
  stopWatchdog();
  const v = els.cameraVideo;
  camLastTime = -1;
  camStalls = 0;
  // camRetries is owned by openCamera (reset on a fresh open) so it survives the
  // rebuilds a stalled feed triggers — startStream/startWatchdog run per rebuild.
  camWatchdog = setInterval(() => {
    if (els.cameraModal.hidden) { stopWatchdog(); return; }
    if (document.hidden) return;                    // backgrounded tab isn't a stall
    if (v.paused || v.readyState < 2) return;       // still starting up
    if (v.currentTime > camLastTime + 0.05) {       // progress → healthy
      camLastTime = v.currentTime;
      camStalls = 0;
      camRetries = 0;
      if (els.cameraStatus.textContent === "Reconnecting…") setCameraStatus("");
      return;
    }
    if (++camStalls < 3) return;                    // tolerate a brief hiccup (~3s)
    camStalls = 0;
    if (++camRetries > 4) {                          // repeated rebuilds → give up
      stopWatchdog();
      setCameraStatus("Couldn't keep the live feed going.");
      return;
    }
    setCameraStatus("Reconnecting…");
    restartStream(cam);
  }, 1000);
}

function stopWatchdog() {
  if (camWatchdog) { clearInterval(camWatchdog); camWatchdog = null; }
}

// Rebuild the live session from scratch. camRetries is intentionally left alone
// (owned by openCamera) so the give-up bound counts across rebuilds.
function restartStream(cam) {
  startStream(cam);                                 // tears down + reattaches hls
}

function setCameraStatus(text) {
  els.cameraStatus.textContent = text || "";
  els.cameraStatus.hidden = !text;
}

function teardownHls() {
  if (hls) { try { hls.destroy(); } catch {} hls = null; }
}

function closeCamera() {
  stopWatchdog();
  teardownHls();
  const v = els.cameraVideo;
  try { v.pause(); v.removeAttribute("src"); v.removeAttribute("poster"); v.load(); } catch {}
  els.cameraModal.hidden = true;
  setCameraStatus("");
}

// ── Manual PDF viewer modal (#63) ───────────────────────────────────────────
// Opens the original PDF (served inline same-origin from /api/manuals/{id}/file)
// in a centered modal, jumping to a page via the #page=N fragment the browser's
// built-in PDF viewer understands. Reachable from a reply's citation chips and the
// Settings library, so a manual Nova just quoted is one click from a close look.
function openManual(id, page, title) {
  if (!id) { showToast("That manual is unavailable", "error"); return; }
  closeOverlays();
  if (!els.cameraModal.hidden) closeCamera();
  const base = `/api/manuals/${encodeURIComponent(id)}/file`;
  const src = Number.isFinite(page) ? `${base}#page=${page}` : base;
  els.pdfTitle.textContent = title || "Manual";
  els.pdfOpen.href = src;
  els.pdfFrame.src = src;
  els.pdfModal.hidden = false;
}

function closeManual() {
  els.pdfModal.hidden = true;
  els.pdfFrame.removeAttribute("src");   // stop rendering / free the PDF
}

// Generic SSE reader shared by the chat loop and Claude escalation.
async function streamSSE(url, body, onDelta, onTool, onMeta) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok || !res.body) throw new Error("stream " + res.status);
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const events = buf.split("\n\n");
    buf = events.pop();
    for (const evt of events) {
      const line = evt.split("\n").find((l) => l.startsWith("data:"));
      if (!line) continue;
      const payload = JSON.parse(line.slice(5).trim());
      if (payload.error) throw new Error(payload.error);
      if (payload.tool) onTool?.(payload);
      if (payload.sources) onMeta?.(payload);
      if (payload.conversation) onMeta?.(payload);
      if (payload.escalated) onMeta?.(payload);
      if (payload.model) onMeta?.(payload);
      if (payload.delta) onDelta(payload.delta);
      if (payload.done) return;
    }
  }
}

// Icon + settled label for a tool call — the single source of truth for chips
// and notifications. `tool` is the bare name (backend strips the namespace).
function toolInfo({ tool, args = {} }) {
  const ent = prettyEntity(args.entity_id);
  switch (tool) {
    case "call_service":  return { icon: "⚙️", text: `${(args.service || "").replace(/_/g, " ")} ${ent}`.trim() || "controlled a device" };
    case "find_entities": return { icon: "🔎", text: `searched for “${args.query || ent}”` };
    case "get_state":     return { icon: "📈", text: `read ${ent || "a device"}` };
    case "start_timer":   return { icon: "⏲️", text: "set a timer" };
    case "set_reminder":  return { icon: "⏰", text: args.alarm ? "set an alarm" : "set a reminder" };
    case "list_timers":   return { icon: "📋", text: "checked timers" };
    case "cancel_timer":  return { icon: "🗑️", text: `cancelled ${args.label || "a timer"}`.trim() };
    case "dismiss_alarm": return { icon: "🔕", text: "dismissed the alarm" };
    case "calculate":     return { icon: "🧮", text: "calculated" };
    case "date_diff":     return { icon: "📆", text: "did date math" };
    case "add_to_list":   return { icon: "🛒", text: `added ${args.item || "an item"} to ${args.name || "list"}`.trim() };
    case "read_list":     return { icon: "🛒", text: `read ${args.name || "a list"}` };
    case "create_list":   return { icon: "🛒", text: `created ${args.name || "a list"}` };
    case "check_item":    return { icon: "✅", text: `checked off ${args.item || "an item"}`.trim() };
    case "remove_item":   return { icon: "➖", text: `removed ${args.item || "an item"}`.trim() };
    case "clear_list":    return { icon: "🧹", text: `cleared ${args.name || "a list"}` };
    case "list_tasks":    return { icon: "🗒️", text: "checked tasks" };
    case "capture":       return { icon: "📝", text: args.section === "tasks" ? "added a task" : "made a note" };
    case "search_notes":  return { icon: "🔎", text: "searched notes" };
    case "read_note":     return { icon: "📖", text: "read a note" };
    case "create_note":   return { icon: "📝", text: "created a note" };
    case "append_note":   return { icon: "📝", text: "updated a note" };
    case "create_project":return { icon: "📁", text: `created ${args.name || "a project"}` };
    case "add_project_doc":return { icon: "📄", text: "added a project doc" };
    case "list_projects": return { icon: "📁", text: "listed projects" };
    case "remember":      return { icon: "🧠", text: "saved a memory" };
    case "forget":        return { icon: "🧠", text: "forgot a memory" };
    case "web_search":    return { icon: "🌐", text: "searched the web" };
    case "search_manuals":return { icon: "📘", text: "checked the manuals" };
    default:              return { icon: "⚙️", text: tool };
  }
}

function prettyEntity(id) { return id ? id.split(".").pop().replace(/_/g, " ") : ""; }

function toolChip(payload) {
  const { icon, text } = toolInfo(payload);
  const chip = document.createElement("span");
  chip.className = "tool-chip";
  chip.textContent = `${icon} ${text}`;
  const args = payload.args && Object.keys(payload.args).length ? JSON.stringify(payload.args) : "";
  chip.title = `${payload.tool}${args ? "(" + args + ")" : ""}`;
  return chip;
}

// Get-or-create the chips/badge row under an assistant bubble (above its
// timestamp), so the tool chips and the model badge share one line (#66).
function ensureChips(bubble) {
  if (!bubble.chipsEl) {
    const wrap = document.createElement("div");
    wrap.className = "tool-chips";
    bubble.rowEl.insertBefore(wrap, bubble.metaEl);
    bubble.chipsEl = wrap;
  }
  return bubble.chipsEl;
}

function addToolChip(bubble, payload) {
  ensureChips(bubble).appendChild(toolChip(payload));
}

// Citation chips under a reply: one per cited manual/page, each opening the stored
// PDF at that page in the viewer modal (#63). Given fresh each turn, so replace any
// prior row on this bubble rather than appending.
function renderSources(bubble, sources) {
  if (!Array.isArray(sources) || !sources.length) return;
  if (bubble.sourcesEl) bubble.sourcesEl.remove();
  const row = document.createElement("div");
  row.className = "reply-sources";
  for (const s of sources) {
    const chip = document.createElement("button");
    chip.type = "button"; chip.className = "source-chip";
    const page = Number.isFinite(s.page) ? ` · p.${s.page}` : "";
    chip.textContent = `📘 ${s.title || "Manual"}${page}`;
    chip.title = `Open “${s.title || "manual"}”${page ? " at page " + s.page : ""}`;
    chip.addEventListener("click", () => openManual(s.id, s.page, s.title));
    row.appendChild(chip);
  }
  bubble.rowEl.insertBefore(row, bubble.metaEl);
  bubble.sourcesEl = row;
}

// Tag an assistant reply with the model that produced it — the local model, or a
// Claude tier (✦) on an auto-escalated turn (#65): violet for the deep/Opus brain,
// pink for the fast/Haiku brain, matching their constellation nodes. End of the row.
function setModelBadge(bubble, model) {
  if (!model) return;
  if (bubble.modelBadge) bubble.modelBadge.remove();
  const cloud = /claude/i.test(model);
  const fast = cloud && /haiku/i.test(model);   // the fast/cheap reactive tier
  const badge = document.createElement("span");
  badge.className = "model-badge" + (cloud ? " cloud" : "") + (fast ? " fast" : "");
  badge.textContent = (cloud ? "✦ " : "") + model;
  badge.title = "Generated by " + model;
  ensureChips(bubble).appendChild(badge);
  bubble.modelBadge = badge;
}

// ── Markdown — assistant replies stream as plain text, then format on completion
// (lists / headings / bold / code / links) into HTML (#64). ──────────────────
function renderMarkdown(bubble, text) {
  bubble.innerHTML = mdToHtml(text);
  bubble.classList.add("markdown");
}
function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function renderInline(text) {
  return text.split(/(`[^`]+`)/g).map((part) => {
    if (part.length >= 2 && part.startsWith("`") && part.endsWith("`")) {
      return "<code>" + part.slice(1, -1) + "</code>";
    }
    let s = part.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (m, txt, url) =>
      /^(https?:|mailto:)/i.test(url)
        ? `<a href="${url}" target="_blank" rel="noopener noreferrer">${txt}</a>` : m);
    s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
         .replace(/__([^_]+)__/g, "<strong>$1</strong>")
         .replace(/(^|[^*])\*(?!\s)([^*]+?)\*/g, "$1<em>$2</em>")
         .replace(/(^|[^\w])_(?!\s)([^_]+?)_/g, "$1<em>$2</em>");
    return s;
  }).join("");
}
function mdToHtml(src) {
  const lines = escapeHtml((src || "").replace(/\r\n/g, "\n")).split("\n");
  const isUl = (l) => /^\s*[-*+]\s+/.test(l);
  const isOl = (l) => /^\s*\d+\.\s+/.test(l);
  const isSpecial = (l) => /^```/.test(l) || /^#{1,6}\s/.test(l) || isUl(l) || isOl(l) || /^>\s?/.test(l);
  const out = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (/^```/.test(line)) {                              // fenced code block
      const buf = []; i++;
      while (i < lines.length && !/^```\s*$/.test(lines[i])) buf.push(lines[i++]);
      i++;                                                // skip closing fence
      out.push("<pre><code>" + buf.join("\n") + "</code></pre>");
    } else if (/^(#{1,6})\s+/.test(line)) {               // heading
      const m = line.match(/^(#{1,6})\s+(.*)$/);
      const lvl = Math.min(m[1].length, 4);
      out.push(`<h${lvl}>${renderInline(m[2])}</h${lvl}>`); i++;
    } else if (/^>\s?/.test(line)) {                      // blockquote
      const buf = [];
      while (i < lines.length && /^>\s?/.test(lines[i])) buf.push(lines[i++].replace(/^>\s?/, ""));
      out.push("<blockquote>" + renderInline(buf.join(" ")) + "</blockquote>");
    } else if (isUl(line) || isOl(line)) {                // list
      const ordered = isOl(line);
      const match = ordered ? isOl : isUl;
      const items = [];
      while (i < lines.length && match(lines[i])) {
        items.push("<li>" + renderInline(lines[i].replace(/^\s*(?:[-*+]|\d+\.)\s+/, "")) + "</li>"); i++;
      }
      out.push((ordered ? "<ol>" : "<ul>") + items.join("") + (ordered ? "</ol>" : "</ul>"));
    } else if (/^\s*$/.test(line)) {                      // blank
      i++;
    } else {                                              // paragraph
      const buf = [];
      while (i < lines.length && !/^\s*$/.test(lines[i]) && !isSpecial(lines[i])) buf.push(lines[i++]);
      out.push("<p>" + renderInline(buf.join(" ")) + "</p>");
    }
  }
  return out.join("");
}

// ── Settings ───────────────────────────────────────────────────────────────
let settingsLoaded = false;

async function openSettings() {
  closeOverlays();
  els.settings.hidden = false;
  switchSettingsTab("general");
  if (!settingsLoaded) await loadSettings();
}
function closeSettings() { els.settings.hidden = true; stopLogPolling(); }

function switchSettingsTab(name) {
  document.querySelectorAll("#settings [role=tabpanel]").forEach((panel) => {
    panel.hidden = panel.id !== `tab-${name}`;
  });
  document.querySelectorAll(".settings-tab").forEach((btn) =>
    btn.classList.toggle("is-active", btn.dataset.tab === name));
  if (name === "logs") startLogPolling(); else stopLogPolling();
  if (name === "memory") loadMemories();
  if (name === "home") loadHaDevices(true);
  if (name === "manuals") loadManuals();
}

async function loadSettings() {
  setSettingsStatus("");
  try {
    const res = await fetch("/api/settings");
    if (!res.ok) throw new Error("settings " + res.status);
    const { settings, options } = await res.json();
    fillSelect(els.voiceSetting, options.voice || [], settings.voice);
    fillSelect(els.modelSetting, options.model || [], settings.model);
    els.voicePromptSetting.value = settings.voice_prompt || "";
    els.textPromptSetting.value = settings.text_prompt || "";
    els.notifyEnabled.checked = settings.notifications_enabled === "1";
    const cats = new Set((settings.notification_categories || "").split(",").map((c) => c.trim()));
    document.querySelectorAll(".notifyCat").forEach((cb) => { cb.checked = cats.has(cb.value); });
    els.notifyAddress.value = settings.mqtt_address || "";
    els.notifyTopic.value = settings.mqtt_topic || "";
    els.autoEscalate.checked = settings.auto_escalate === "1";
    settingsLoaded = true;
  } catch (err) { setSettingsStatus("Couldn't load settings: " + err.message, true); }
}

function fillSelect(select, values, current) {
  select.innerHTML = "";
  for (const v of values) {
    const opt = document.createElement("option");
    opt.value = v; opt.textContent = v;
    if (v === current) opt.selected = true;
    select.appendChild(opt);
  }
}

async function putSettings(payload) {
  setSettingsStatus("Saving…");
  const res = await fetch("/api/settings", {
    method: "PUT", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || res.status);
  }
  setSettingsStatus("Saved — applies to this app and the satellite.");
  return (await res.json()).settings;
}

async function saveVoice() {
  try { await putSettings({ voice: els.voiceSetting.value }); }
  catch (err) { setSettingsStatus("Couldn't save: " + err.message, true); }
}
async function saveModel() {
  try {
    const s = await putSettings({ model: els.modelSetting.value });
    if (s && s.model) { MODEL = s.model; setModelNode(MODEL, MODEL_COLOR); }
  } catch (err) { setSettingsStatus("Couldn't save: " + err.message, true); }
}
async function saveAutoEscalate() {
  try { await putSettings({ auto_escalate: els.autoEscalate.checked ? "1" : "0" }); }
  catch (err) { setSettingsStatus("Couldn't save: " + err.message, true); }
}

async function saveNotifications() {
  const categories = [...document.querySelectorAll(".notifyCat")]
    .filter((cb) => cb.checked).map((cb) => cb.value).join(",");
  try {
    await putSettings({
      notifications_enabled: els.notifyEnabled.checked ? "1" : "0",
      notification_categories: categories,
      mqtt_address: els.notifyAddress.value.trim(),
      mqtt_topic: els.notifyTopic.value.trim(),
    });
  } catch (err) { setSettingsStatus("Couldn't save: " + err.message, true); }
}

async function testNotification() {
  setSettingsStatus("Saving, then sending a test…");
  try {
    await saveNotifications();
    const res = await fetch("/api/notifications/test", { method: "POST" });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || res.status);
    }
    setSettingsStatus("Test sent — check your phone. If nothing arrives, see the activity log.");
  } catch (err) { setSettingsStatus("Test failed: " + err.message, true); }
}

async function savePrompt(key, value) {
  try { await putSettings({ [key]: value }); }
  catch (err) { setSettingsStatus("Couldn't save: " + err.message, true); }
}
async function resetPrompt(key) {
  try {
    const settings = await putSettings({ reset: [key] });
    const el = key === "text_prompt" ? els.textPromptSetting : els.voicePromptSetting;
    el.value = settings[key] || "";
    setSettingsStatus("Reset to the built-in default.");
  } catch (err) { setSettingsStatus("Couldn't reset: " + err.message, true); }
}

function setSettingsStatus(text, isError = false) {
  els.settingsStatus.textContent = text;
  els.settingsStatus.classList.toggle("error", isError);
}

async function loadHaDevices(refresh = false) {
  els.haRefresh.disabled = true;
  els.haRefresh.textContent = refresh ? "Refreshing…" : "Refresh";
  try {
    const res = await fetch("/api/ha/devices" + (refresh ? "?refresh=1" : ""));
    if (!res.ok) throw new Error("ha " + res.status);
    const { enabled, label, devices } = await res.json();
    els.haDeviceList.innerHTML = "";
    if (!enabled) showHaEmpty("Home Assistant isn't configured on the server.");
    else if (!devices.length) showHaEmpty(`No devices are labeled “${label}” in Home Assistant yet, so Nova can't control anything.`);
    else { els.haDeviceEmpty.hidden = true; for (const d of devices) els.haDeviceList.appendChild(renderHaDevice(d)); }
  } catch (err) { showHaEmpty("Couldn't load devices: " + err.message); }
  finally { els.haRefresh.disabled = false; els.haRefresh.textContent = "Refresh"; }
}
function showHaEmpty(text) { els.haDeviceList.innerHTML = ""; els.haDeviceEmpty.textContent = text; els.haDeviceEmpty.hidden = false; }
function renderHaDevice(d) {
  const li = document.createElement("li"); li.className = "ha-item";
  const row = document.createElement("div"); row.className = "ha-row";
  const name = document.createElement("span"); name.className = "ha-name"; name.textContent = d.name;
  const state = document.createElement("span");
  const on = /^(on|open|home|heat|cool|playing|locked)$/i.test(d.state || "");
  state.className = "ha-state" + (on ? " on" : ""); state.textContent = d.state ?? "";
  row.append(name, state);
  const meta = document.createElement("div"); meta.className = "ha-meta";
  meta.textContent = [d.domain, d.area, d.entity_id].filter(Boolean).join(" · ");
  li.append(row, meta);
  return li;
}

// ── Manuals tab (#63) ─────────────────────────────────────────────────────────
// Upload PDF manuals; the server extracts, embeds, and indexes them so Nova can
// answer from them (search_manuals). This tab manages the library.
let manualsCache = [];            // full library from the last fetch
let manualAttentionOnly = false;  // "N need attention" filter toggle
let manualPollTimer = null;
async function loadManuals() {
  try {
    const res = await fetch("/api/manuals");
    if (!res.ok) throw new Error("manuals " + res.status);
    const { manuals, enabled } = await res.json();
    els.manualDisabled.hidden = enabled;
    els.manualAddForm.hidden = !enabled;
    manualsCache = manuals || [];
    applyManualView();
  } catch (err) { setManualStatus("Couldn't load manuals: " + err.message, true); }
}

const MANUAL_ATTENTION = (m) => m.status === "empty" || m.status === "error";
const MANUAL_BUSY = (m) => m.status === "pending" || m.status === "processing";

// Render the library through the current search box, sort, and attention filter.
// Runs on every keystroke/sort change and on each poll — all from manualsCache, so
// it never refetches. A flat 45-item list becomes findable: filter down, order by
// name/recency/status, and jump straight to the ones that failed to index.
function applyManualView() {
  const all = manualsCache;
  const attention = all.filter(MANUAL_ATTENTION).length;
  if (!attention) manualAttentionOnly = false;   // nothing to filter to

  const q = els.manualSearch.value.trim().toLowerCase();
  let view = all.slice();
  if (manualAttentionOnly) view = view.filter(MANUAL_ATTENTION);
  if (q) view = view.filter((m) =>
    (m.title || "").toLowerCase().includes(q) || (m.filename || "").toLowerCase().includes(q));
  sortManuals(view, els.manualSort.value);

  els.manualList.innerHTML = "";
  for (const m of view) els.manualList.appendChild(manualItem(m));

  els.manualTools.hidden = all.length === 0;
  renderManualCount(all, attention);
  if (all.length === 0) {
    els.manualEmpty.textContent = "No manuals uploaded yet.";
    els.manualEmpty.hidden = false;
  } else if (view.length === 0) {
    els.manualEmpty.textContent = manualAttentionOnly ? "Nothing needs attention." : "No manuals match your search.";
    els.manualEmpty.hidden = false;
  } else {
    els.manualEmpty.hidden = true;
  }

  // Keep polling while anything is still indexing (based on the full library, not
  // the filtered view), until it settles or the tab leaves the screen.
  clearTimeout(manualPollTimer);
  if (all.some(MANUAL_BUSY) && els.manualList.offsetParent !== null) {
    manualPollTimer = setTimeout(loadManuals, 2500);
  }
}

function sortManuals(list, mode) {
  const byTitle = (a, b) => (a.title || "").localeCompare(b.title || "", undefined, { sensitivity: "base" });
  if (mode === "new") return list.sort((a, b) => (b.created_at || 0) - (a.created_at || 0) || byTitle(a, b));
  if (mode === "status") {
    // Problems first, so failed/empty imports surface at the top; then A–Z.
    const rank = { error: 0, empty: 1, processing: 2, pending: 3, ready: 4 };
    return list.sort((a, b) => (rank[a.status] ?? 9) - (rank[b.status] ?? 9) || byTitle(a, b));
  }
  return list.sort(byTitle);  // A–Z (default management view)
}

function renderManualCount(all, attention) {
  els.manualCount.innerHTML = "";
  els.manualCount.hidden = all.length === 0;
  if (!all.length) return;
  const ready = all.filter((m) => m.status === "ready").length;
  const busy = all.filter(MANUAL_BUSY).length;
  const summary = document.createElement("span");
  summary.textContent = `${all.length} manual${all.length === 1 ? "" : "s"} · ${ready} ready`
    + (busy ? ` · ${busy} indexing` : "");
  els.manualCount.appendChild(summary);
  if (attention) {
    els.manualCount.append(" · ");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "manual-attention" + (manualAttentionOnly ? " active" : "");
    btn.textContent = `${attention} need attention`;
    btn.title = manualAttentionOnly ? "Show all manuals" : "Show only manuals that need attention";
    btn.addEventListener("click", () => { manualAttentionOnly = !manualAttentionOnly; applyManualView(); });
    els.manualCount.appendChild(btn);
  }
}
function manualItem(m) {
  const li = document.createElement("li"); li.className = "ha-item";
  const row = document.createElement("div"); row.className = "ha-row";
  const name = document.createElement("span"); name.className = "ha-name"; name.textContent = m.title;
  // A 'ready' manual links to its stored PDF — a plain click opens the in-app
  // viewer; a modifier/middle click follows the href to a new tab as usual.
  if (m.status === "ready") {
    const view = document.createElement("a");
    view.className = "ha-view"; view.href = `/api/manuals/${m.id}/file`;
    view.rel = "noopener"; view.textContent = "View PDF";
    view.addEventListener("click", (e) => {
      if (e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return;
      e.preventDefault();
      openManual(m.id, null, m.title);
    });
    row.append(name, view);
  } else {
    row.append(name);
  }
  const del = document.createElement("button"); del.className = "mem-del"; del.type = "button"; del.textContent = "✕"; del.title = "Delete this manual";
  del.addEventListener("click", () => deleteManual(m.id));
  row.append(del);
  const meta = document.createElement("div"); meta.className = "ha-meta";
  if (m.status === "ready") meta.textContent = `${m.pages} page${m.pages === 1 ? "" : "s"} · ${m.chunks} chunks indexed`;
  else if (m.status === "pending") meta.textContent = "⏳ Queued for indexing…";
  else if (m.status === "processing") meta.textContent = "⏳ Indexing…";
  else if (m.status === "empty") meta.textContent = "⚠ No readable text — a scanned/image-only PDF can't be indexed.";
  else meta.textContent = "⚠ " + (m.error || "Couldn't be indexed.");
  li.append(row, meta);
  return li;
}
async function uploadManual(e) {
  e.preventDefault();
  const files = [...els.manualFile.files];
  if (!files.length) { setManualStatus("Choose one or more PDFs first.", true); return; }
  const body = new FormData();
  for (const file of files) body.append("files", file);
  els.manualUpload.disabled = true;
  const label = files.length === 1 ? `“${files[0].name}”` : `${files.length} manuals`;
  setManualStatus(`Uploading ${label}…`);
  try {
    const res = await fetch("/api/manuals/batch", { method: "POST", body });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || res.status);
    const staged = data.staged || [], errors = data.errors || [];
    let msg = staged.length
      ? `Queued ${staged.length} manual${staged.length === 1 ? "" : "s"} for indexing…`
      : "Nothing was queued.";
    if (errors.length) msg += ` Skipped ${errors.length}: ${errors.map((x) => `${x.filename} (${x.error})`).join(", ")}.`;
    setManualStatus(msg, staged.length === 0);
    els.manualFile.value = "";
    loadManuals();
  } catch (err) {
    setManualStatus("Upload failed: " + err.message, true);
  } finally {
    els.manualUpload.disabled = false;
  }
}
async function deleteManual(id) {
  try {
    await fetch(`/api/manuals/${id}`, { method: "DELETE" });
    setManualStatus("Deleted.");
    loadManuals();
  } catch (err) { setManualStatus("Couldn't delete: " + err.message, true); }
}
function setManualStatus(text, isError = false) {
  els.manualStatus.textContent = text;
  els.manualStatus.classList.toggle("error", isError);
}

// ── Activity log viewer ────────────────────────────────────────────────────
const LOG_PAGE = 500;
let logTimer = null;
let logOldestId = null;
let logNewestId = 0;
let logAtStart = false;

function startLogPolling() { loadLog(); if (logTimer) clearInterval(logTimer); logTimer = setInterval(appendNewLog, 3000); }
function stopLogPolling() { if (logTimer) clearInterval(logTimer); logTimer = null; }
function logQuery(params) {
  const level = els.logLevel.value;
  const q = new URLSearchParams({ limit: String(LOG_PAGE), ...params });
  if (level) q.set("level", level);
  if (els.logHideHealth.checked) q.set("hide_health", "1");
  return "/api/log?" + q.toString();
}
async function loadLog() {
  try {
    const res = await fetch(logQuery({}));
    if (!res.ok) throw new Error("log " + res.status);
    const { log, retention_hours } = await res.json();
    if (retention_hours) els.logRetention.textContent = retention_hours;
    els.logView.innerHTML = "";
    logOldestId = null; logNewestId = 0; logAtStart = log.length < LOG_PAGE;
    els.logEmpty.hidden = log.length > 0;
    appendLines(log.slice().reverse(), "bottom");
    els.logView.scrollTop = els.logView.scrollHeight;
    updateOlderButton();
  } catch (err) { els.logView.textContent = "Couldn't load the log: " + err.message; }
}
async function loadOlderLog() {
  if (logOldestId == null || logAtStart) return;
  els.logOlder.disabled = true;
  const v = els.logView, beforeHeight = v.scrollHeight, beforeTop = v.scrollTop;
  try {
    const res = await fetch(logQuery({ before: String(logOldestId) }));
    if (!res.ok) throw new Error("log " + res.status);
    const { log } = await res.json();
    logAtStart = log.length < LOG_PAGE;
    appendLines(log.slice().reverse(), "top");
    v.scrollTop = beforeTop + (v.scrollHeight - beforeHeight);
    updateOlderButton();
  } catch (err) { setSettingsStatus("Couldn't load older log: " + err.message, true); }
  finally { els.logOlder.disabled = false; }
}
async function appendNewLog() {
  if (!logNewestId) return;
  try {
    const res = await fetch(logQuery({ since: String(logNewestId) }));
    if (!res.ok) return;
    const { log } = await res.json();
    if (!log.length) return;
    const v = els.logView, atBottom = v.scrollHeight - v.scrollTop - v.clientHeight < 24;
    els.logEmpty.hidden = true;
    appendLines(log.slice().reverse(), "bottom");
    if (atBottom) v.scrollTop = v.scrollHeight;
  } catch { /* transient */ }
}
function appendLines(entries, where) {
  const frag = document.createDocumentFragment();
  for (const e of entries) {
    frag.appendChild(logLine(e));
    if (logOldestId == null || e.id < logOldestId) logOldestId = e.id;
    if (e.id > logNewestId) logNewestId = e.id;
  }
  if (where === "top") els.logView.insertBefore(frag, els.logView.firstChild);
  else els.logView.appendChild(frag);
}
function logLine(e) {
  const line = document.createElement("span");
  line.className = "log-line " + (e.level || "").toLowerCase();
  const t = document.createElement("span"); t.className = "log-time";
  t.textContent = new Date(e.ts * 1000).toLocaleString([], {
    month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false,
  }) + " ";
  const lvl = document.createElement("span"); lvl.textContent = (e.level || "").padEnd(5) + " ";
  const name = document.createElement("span"); name.className = "log-logger"; name.textContent = e.logger + " ";
  line.append(t, lvl, name, document.createTextNode(e.message));
  return line;
}
function updateOlderButton() { els.logOlder.hidden = logAtStart || logOldestId == null; }
async function clearLog() {
  els.logClear.disabled = true;
  try { await fetch("/api/log", { method: "DELETE" }); await loadLog(); }
  catch (err) { els.logView.textContent = "Couldn't clear the log: " + err.message; }
  finally { els.logClear.disabled = false; }
}

// ── Memory tab ───────────────────────────────────────────────────────────────
const MEM_TIERS = [
  { key: "long", label: "Long-term" },
  { key: "mid", label: "This month" },
  { key: "short", label: "Today" },
];
async function loadMemories() {
  try {
    const res = await fetch("/api/memories");
    if (!res.ok) throw new Error("memories " + res.status);
    renderMemories((await res.json()).memories);
  } catch (err) { setMemStatus("Couldn't load memory: " + err.message, true); }
}
function renderMemories(memories) {
  els.memList.innerHTML = "";
  els.memEmpty.hidden = memories.length > 0;
  for (const { key, label } of MEM_TIERS) {
    const items = memories.filter((m) => m.tier === key);
    if (!items.length) continue;
    const group = document.createElement("div"); group.className = "mem-group";
    const title = document.createElement("p"); title.className = "mem-group-title"; title.textContent = label;
    const list = document.createElement("div"); list.className = "mem-items";
    for (const m of items) list.appendChild(memItem(m));
    group.append(title, list);
    els.memList.appendChild(group);
  }
}
function memItem(m) {
  const row = document.createElement("div"); row.className = "mem-item";
  const text = document.createElement("input"); text.className = "mem-text"; text.value = m.text;
  text.setAttribute("aria-label", "Memory text");
  const commit = () => { const v = text.value.trim(); if (v && v !== m.text) updateMemory(m.id, { text: v }); else text.value = m.text; };
  text.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); text.blur(); } });
  text.addEventListener("blur", commit);
  const tier = document.createElement("select"); tier.className = "mem-tier"; tier.setAttribute("aria-label", "How long to keep it");
  for (const { key, label } of MEM_TIERS) {
    const opt = document.createElement("option"); opt.value = key; opt.textContent = label;
    if (key === m.tier) opt.selected = true; tier.appendChild(opt);
  }
  tier.addEventListener("change", () => updateMemory(m.id, { tier: tier.value }));
  const del = document.createElement("button"); del.className = "mem-del"; del.type = "button"; del.textContent = "✕"; del.title = "Forget this";
  del.addEventListener("click", () => deleteMemory(m.id));
  row.append(text, tier, del);
  if (m.source === "assistant") {
    const auto = document.createElement("span"); auto.className = "mem-auto"; auto.textContent = "•"; auto.title = "Remembered by Nova";
    row.insertBefore(auto, del);
  }
  return row;
}
async function addMemory(e) {
  e.preventDefault();
  const text = els.memText.value.trim();
  if (!text) return;
  try {
    const res = await fetch("/api/memories", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, tier: els.memTier.value }),
    });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.status);
    els.memText.value = ""; setMemStatus("Saved."); loadMemories();
  } catch (err) { setMemStatus("Couldn't add: " + err.message, true); }
}
async function updateMemory(id, patch) {
  try {
    const res = await fetch(`/api/memories/${id}`, {
      method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(patch),
    });
    if (!res.ok) throw new Error(res.status);
    setMemStatus("Updated."); loadMemories();
  } catch (err) { setMemStatus("Couldn't update: " + err.message, true); }
}
async function deleteMemory(id) {
  try { await fetch(`/api/memories/${id}`, { method: "DELETE" }); loadMemories(); }
  catch (err) { setMemStatus("Couldn't delete: " + err.message, true); }
}
function setMemStatus(text, isError = false) { els.memStatus.textContent = text; els.memStatus.classList.toggle("error", isError); }

// ── Conversations drawer ────────────────────────────────────────────────────
async function loadConvoList(silent = false) {
  // Only refresh when the drawer is open (or explicitly requested after a turn).
  if (!els.convos.classList.contains("open") && !silent) return;
  try {
    const res = await fetch("/api/conversations");
    if (!res.ok) throw new Error("conversations " + res.status);
    const { conversations } = await res.json();
    els.convoList.innerHTML = "";
    els.convoEmpty.hidden = conversations.length > 0;
    for (const c of conversations) els.convoList.appendChild(renderConvo(c));
    markActive(conversationId);
  } catch (err) { if (!silent) showToast("Couldn't load conversations: " + err.message, "error"); }
}

const SOURCE_ICON = {
  web: '<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="6.3" fill="none" stroke="currentColor" stroke-width="1.2"/><path d="M1.7 8h12.6M8 1.7c1.7 1.8 1.7 10.8 0 12.6M8 1.7c-1.7 1.8-1.7 10.8 0 12.6" fill="none" stroke="currentColor" stroke-width="1"/></svg>',
  satellite: '<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="4" cy="12" r="2" fill="currentColor"/><path d="M7 12A7 7 0 0 0 4 6.2M10.5 12A10.5 10.5 0 0 0 4 2.5" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>',
};
function sourceChip(source) {
  const chip = document.createElement("span");
  chip.className = "src-chip src-" + source;
  if (SOURCE_ICON[source]) chip.innerHTML = SOURCE_ICON[source];
  chip.append(document.createTextNode(source));
  return chip;
}
function renderConvo(c) {
  const li = document.createElement("li");
  li.className = "convo-item"; li.dataset.id = c.id;
  li.addEventListener("click", () => selectConversation(c.id));
  const row = document.createElement("div"); row.className = "c-row";
  const title = document.createElement("span"); title.className = "c-title"; title.textContent = c.title;
  const del = document.createElement("button"); del.className = "c-del"; del.type = "button"; del.textContent = "✕"; del.title = "Delete";
  del.addEventListener("click", (e) => { e.stopPropagation(); deleteConversation(c.id); });
  row.append(title, del);
  const meta = document.createElement("div"); meta.className = "c-meta";
  meta.append(sourceChip(c.source), document.createTextNode(` · ${c.message_count} msg · ${timeAgo(c.updated_at)}`));
  li.append(row, meta);
  return li;
}
async function selectConversation(id) {
  try {
    const res = await fetch(`/api/conversations/${id}`);
    if (!res.ok) throw new Error("load " + res.status);
    const conv = await res.json();
    history.length = 0;
    clearTranscript();
    let lastUserAt = null;
    for (const m of conv.messages) {
      const at = m.created_at ? new Date(m.created_at * 1000) : null;
      const b = addMessage(m.role, m.content, at ? fmtClock(at) : "");
      for (const tc of m.tool_calls || []) addToolChip(b, tc);
      if (m.role === "assistant") {
        if (m.content) renderMarkdown(b, m.content);   // re-format stored markdown
        if (m.model) setModelBadge(b, m.model);         // re-badge the model
      }
      if (m.role === "user") lastUserAt = m.created_at || null;
      else if (m.role === "assistant" && lastUserAt && m.created_at) {
        const secs = m.created_at - lastUserAt;
        if (secs >= 0) b.metaEl.textContent += ` · ${secs}s`;
        lastUserAt = null;
      }
      history.push({ role: m.role, content: m.content });
    }
    conversationId = id;
    markActive(id);
    clearRoute();   // switching context clears the last turn's graph highlight
    closeOverlays();
  } catch (err) { showToast("Couldn't open conversation: " + err.message, "error"); }
}
async function deleteConversation(id) {
  try {
    const res = await fetch(`/api/conversations/${id}`, { method: "DELETE" });
    if (!res.ok) throw new Error("delete " + res.status);
  } catch (err) { showToast("Couldn't delete: " + err.message, "error"); return; }
  if (id === conversationId) newChat();
  loadConvoList();
}
function newChat() {
  history.length = 0; conversationId = null;
  clearTranscript(); markActive(null); clearRoute(); closeOverlays();
}
function markActive(id) {
  for (const li of els.convoList.children) li.classList.toggle("active", li.dataset.id === id);
}
function clearTranscript() {
  els.transcript.innerHTML = "";
  const hint = document.createElement("div");
  hint.id = "hint";
  hint.className = "hint";
  hint.textContent = "Tap the mic or type a command to begin.";
  els.transcript.appendChild(hint);
  els.hint = hint;
  updateThreadFade();   // reset the top fade when the thread empties / switches
}
function timeAgo(epochSec) {
  const s = Math.floor(Date.now() / 1000) - epochSec;
  if (s < 60) return "just now";
  const m = Math.floor(s / 60); if (m < 60) return m + "m ago";
  const h = Math.floor(m / 60); if (h < 24) return h + "h ago";
  return Math.floor(h / 24) + "d ago";
}

// ── Timers & reminders (popover + top-bar pill) ────────────────────────────
async function loadTimers() {
  try {
    const res = await fetch("/api/timers");
    if (!res.ok) throw new Error("timers " + res.status);
    const data = await res.json();
    clockOffset = data.now - Math.floor(Date.now() / 1000);
    firedAlerted.clear();
    timers = data.timers;
    ringingAlarms = data.ringing || [];
    renderTimers();
    syncAlarmLoop();
  } catch { /* keep last-known; the diamond flags outages */ }
}
function renderTimers() {
  els.taskList.innerHTML = "";
  els.taskEmpty.hidden = timers.length > 0 || ringingAlarms.length > 0;
  for (const a of ringingAlarms) els.taskList.appendChild(renderRingingAlarm(a));
  for (const t of timers) els.taskList.appendChild(renderTask(t));
  tickTimers();
}
function renderRingingAlarm(a) {
  const li = document.createElement("li"); li.className = "task-item ringing-alarm";
  const row = document.createElement("div"); row.className = "t-row";
  const label = document.createElement("span"); label.className = "t-label"; label.textContent = `🔔 ${a.label}`;
  const dismiss = document.createElement("button"); dismiss.className = "t-dismiss"; dismiss.type = "button"; dismiss.textContent = "Dismiss";
  dismiss.addEventListener("click", () => dismissAlarm(a.id));
  row.append(label, dismiss);
  const meta = document.createElement("div"); meta.className = "t-meta";
  meta.textContent = `Alarm${a.cadence ? " · ↻ " + a.cadence : ""} · ${a.when}`;
  li.append(row, meta);
  return li;
}
function syncAlarmLoop() {
  if (ringingAlarms.length && !alarmTimer) { playAlarm(); alarmTimer = setInterval(playAlarm, 1800); }
  else if (!ringingAlarms.length && alarmTimer) { clearInterval(alarmTimer); alarmTimer = null; }
}
async function dismissAlarm(id) {
  ringingAlarms = ringingAlarms.filter((a) => a.id !== id);
  renderTimers(); syncAlarmLoop();
  try { await fetch(`/api/alarms/${id}`, { method: "DELETE" }); } catch { /* refresh reconciles */ }
  loadTimers();
}
function renderTask(t) {
  const li = document.createElement("li"); li.className = "task-item";
  li.dataset.id = t.id; li.dataset.fire = t.fire_at; li.dataset.alarm = t.is_alarm ? "1" : "0";
  const row = document.createElement("div"); row.className = "t-row";
  const label = document.createElement("span"); label.className = "t-label"; label.textContent = t.label;
  const cancel = document.createElement("button"); cancel.className = "t-cancel"; cancel.type = "button"; cancel.textContent = "✕"; cancel.title = "Cancel";
  cancel.addEventListener("click", () => cancelTimer(t.id));
  row.append(label, cancel);
  const count = document.createElement("div"); count.className = "t-count";
  const meta = document.createElement("div"); meta.className = "t-meta";
  meta.append(sourceChip(t.source), document.createTextNode(` · ${t.is_alarm ? "alarm" : t.kind}`));
  if (t.cadence) {
    const rep = document.createElement("span"); rep.className = "t-cadence"; rep.textContent = `↻ ${t.cadence}`;
    meta.append(document.createTextNode(" · "), rep);
  }
  li.append(row, count, meta);
  return li;
}
function tickTimers() {
  const now = Math.floor(Date.now() / 1000) + clockOffset;
  for (const li of els.taskList.children) {
    if (!li.dataset.fire) continue;   // ringing-alarm rows have no countdown
    const remaining = Number(li.dataset.fire) - now;
    const count = li.querySelector(".t-count");
    if (remaining <= 0) {
      if (count) count.textContent = "0:00";
      li.classList.add("firing");
      const id = li.dataset.id;
      if (!firedAlerted.has(id)) {
        firedAlerted.add(id);
        const label = li.querySelector(".t-label")?.textContent || "Timer";
        if (li.dataset.alarm === "1") { loadTimers(); }
        else { playAlert(); pushNotif(`⏲️ ${label} fired`, "#ffc857"); setTimeout(loadTimers, 1500); }
      }
    } else if (count) {
      count.textContent = formatCountdown(remaining);
    }
  }
  updateTimerPill(now);
}
function updateTimerPill(now) {
  const active = timers.filter((t) => Number(t.fire_at) - now > 0);
  const count = active.length + ringingAlarms.length;
  if (!count) { els.timerPill.hidden = true; return; }
  els.timerPill.hidden = false;
  if (active.length) {
    const nearest = active.reduce((a, b) => (Number(a.fire_at) < Number(b.fire_at) ? a : b));
    const remaining = Number(nearest.fire_at) - now;
    const prev = timerMaxRemaining[nearest.id] || 0;
    const total = timerMaxRemaining[nearest.id] = Math.max(prev, remaining);
    const pct = total > 0 ? Math.max(0, Math.min(100, (remaining / total) * 100)) : 0;
    els.timerRing.style.setProperty("--arc", pct.toFixed(0) + "%");
    els.timerPillTime.textContent = formatCountdown(remaining);
  } else {
    els.timerRing.style.setProperty("--arc", "100%");
    els.timerPillTime.textContent = "•";
  }
  els.timerPillCount.textContent = `${count} ACTIVE`;
}
async function cancelTimer(id) {
  try { await fetch(`/api/timers/${id}`, { method: "DELETE" }); } catch { /* refresh reconciles */ }
  loadTimers();
}
function formatCountdown(sec) {
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  const pad = (n) => String(n).padStart(2, "0");
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`;
}
function playAlert() {
  try {
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    if (audioCtx.state === "suspended") audioCtx.resume();
    const beep = (t, f) => {
      const o = audioCtx.createOscillator(), g = audioCtx.createGain();
      o.type = "sine"; o.frequency.value = f; o.connect(g); g.connect(audioCtx.destination);
      g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(0.3, t + 0.02);
      g.gain.exponentialRampToValueAtTime(0.0001, t + 0.35); o.start(t); o.stop(t + 0.36);
    };
    const t0 = audioCtx.currentTime; beep(t0, 880); beep(t0 + 0.45, 1175); beep(t0 + 0.9, 880);
  } catch { /* visual firing state still shows */ }
}
function playAlarm() {
  try {
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    if (audioCtx.state === "suspended") audioCtx.resume();
    const beep = (t, f) => {
      const o = audioCtx.createOscillator(), g = audioCtx.createGain();
      o.type = "square"; o.frequency.value = f; o.connect(g); g.connect(audioCtx.destination);
      g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(0.28, t + 0.02);
      g.gain.exponentialRampToValueAtTime(0.0001, t + 0.22); o.start(t); o.stop(t + 0.23);
    };
    const t0 = audioCtx.currentTime; for (let i = 0; i < 4; i++) beep(t0 + i * 0.25, 1046 + i * 180);
  } catch { /* ringing item still shows */ }
}

// ── Devices (satellites) ───────────────────────────────────────────────────
async function loadSatellites(force = false) {
  let sats;
  try {
    const res = await fetch("/api/satellites");
    if (!res.ok) throw new Error("satellites " + res.status);
    sats = (await res.json()).satellites;
  } catch { return; }

  // Notify on an offline→online transition (skip the very first load).
  const nowOnline = new Set(sats.filter((s) => s.online).map((s) => s.id));
  if (satInitialized) {
    for (const s of sats) if (s.online && !satOnline.has(s.id)) pushNotif(`${s.name} reconnected`, "#7de3a0");
  }
  satOnline = nowOnline;
  satInitialized = true;
  els.devicesBadge.textContent = String(nowOnline.size);
  els.devicesBadge.hidden = nowOnline.size === 0;

  const sig = JSON.stringify(sats.map((s) => [s.id, s.online, s.muted, s.hw_muted, s.volume, s.name]));
  if (sig === lastSatSig && !force) return;
  // Don't rebuild the list mid-rename — it would blow away the open input. The
  // commit/cancel path clears satEditing and forces a reload, which bypasses this.
  if (satEditing !== null && !force) return;
  lastSatSig = sig;

  els.satList.innerHTML = "";
  els.satEmpty.hidden = sats.length > 0;
  els.satCount.textContent = String(sats.length);
  els.satCount.hidden = sats.length === 0;
  for (const s of sats) els.satList.appendChild(renderSatellite(s));
}
function renderSatellite(s) {
  const li = document.createElement("li"); li.className = "sat-item" + (s.online ? "" : " offline");
  const head = document.createElement("div"); head.className = "sat-head";
  const dot = document.createElement("span"); dot.className = "sat-dot" + (s.online ? " online" : ""); dot.title = s.online ? "online" : "offline";
  const name = document.createElement("span"); name.className = "sat-name"; name.textContent = s.name;
  name.title = "Click to rename"; name.tabIndex = 0; name.setAttribute("role", "button");
  name.addEventListener("click", () => beginRename(li, s));
  name.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); beginRename(li, s); } });
  const mute = document.createElement("button"); mute.className = "sat-mute" + (s.muted ? " muted" : ""); mute.type = "button";
  mute.textContent = s.muted ? "🔇" : "🎤";
  mute.title = s.muted ? "Assistant muted — tap to unmute" : "Mute assistant (ignore wake word)";
  mute.addEventListener("click", () => setSatellite(s.id, { muted: !s.muted }));
  head.append(dot, name);
  if (s.hw_muted) {
    const hw = document.createElement("span"); hw.className = "sat-hwmute"; hw.textContent = "🔴";
    hw.title = "Muted on the device (physical mute button)"; head.append(hw);
  }
  head.append(mute);
  if (!s.online) {
    const rm = document.createElement("button"); rm.className = "sat-remove"; rm.type = "button"; rm.textContent = "✕"; rm.title = "Forget this device";
    rm.addEventListener("click", () => removeSatellite(s.id)); head.append(rm);
  }
  const vol = document.createElement("div"); vol.className = "sat-vol";
  const icon = document.createElement("span"); icon.className = "sat-vol-icon"; icon.textContent = "🔊";
  const slider = document.createElement("input"); slider.type = "range"; slider.min = "0"; slider.max = "100";
  slider.value = s.volume; slider.className = "sat-vol-slider"; slider.style.setProperty("--fill", s.volume + "%");
  const val = document.createElement("span"); val.className = "sat-vol-val"; val.textContent = s.volume + "%";
  slider.addEventListener("input", () => { val.textContent = slider.value + "%"; slider.style.setProperty("--fill", slider.value + "%"); });
  slider.addEventListener("change", () => setSatellite(s.id, { volume: Number(slider.value) }));
  vol.append(icon, slider, val);
  li.append(head, vol);
  return li;
}
// Swap a satellite's name label for an input to rename it. Enter/blur commits,
// Escape cancels; clearing the field reverts to the device's self-reported name.
// The name is sticky server-side (`display_name`), so heartbeats won't overwrite it.
function beginRename(li, s) {
  if (satEditing !== null) return;
  satEditing = s.id;
  const nameEl = li.querySelector(".sat-name");
  const input = document.createElement("input");
  input.type = "text"; input.className = "sat-name-edit"; input.value = s.name;
  input.maxLength = 40; input.setAttribute("aria-label", "Device name");
  input.placeholder = s.default_name || "device name";
  nameEl.replaceWith(input);
  input.focus(); input.select();
  let done = false;
  const finish = (commit) => {
    if (done) return; done = true;
    satEditing = null;
    const v = input.value.trim();
    if (commit && v !== s.name) setSatellite(s.id, { name: v });   // "" ⇒ revert to device name
    else loadSatellites(true);                                     // cancel/no-op → redraw the label
  };
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); finish(true); }
    else if (e.key === "Escape") { e.preventDefault(); finish(false); }
  });
  input.addEventListener("blur", () => finish(true));
}
async function setSatellite(id, patch) {
  try {
    await fetch(`/api/satellites/${id}`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(patch) });
  } catch { /* poll reconciles */ }
  loadSatellites(true);
}
async function removeSatellite(id) {
  try { await fetch(`/api/satellites/${id}`, { method: "DELETE" }); } catch { /* ignore */ }
  loadSatellites(true);
}

// ── Dashboard (Homepage replacement: editable link groups) ──────────────────
let dashState = null;      // the whole {version, groups:[{title, tiles:[…]}]} document
let dashEditing = false;
let dashDrag = null;       // in-flight drag payload: {kind:"tile"|"group", gi, ti?}

// Reconcile the dashboard view against the URL hash (the route's source of truth).
// Called on load and on every hashchange (button clicks, browser back/forward,
// pasted links). Both open/close are guarded, so this never loops.
function syncRoute() {
  const wantDash = location.hash === "#dashboard";
  if (wantDash && els.dashboard.hidden) openDashboard();
  else if (!wantDash && !els.dashboard.hidden) closeDashboard();
}

function openDashboard() {
  closeOverlays();
  els.dashboard.hidden = false;
  setDashEdit(false);
  loadDashboard();
  // Reflect the view in the URL so it's deep-linkable and browser Back closes it.
  // Assignment (not replaceState) adds a history entry; the no-op guard avoids a
  // redundant entry when we arrived here from the hash itself.
  if (location.hash !== "#dashboard") location.hash = "dashboard";
}
function closeDashboard() {
  if (dashEditing) saveDashboard();   // flush any pending edit
  dashEditing = false;
  els.dashboard.hidden = true;
  // Drop the route without a trailing "#" and without firing hashchange.
  if (location.hash === "#dashboard")
    history.replaceState(null, "", location.pathname + location.search);
}
function setDashEdit(on) {
  dashEditing = on;
  els.dashEditBtn.hidden = on;
  els.dashDone.hidden = !on;
  els.dashAddGroup.hidden = !on;
  els.dashReset.hidden = !on;
  dashStatusMsg(on ? "Editing — changes save automatically." : "");
  renderDashboard();
}
async function loadDashboard() {
  try {
    const r = await fetch("/api/dashboard");
    if (!r.ok) throw new Error("dashboard " + r.status);
    dashState = await r.json();
  } catch (e) {
    dashState = dashState || { version: 1, groups: [] };
    dashStatusMsg("Couldn't load dashboard: " + e.message, true);
  }
  renderDashboard();
}
// Persist the whole document. We deliberately do NOT adopt the server's response
// back into dashState: the open edit inputs hold references to the current tile
// objects, and swapping them out mid-edit would strip focus / lose keystrokes. The
// server-normalized form (filled ids, dropped-empty tiles) is picked up on reload.
async function saveDashboard() {
  if (!dashState) return;
  try {
    const r = await fetch("/api/dashboard", {
      method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(dashState),
    });
    if (!r.ok) throw new Error("save " + r.status);
    if (dashEditing) dashStatusMsg("Saved ✓");
  } catch (e) { dashStatusMsg("Save failed: " + e.message, true); }
}
function saveRender() { saveDashboard(); renderDashboard(); }
function dashStatusMsg(text, isError = false) {
  els.dashStatus.textContent = text;
  els.dashStatus.classList.toggle("error", isError);
}
function moveItem(arr, i, dir) {
  const j = i + dir;
  if (j < 0 || j >= arr.length) return;
  [arr[i], arr[j]] = [arr[j], arr[i]];
  saveRender();
}
// Move a tile out of (fromGi,fromTi) and insert before index `toIndex` in group
// `toGi` — supports moving a tile to another group. `toIndex === length` appends.
function moveTile(fromGi, fromTi, toGi, toIndex) {
  const groups = dashState.groups;
  if (!groups[fromGi] || !groups[toGi]) return;
  const [tile] = groups[fromGi].tiles.splice(fromTi, 1);
  if (fromGi === toGi && fromTi < toIndex) toIndex--;   // removal shifted the target
  groups[toGi].tiles.splice(toIndex, 0, tile);
  saveRender();
}
function moveGroup(fromGi, toGi) {
  const groups = dashState.groups;
  if (fromGi === toGi || !groups[fromGi] || !groups[toGi]) return;
  const [g] = groups.splice(fromGi, 1);
  if (fromGi < toGi) toGi--;
  groups.splice(toGi, 0, g);
  saveRender();
}
// Drag-and-drop reorder. A small grip handle initiates the drag (so the tile's
// text inputs stay fully editable); the ↑/↓ buttons remain the touch/keyboard
// fallback since HTML5 DnD doesn't fire on touch.
function makeGrip(title) {
  const g = document.createElement("span");
  g.className = "dash-grip"; g.textContent = "⠿"; g.title = title;
  g.setAttribute("aria-hidden", "true");
  return g;
}
function clearDropMarks() {
  els.dashboardBody.querySelectorAll(".drag-over, .drag-over-end")
    .forEach((n) => n.classList.remove("drag-over", "drag-over-end"));
}
function makeDraggable(el, grip, payload) {
  grip.addEventListener("mousedown", () => { el.draggable = true; });
  grip.addEventListener("mouseup", () => { el.draggable = false; });   // grabbed but not dragged
  el.addEventListener("dragstart", (e) => {
    // A tile lives inside a group <section> that is ALSO a drag source; without
    // this, the tile's dragstart bubbles up and the group handler overwrites the
    // payload to {kind:"group"}, so cross-group tile drops silently fail.
    e.stopPropagation();
    dashDrag = payload;
    e.dataTransfer.effectAllowed = "move";
    e.dataTransfer.setData("text/plain", "");   // Firefox requires some data
    el.classList.add("dragging");
  });
  el.addEventListener("dragend", () => {
    el.draggable = false; el.classList.remove("dragging"); clearDropMarks(); dashDrag = null;
  });
}
function attachTileDrop(card, gi, ti) {
  card.addEventListener("dragover", (e) => {
    if (dashDrag?.kind !== "tile") return;
    e.preventDefault(); e.dataTransfer.dropEffect = "move";
    clearDropMarks(); card.classList.add("drag-over");
  });
  card.addEventListener("drop", (e) => {
    if (dashDrag?.kind !== "tile") return;
    e.preventDefault(); e.stopPropagation();          // don't also append via the grid
    moveTile(dashDrag.gi, dashDrag.ti, gi, ti);       // insert before this tile
    dashDrag = null;
  });
}
function attachGridDnD(grid, gi) {
  grid.addEventListener("dragover", (e) => {
    if (dashDrag?.kind !== "tile") return;
    e.preventDefault(); e.dataTransfer.dropEffect = "move";
    if (e.target === grid) { clearDropMarks(); grid.classList.add("drag-over-end"); }
  });
  grid.addEventListener("drop", (e) => {
    if (dashDrag?.kind !== "tile") return;             // only fires on empty grid area
    e.preventDefault();
    moveTile(dashDrag.gi, dashDrag.ti, gi, dashState.groups[gi].tiles.length);  // append
    dashDrag = null;
  });
}
function attachGroupDnD(section, head, gi) {
  const grip = makeGrip("Drag to reorder group");
  grip.classList.add("dash-grip-group");
  head.insertBefore(grip, head.firstChild);
  makeDraggable(section, grip, { kind: "group", gi });
  section.addEventListener("dragover", (e) => {
    if (dashDrag?.kind !== "group") return;
    e.preventDefault(); e.dataTransfer.dropEffect = "move";
    clearDropMarks(); section.classList.add("drag-over");
  });
  section.addEventListener("drop", (e) => {
    if (dashDrag?.kind !== "group") return;
    e.preventDefault();
    moveGroup(dashDrag.gi, gi);
    dashDrag = null;
  });
}
function dashAddGroup() {
  dashState.groups.push({ id: "", title: "New group", tiles: [] });
  saveRender();
}
async function dashReset() {
  if (!confirm("Reset the dashboard to the imported defaults? Your edits will be lost.")) return;
  try {
    const r = await fetch("/api/dashboard/reset", { method: "POST" });
    dashState = await r.json();
    dashStatusMsg("Reset to defaults ✓");
  } catch (e) { dashStatusMsg("Reset failed: " + e.message, true); }
  renderDashboard();
}

function renderDashboard() {
  const body = els.dashboardBody;
  body.innerHTML = "";
  const groups = (dashState && dashState.groups) || [];
  if (!groups.length) {
    const p = document.createElement("p"); p.className = "setting-hint";
    p.textContent = dashEditing ? "Empty — tap “+ Group” to start." : "No links yet. Tap Edit to add some.";
    body.appendChild(p); return;
  }
  groups.forEach((g, gi) => {
    const sec = document.createElement("section"); sec.className = "dash-group";
    if (dashEditing) {
      const head = document.createElement("div"); head.className = "dash-group-head";
      const title = document.createElement("input");
      title.className = "dash-group-title-input"; title.value = g.title; title.placeholder = "Group name";
      title.addEventListener("input", () => { g.title = title.value; });
      title.addEventListener("change", saveDashboard);
      head.append(
        title,
        miniBtn("↑", "Move group up", () => moveItem(groups, gi, -1)),
        miniBtn("↓", "Move group down", () => moveItem(groups, gi, 1)),
        miniBtn("+ Link", "Add a link", () => { g.tiles.push({ id: "", name: "", href: "", icon: "", desc: "" }); saveRender(); }),
        miniBtn("✕", "Delete group", () => { groups.splice(gi, 1); saveRender(); }, true),
      );
      sec.appendChild(head);
      attachGroupDnD(sec, head, gi);
    } else {
      const h = document.createElement("h2"); h.className = "dash-group-title"; h.textContent = g.title;
      sec.appendChild(h);
    }
    const grid = document.createElement("div"); grid.className = "dash-grid";
    (g.tiles || []).forEach((t, ti) => grid.appendChild(dashEditing ? tileEditCard(g, gi, t, ti) : tileLink(t)));
    if (dashEditing) attachGridDnD(grid, gi);
    sec.appendChild(grid);
    body.appendChild(sec);
  });
}
function miniBtn(label, title, onClick, danger = false) {
  const b = document.createElement("button"); b.type = "button";
  b.className = "dash-mini" + (danger ? " danger" : "");
  b.textContent = label; b.title = title;
  b.addEventListener("click", onClick);
  return b;
}
function tileLink(t) {
  const a = document.createElement("a"); a.className = "dash-tile";
  if (t.href) { a.href = t.href; a.target = "_blank"; a.rel = "noopener noreferrer"; }
  else { a.style.cursor = "default"; }
  a.appendChild(tileIconEl(t));
  const tx = document.createElement("span"); tx.className = "dash-tx";
  const nm = document.createElement("span"); nm.className = "dash-name"; nm.textContent = t.name || t.href || "(untitled)";
  tx.appendChild(nm);
  if (t.desc) { const d = document.createElement("span"); d.className = "dash-desc"; d.textContent = t.desc; tx.appendChild(d); }
  a.appendChild(tx);
  return a;
}
function tileEditCard(g, gi, t, ti) {
  const card = document.createElement("div"); card.className = "dash-tile-edit";
  const field = (ph, key) => {
    const i = document.createElement("input"); i.placeholder = ph; i.value = t[key] || "";
    i.addEventListener("input", () => { t[key] = i.value; });
    i.addEventListener("change", saveDashboard);
    return i;
  };
  card.append(field("Name", "name"), field("https://…", "href"), field("icon — emoji or image URL (optional)", "icon"));
  const acts = document.createElement("div"); acts.className = "dash-tile-edit-actions";
  const grip = makeGrip("Drag to reorder (or move to another group)");
  acts.append(
    grip,
    miniBtn("↑", "Move up", () => moveItem(g.tiles, ti, -1)),
    miniBtn("↓", "Move down", () => moveItem(g.tiles, ti, 1)),
    miniBtn("✕", "Delete link", () => { g.tiles.splice(ti, 1); saveRender(); }, true),
  );
  card.appendChild(acts);
  makeDraggable(card, grip, { kind: "tile", gi, ti });
  attachTileDrop(card, gi, ti);
  return card;
}
// Icons, all self-hosted (no external CDN — the real service icons were vendored
// from the dashboard-icons set into /icons/dashboard/ at build time). Resolution:
// a local path or URL → <img>; a bare slug → the vendored /icons/dashboard/<slug>.svg;
// an emoji/symbol → text; nothing → a colored monogram. Any image that 404s falls
// back to the monogram, so a missing icon is never a broken image.
function tileIconEl(t) {
  const wrap = document.createElement("span"); wrap.className = "dash-ic";
  const ic = (t.icon || "").trim();
  const asImg = (src) => {
    const img = document.createElement("img"); img.src = src; img.alt = "";
    img.addEventListener("error", () => { wrap.innerHTML = ""; applyMonogram(wrap, t.name || t.href); });
    wrap.appendChild(img);
  };
  if (ic.startsWith("/") || /^https?:\/\//i.test(ic)) {
    asImg(ic);
  } else if (ic && /^[a-z0-9][a-z0-9._-]*$/i.test(ic)) {
    asImg("/icons/dashboard/" + ic + ".svg");   // a bare slug → the vendored set
  } else if (ic) {
    wrap.textContent = ic; wrap.style.fontSize = "18px";   // emoji / symbol
  } else {
    applyMonogram(wrap, t.name || t.href);
  }
  return wrap;
}
function applyMonogram(wrap, name) {
  const palette = ["#8fd0ff", "#7de3a0", "#e6a8d8", "#ffcf4a", "#c9a2ff", "#66b8ff", "#ff9d7a", "#7fe0d0"];
  const s = (name || "?").trim();
  const words = s.split(/\s+/).filter(Boolean);
  const text = (words.length >= 2 ? words[0][0] + words[1][0] : s.slice(0, 2) || "?").toUpperCase();
  let h = 0; for (const c of s) h = (h * 31 + c.charCodeAt(0)) & 0x7fffffff;
  const color = palette[h % palette.length];
  wrap.textContent = text; wrap.style.background = hexA(color, 0.18); wrap.style.color = color;
}

// ── Constellation HUD (live homelab vitals around the graph edges) ──────────
async function loadMetrics() {
  let data;
  try {
    const r = await fetch("/api/metrics");
    if (!r.ok) throw new Error("metrics " + r.status);
    data = await r.json();
  } catch { return; }          // transient — leave the last readout on screen
  renderHud(data);
}
// Edge slots framing the constellation, computed for however many readouts came
// back: the set is split down the middle and stacked column-major — first half
// down the left edge, second half down the right — evenly spread between
// HUD_TOP and HUD_BOTTOM so the bottom band stays clear for the legend.
const HUD_TOP = 4, HUD_BOTTOM = 72;      // % of the stage height
function hudSlots(n) {
  const perSide = Math.ceil(n / 2);
  const step = perSide > 1 ? (HUD_BOTTOM - HUD_TOP) / (perSide - 1) : 0;
  return Array.from({ length: n }, (_, i) => ({
    side: i < perSide ? "left" : "right",
    y: perSide > 1 ? HUD_TOP + (i % perSide) * step : (HUD_TOP + HUD_BOTTOM) / 2,
  }));
}
// Aesthetic placement (like GRAPH_LAYOUT): matched loosely against the host name —
// Alfred/Eleven/Nova/Kali down the left, Sam/Blue/Shuri/Frigate down the right.
// Anything unlisted (a host newly added to Beszel) keeps its server order (online
// first, then by name) and lands after these.
const HUD_ORDER = ["alfred", "eleven", "nova", "kali", "sam", "blue", "shuri", "frigate"];
function hudRank(name) {
  const n = (name || "").toLowerCase();
  const i = HUD_ORDER.findIndex((k) => n.includes(k));
  return i < 0 ? HUD_ORDER.length : i;
}
function renderHud(data) {
  const hud = els.stageHud;
  if (!hud) return;
  hud.innerHTML = "";
  const items = (data.hosts || []).map((h) => ({ rank: hudRank(h.name), el: hostHudItem(h) }));
  if (data && data.frigate) items.push({ rank: hudRank("frigate"), el: frigateHudItem(data.frigate) });
  items.sort((a, b) => a.rank - b.rank);   // stable: unlisted hosts keep server order
  const slots = hudSlots(items.length);
  items.forEach(({ el }, i) => {
    const s = slots[i];
    if (s.side === "right") { el.classList.add("right"); el.style.right = "10px"; }
    else el.style.left = "10px";
    el.style.top = s.y + "%";
    hud.appendChild(el);
  });
}
function hudPctColor(v) {
  if (v == null) return "var(--text)";
  if (v >= 90) return "var(--red)";
  if (v >= 70) return "var(--amber)";
  return "var(--text)";
}
function hudShortHost(name) {
  return (name || "").replace(/\bserver\b/i, "").trim().toUpperCase() || "?";
}
function hudNum(v) { return v == null ? "–" : Math.round(v) + "%"; }
function hudItemEl(label, up) {
  const el = document.createElement("div"); el.className = "hud-item";
  const head = document.createElement("div"); head.className = "hud-head";
  const dot = document.createElement("span"); dot.className = "hud-dot " + (up ? "up" : "down");
  const name = document.createElement("span"); name.className = "hud-name"; name.textContent = label;
  head.append(dot, name); el.appendChild(head);
  return el;
}
function hudExtraLabel(dev) {
  return /^md\d+/i.test(dev) ? "RAID" : dev.toUpperCase();   // mdadm arrays → "RAID"
}
// Tiny inline SVG sparkline of a host's recent load average (Beszel `1m` history).
// Load is core-relative, not a percent, so the line auto-scales to its own window
// — it shows the *shape/trend* of load, which is the useful part. No library.
const SVG_NS = "http://www.w3.org/2000/svg";
function hudSparkline(series) {
  if (!Array.isArray(series) || series.length < 2) return null;
  const W = 74, H = 13, pad = 1.5;
  const min = Math.min(...series), max = Math.max(...series);
  const span = max - min || 1;                 // flat series → a centered line
  const n = series.length;
  const pts = series.map((v, i) => {
    const x = pad + (i / (n - 1)) * (W - 2 * pad);
    const y = H - pad - ((v - min) / span) * (H - 2 * pad);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("class", "hud-spark");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("width", W); svg.setAttribute("height", H);
  const poly = document.createElementNS(SVG_NS, "polyline");
  poly.setAttribute("points", pts);
  svg.appendChild(poly);
  return svg;
}
function hostHudItem(h) {
  const el = hudItemEl(hudShortHost(h.name), h.status === "up");
  const r1 = document.createElement("div"); r1.className = "hud-row";
  r1.innerHTML = `CPU <b style="color:${hudPctColor(h.cpu)}">${hudNum(h.cpu)}</b> · MEM <b style="color:${hudPctColor(h.mem)}">${hudNum(h.mem)}</b>`;
  el.appendChild(r1);
  // Same size + bolding as the CPU/MEM line above: muted label, bold coloured value.
  if (h.disk != null) {
    const r2 = document.createElement("div"); r2.className = "hud-row";
    r2.innerHTML = `DISK <b style="color:${hudPctColor(h.disk)}">${hudNum(h.disk)}</b>`;
    el.appendChild(r2);
  }
  // Extra filesystems (e.g. a RAID array) on their own line, value colour-coded.
  const extra = Object.entries(h.extra || {});
  if (extra.length) {
    const r3 = document.createElement("div"); r3.className = "hud-row";
    r3.innerHTML = extra
      .map(([dev, pct]) => `${hudExtraLabel(dev)} <b style="color:${hudPctColor(pct)}">${hudNum(pct)}</b>`)
      .join(" · ");
    el.appendChild(r3);
  }
  // Load-average readout, with its sparkline (recent trend) on the row below.
  const spark = hudSparkline(h.load);
  if (spark) {
    const cur = h.load[h.load.length - 1];
    const r4 = document.createElement("div"); r4.className = "hud-row";
    r4.innerHTML = `LOAD <b>${cur.toFixed(2)}</b>`;
    el.appendChild(r4);
    const r5 = document.createElement("div"); r5.className = "hud-row hud-spark-row";
    r5.appendChild(spark);
    el.appendChild(r5);
  }
  return el;
}
function frigateHudItem(f) {
  const el = hudItemEl("FRIGATE", true);
  const r1 = document.createElement("div"); r1.className = "hud-row"; r1.textContent = `NVR · ${f.cameras} cams`;
  el.appendChild(r1);
  return el;
}

// ── Notifications tray ─────────────────────────────────────────────────────
function pushNotif(text, dot) {
  notifications.unshift({ text, dot, ts: Math.floor(Date.now() / 1000) });
  notifications = notifications.slice(0, 40);
  if (els.notifPop.hidden) { notifUnread++; updateNotifBadge(); }
  if (!els.notifPop.hidden) renderNotifs();
}
function updateNotifBadge() {
  els.notifBadge.hidden = notifUnread === 0;
  els.notifBadge.textContent = notifUnread > 9 ? "9+" : String(notifUnread);
}
function renderNotifs() {
  els.notifList.innerHTML = "";
  els.notifEmpty.hidden = notifications.length > 0;
  for (const n of notifications) {
    const li = document.createElement("li"); li.className = "notif-item";
    const dot = document.createElement("span"); dot.className = "notif-dot"; dot.style.background = n.dot;
    const body = document.createElement("div");
    const text = document.createElement("div"); text.className = "notif-text"; text.textContent = n.text;
    const time = document.createElement("div"); time.className = "notif-time"; time.textContent = timeAgo(n.ts);
    body.append(text, time); li.append(dot, body); els.notifList.appendChild(li);
  }
}

// ── Overlay orchestration ──────────────────────────────────────────────────
function toggleOverlay(kind) { if (openKind === kind) { closeOverlays(); return; } closeOverlays(); openKind = kind; openOverlay(kind); }
function openOverlay(kind) {
  els.backdrop.classList.add("open");
  if (kind === "convos") {
    els.backdrop.classList.add("dim");
    els.convos.classList.add("open");
    els.historyBtn.classList.add("is-open");
    loadConvoList();
  } else if (kind === "timers") {
    els.timersPop.hidden = false; els.timerPill.classList.add("is-open"); loadTimers();
  } else if (kind === "devices") {
    els.devicesPop.hidden = false; els.devicesBtn.classList.add("is-open"); loadSatellites(true);
  } else if (kind === "notif") {
    els.notifPop.hidden = false; els.notifBtn.classList.add("is-open");
    notifUnread = 0; updateNotifBadge(); renderNotifs();
  }
}
function closeOverlays() {
  els.convos.classList.remove("open");
  els.timersPop.hidden = true; els.devicesPop.hidden = true; els.notifPop.hidden = true;
  els.backdrop.classList.remove("open", "dim");
  els.historyBtn.classList.remove("is-open");
  els.timerPill.classList.remove("is-open");
  els.devicesBtn.classList.remove("is-open");
  els.notifBtn.classList.remove("is-open");
  openKind = null;
}

// ── Speech ───────────────────────────────────────────────────────────────
async function speak(text, bubble) {
  setNovaStatus("speaking");
  try {
    const res = await fetch("/api/tts", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }),
    });
    if (!res.ok) throw new Error("tts " + res.status);
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    bubble?.classList.add("speaking");
    els.player.src = url;
    await els.player.play().catch(() => {});
    await new Promise((resolve) => { els.player.onended = resolve; els.player.onerror = resolve; });
    URL.revokeObjectURL(url);
  } catch (err) { showToast("Speech failed: " + err.message, "error"); }
  finally { bubble?.classList.remove("speaking"); }
}
function stopPlayback() { if (!els.player.paused) { els.player.pause(); els.player.currentTime = 0; } }

// ── UI helpers ───────────────────────────────────────────────────────────
function addMessage(role, text, timeStr) {
  els.hint?.remove();
  const row = document.createElement("div");
  row.className = "msg " + (role === "user" ? "user" : "assistant");
  const bubble = document.createElement("div"); bubble.className = "bubble"; bubble.textContent = text;
  const meta = document.createElement("div"); meta.className = "meta"; meta.textContent = timeStr ?? fmtClock(new Date());
  row.append(bubble, meta);
  els.transcript.appendChild(row);
  bubble.rowEl = row; bubble.metaEl = meta;
  scrollDown();
  return bubble;
}
function fmtClock(d) { return d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit", second: "2-digit", hour12: true }); }
function tickClock() {
  const now = new Date();
  const date = now.toLocaleDateString([], { month: "short", day: "numeric" });
  const time = now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: false });
  els.clock.textContent = `${date} · ${time}`;
}
function scrollDown() { els.transcript.scrollTop = els.transcript.scrollHeight; updateThreadFade(); }
// Only fade the transcript's top edge once something has scrolled above it, so a
// short thread's first message isn't clipped (#74).
function updateThreadFade() {
  els.transcript.classList.toggle("faded-top", els.transcript.scrollTop > 6);
}

function showToast(text, kind = "ok") {
  els.toast.innerHTML = "";
  const dot = document.createElement("span"); dot.className = "toast-dot";
  const t = document.createElement("span"); t.className = "toast-text"; t.textContent = text;
  if (kind === "error") {
    els.toast.style.borderColor = "rgba(255,77,94,0.6)";
    dot.style.background = "#ff4d5e"; dot.style.boxShadow = "0 0 10px #ff4d5e"; t.style.color = "#ffb3ba";
  } else {
    els.toast.style.borderColor = ""; dot.style.background = ""; dot.style.boxShadow = ""; t.style.color = "";
  }
  els.toast.append(dot, t);
  els.toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { els.toast.hidden = true; }, 3600);
}

function setBusy(state) {
  busy = state;
  els.micBtn.disabled = state && !recording;
  els.executeBtn.disabled = state;
}
