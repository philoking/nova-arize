"""Runtime configuration for the Nova Voice backend.

Every value is overridable via environment variables so the same image runs in
local dev (talking to Nova over the LAN) and in a container on Nova itself
(talking to the services over the Docker host gateway). See ``.env.example``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    val = os.environ.get(name, "").strip()
    return val or default


# The assistant shares one identity across surfaces; only the delivery
# constraints differ by response mode. {name} is filled from the configured
# assistant name so re-branding is a single env var.
DEFAULT_VOICE_PROMPT = (
    "You are {name}, a private assistant running entirely on a home server. "
    "You are speaking out loud, so keep replies short, natural, and conversational "
    "— usually one to three sentences. Do not use markdown, bullet points, "
    "emoji, or code blocks; write the way a person talks. If you don't know "
    "something, say so plainly. Be matter-of-fact and direct; never cute, flirty, "
    "or playful."
)

DEFAULT_TEXT_PROMPT = (
    "You are {name}, a private assistant running entirely on a home server, "
    "responding in a text interface. Be clear, helpful, and direct. You may use "
    "markdown (lists, code blocks, headings) when it makes the answer easier to "
    "read. Prefer concise answers, but give as much detail as the task needs. If "
    "you don't know something, say so plainly."
)

# Used for opt-in escalation to the Claude API - reserved for research/writing
# tasks that exceed the local model. The user explicitly confirms each escalation
# (see the confirm-before-send gate), so this brain is never reached silently.
DEFAULT_ESCALATION_PROMPT = (
    "You are {name}'s research and writing assistant. The user has deliberately "
    "handed you a task that needs deeper reasoning, longer-form writing, or web "
    "research than their local model can do well. Produce a thorough, well-"
    "structured answer. Use markdown where it aids readability. When you search "
    "the web, cite your sources with links. Lead with the outcome, then supporting "
    "detail."
)


@dataclass(frozen=True)
class Settings:
    # Upstream AI services. Defaults point at Nova over the LAN so `uvicorn`
    # works out of the box during local development. The Docker deploy overrides
    # these to reach the host-published ports via host.docker.internal.
    stt_url: str = field(default_factory=lambda: _env("NOVA_VOICE_STT_URL", "http://nova.example.internal:8001"))
    tts_url: str = field(default_factory=lambda: _env("NOVA_VOICE_TTS_URL", "http://nova.example.internal:8880"))
    llm_url: str = field(default_factory=lambda: _env("NOVA_VOICE_LLM_URL", "http://nova.example.internal:7869"))

    # Assistant identity. Re-brand the whole thing by setting NOVA_ASSISTANT_NAME.
    assistant_name: str = field(default_factory=lambda: _env("NOVA_ASSISTANT_NAME", "Nova"))

    # Model + voice defaults.
    model: str = field(default_factory=lambda: _env("NOVA_VOICE_MODEL", "qwen3:8b"))
    voice: str = field(default_factory=lambda: _env("NOVA_VOICE_TTS_VOICE", "af_heart"))

    # Per-mode system prompts. Empty = use the built-in default for that mode.
    # NOVA_VOICE_SYSTEM_PROMPT is honoured for the voice prompt for back-compat.
    voice_prompt: str = field(default_factory=lambda: _env("NOVA_VOICE_SYSTEM_PROMPT", ""))
    text_prompt: str = field(default_factory=lambda: _env("NOVA_TEXT_SYSTEM_PROMPT", ""))

    # qwen3 is a "thinking" model; disable the <think> reasoning trace so voice
    # replies are fast and we never speak the model's scratch work.
    think: bool = field(default_factory=lambda: _env("NOVA_VOICE_THINK", "0") == "1")

    # Home Assistant (optional). When ha_url + ha_token are set, the assistant
    # gains tools to query and control HA. Empty token = HA tools disabled.
    ha_url: str = field(default_factory=lambda: _env("NOVA_VOICE_HA_URL", "https://homeassistant.example.internal").rstrip("/"))
    ha_token: str = field(default_factory=lambda: _env("NOVA_VOICE_HA_TOKEN", ""))
    # Domains the assistant may control via call_service. State queries are not
    # restricted. `switch` is included because many lights are exposed as switch
    # entities; `lock` lets it lock/unlock doors on the curated allowlist. Add
    # scenes/fans/media_player/cover/etc. as needed.
    ha_control_domains: str = field(default_factory=lambda: _env("NOVA_VOICE_HA_CONTROL_DOMAINS", "light,climate,switch,lock"))
    # The assistant sees and controls ONLY entities carrying this HA label - a
    # curated allowlist so it can never target a wrong device out of the ~2k in
    # the registry. Curate by adding the label (and clear friendly names) in HA.
    ha_label: str = field(default_factory=lambda: _env("NOVA_VOICE_HA_LABEL", "nova"))
    # How long (s) the labeled-device list is cached before re-fetching from HA.
    ha_devices_ttl: int = field(default_factory=lambda: int(_env("NOVA_VOICE_HA_DEVICES_TTL", "300")))
    # Set to 0 if HA is fronted by an internal CA the container doesn't trust.
    ha_verify_ssl: bool = field(default_factory=lambda: _env("NOVA_VOICE_HA_VERIFY_SSL", "1") == "1")
    # Safety valve on the tool-call loop.
    max_tool_rounds: int = field(default_factory=lambda: int(_env("NOVA_VOICE_MAX_TOOL_ROUNDS", "5")))

    # Claude escalation (optional). When a key is set, the user can opt in to
    # hand a single turn to the Claude API for research/writing - always behind
    # the confirm-before-send gate. Claude never drives the local tools; it only
    # uses Anthropic's own server-side web tools for research. Empty key = off.
    claude_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY", ""))
    # The DEEP escalation brain - used for the upfront path (research, long-form
    # writing, deep reasoning, "ask Claude"), where big-model quality is the point.
    claude_model: str = field(default_factory=lambda: _env("NOVA_CLAUDE_MODEL", "claude-opus-4-8"))
    # The FAST/cheap escalation brain - used for the reactive path (qwen3 tried
    # and struggled: a punt or a fumbled tool chain). Cheap enough to escalate
    # liberally, and far more reliable than qwen3:8b at multi-step tool chains.
    # Both tiers need claude_enabled and are gated by the single auto_escalate
    # toggle (off ⇒ nothing leaves the LAN).
    claude_fast_model: str = field(default_factory=lambda: _env("NOVA_CLAUDE_FAST_MODEL", "claude-haiku-4-5"))
    claude_max_tokens: int = field(default_factory=lambda: int(_env("NOVA_CLAUDE_MAX_TOKENS", "8000")))
    claude_prompt: str = field(default_factory=lambda: _env("NOVA_CLAUDE_SYSTEM_PROMPT", ""))

    # Auto-escalation: run qwen3 first and, when it struggles (empty answer or a
    # punt, see escalation.py), silently retry on Claude with the same local tools.
    # Off by default because it changes the privacy posture: data leaves the LAN
    # without a per-turn confirm. A struggle retries on the fast tier (Haiku); the
    # upfront path uses the deeper model (Opus 4.8).
    auto_escalate: bool = field(default_factory=lambda: _env("NOVA_VOICE_AUTO_ESCALATE", "0") == "1")

    # Phone notifications over MQTT (#16). A reminder/timer/alarm firing publishes
    # JSON to the broker; one Home Assistant automation subscribes and pushes it via
    # the companion app, so reminders arrive with the PWA closed. These are only
    # defaults: toggle, categories, address and topic are user-editable at runtime.
    # Broker auth is env-only; leave blank for anonymous.
    notifications_enabled: bool = field(default_factory=lambda: _env("NOVA_VOICE_NOTIFICATIONS_ENABLED", "0") == "1")
    notification_categories: str = field(default_factory=lambda: _env("NOVA_VOICE_NOTIFICATION_CATEGORIES", "reminders,timers"))
    mqtt_address: str = field(default_factory=lambda: _env("NOVA_VOICE_MQTT_ADDRESS", ""))
    mqtt_topic: str = field(default_factory=lambda: _env("NOVA_VOICE_MQTT_TOPIC", "nova/notify"))
    mqtt_username: str = field(default_factory=lambda: _env("NOVA_VOICE_MQTT_USERNAME", ""))
    mqtt_password: str = field(default_factory=lambda: _env("NOVA_VOICE_MQTT_PASSWORD", ""))

    # Obsidian notes (optional). Points at the "Local REST API" plugin running
    # in Obsidian on the user's machine, reached over the LAN. Both a URL and a
    # key are required to enable the notes tools.
    obsidian_url: str = field(default_factory=lambda: _env("NOVA_OBSIDIAN_URL", "").rstrip("/"))
    obsidian_api_key: str = field(default_factory=lambda: _env("NOVA_OBSIDIAN_API_KEY", ""))
    # Base folder new projects are created under, vault-relative.
    obsidian_projects_dir: str = field(default_factory=lambda: _env("NOVA_OBSIDIAN_PROJECTS_DIR", "Projects"))
    # Folder named checklists (shopping, to-do, …) live in - same place as the
    # daily notes. Vault-relative; empty means the vault root.
    obsidian_daily_dir: str = field(default_factory=lambda: _env("NOVA_OBSIDIAN_DAILY_DIR", "Daily Notes"))
    # Only relevant for an https:// URL (the plugin's self-signed cert); ignored
    # for plain http on the LAN.
    obsidian_verify_ssl: bool = field(default_factory=lambda: _env("NOVA_OBSIDIAN_VERIFY_SSL", "1") == "1")

    # Web search (optional). Points at a self-hosted SearXNG instance's JSON API,
    # so the assistant can look up current/external facts instead of disclaiming
    # internet access. Enabled only when a URL is set. SearXNG must have the
    # json format enabled (search.formats in its settings.yml).
    searxng_url: str = field(default_factory=lambda: _env("NOVA_SEARXNG_URL", "").rstrip("/"))
    # How many results to hand the model per search (kept small - it summarizes).
    searxng_results: int = field(default_factory=lambda: int(_env("NOVA_SEARXNG_RESULTS", "5")))
    # Set 0 if SearXNG is fronted by a cert the container doesn't trust.
    searxng_verify_ssl: bool = field(default_factory=lambda: _env("NOVA_SEARXNG_VERIFY_SSL", "1") == "1")

    # Shop-manual RAG (optional, #63). Upload PDF manuals, extract and chunk the
    # text, embed via Ollama, store the vectors in Qdrant, so the assistant answers
    # from your manuals instead of guessing. Enabled only when a Qdrant URL is set.
    # The embed model needs pulling on Nova once: ollama pull nomic-embed-text.
    qdrant_url: str = field(default_factory=lambda: _env("NOVA_QDRANT_URL", "").rstrip("/"))
    qdrant_collection: str = field(default_factory=lambda: _env("NOVA_QDRANT_COLLECTION", "nova_manuals"))
    # Optional Qdrant API key (sent as the api-key header); empty on a LAN.
    qdrant_api_key: str = field(default_factory=lambda: _env("NOVA_QDRANT_API_KEY", ""))
    # Set 0 if Qdrant is fronted by a cert the container doesn't trust.
    qdrant_verify_ssl: bool = field(default_factory=lambda: _env("NOVA_QDRANT_VERIFY_SSL", "1") == "1")
    # Ollama embedding model. 768-dim; must be installed on Nova. The vector size
    # is detected from the first embedding, so a different model just works.
    embed_model: str = field(default_factory=lambda: _env("NOVA_EMBED_MODEL", "nomic-embed-text"))
    # How many manual chunks to hand the model per search (kept small - it reads
    # them and answers in its own words).
    manuals_top_k: int = field(default_factory=lambda: int(_env("NOVA_MANUALS_TOP_K", "5")))
    # Chunking of the extracted PDF text, in characters (~4 chars per token).
    manuals_chunk_chars: int = field(default_factory=lambda: int(_env("NOVA_MANUALS_CHUNK_CHARS", "1200")))
    manuals_chunk_overlap: int = field(default_factory=lambda: int(_env("NOVA_MANUALS_CHUNK_OVERLAP", "200")))
    # Drop retrieved chunks below this cosine score (weak, off-topic matches).
    manuals_min_score: float = field(default_factory=lambda: float(_env("NOVA_MANUALS_MIN_SCORE", "0.3")))

    # Frigate NVR (optional, #68). With a URL set, a spoken camera name matches
    # against Frigate's list and the PWA opens a modal with the live feed. Frigate
    # bundles go2rtc, which restreams RTSP as browser-playable HLS; the backend
    # proxies that same-origin, so the browser never talks to Frigate directly.
    frigate_url: str = field(default_factory=lambda: _env("NOVA_FRIGATE_URL", "").rstrip("/"))
    # Set 0 if Frigate is fronted by a cert the container doesn't trust.
    frigate_verify_ssl: bool = field(default_factory=lambda: _env("NOVA_FRIGATE_VERIFY_SSL", "1") == "1")
    # How long (s) the camera list is cached before re-fetching from Frigate.
    frigate_cameras_ttl: int = field(default_factory=lambda: int(_env("NOVA_FRIGATE_CAMERAS_TTL", "300")))

    # Live homelab metrics for the constellation HUD (#79). Read-only, LAN-direct.
    # Beszel is a PocketBase monitoring hub - per-host CPU/mem/disk (auth by
    # email+password). Optional; empty ⇒ the source is skipped and the HUD shows
    # whatever's left (Frigate's NVR readout).
    beszel_url: str = field(default_factory=lambda: _env("NOVA_BESZEL_URL", "").rstrip("/"))
    beszel_username: str = field(default_factory=lambda: _env("NOVA_BESZEL_USERNAME", ""))
    beszel_password: str = field(default_factory=lambda: _env("NOVA_BESZEL_PASSWORD", ""))
    beszel_verify_ssl: bool = field(default_factory=lambda: _env("NOVA_BESZEL_VERIFY_SSL", "1") == "1")
    # How many recent Beszel history points feed each host's load-average sparkline
    # in the HUD (the finest `1m` bucket, ~every 15-20s). 0 disables the sparklines.
    beszel_load_points: int = field(default_factory=lambda: int(_env("NOVA_BESZEL_LOAD_POINTS", "40")))
    # How long (s) the aggregated metrics are cached (shared across surfaces).
    metrics_ttl: int = field(default_factory=lambda: int(_env("NOVA_VOICE_METRICS_TTL", "12")))

    # Arize AX tracing (optional). Traces the agent loop over OpenTelemetry /
    # OpenInference. Space id and key are named as Arize's onboarding presents them,
    # so the values paste straight across. An empty key leaves tracing inert.
    # NOTE: an enabled exporter sends prompts and tool I/O off the LAN, the one
    # deliberate exception to this app's "nothing leaves the network" rule.
    arize_space_id: str = field(default_factory=lambda: _env("ARIZE_SPACE_ID", ""))
    arize_api_key: str = field(default_factory=lambda: _env("ARIZE_API_KEY", ""))
    arize_project: str = field(default_factory=lambda: _env("NOVA_ARIZE_PROJECT", "nova-voice"))

    @property
    def arize_enabled(self) -> bool:
        return bool(self.arize_space_id and self.arize_api_key)

    @property
    def frigate_enabled(self) -> bool:
        return bool(self.frigate_url)

    @property
    def beszel_enabled(self) -> bool:
        return bool(self.beszel_url and self.beszel_username and self.beszel_password)

    @property
    def metrics_enabled(self) -> bool:
        return self.beszel_enabled or self.frigate_enabled

    @property
    def obsidian_enabled(self) -> bool:
        return bool(self.obsidian_url and self.obsidian_api_key)

    @property
    def websearch_enabled(self) -> bool:
        return bool(self.searxng_url)

    @property
    def manuals_enabled(self) -> bool:
        return bool(self.qdrant_url)

    @property
    def claude_enabled(self) -> bool:
        return bool(self.claude_api_key)

    def escalation_prompt(self) -> str:
        return self.claude_prompt or DEFAULT_ESCALATION_PROMPT.format(name=self.assistant_name)

    def prompt_for(self, response_mode: str) -> str:
        """The base system prompt for a ``voice`` or ``text`` response.

        An explicit env override is used verbatim; otherwise the built-in
        default for that mode is filled with the assistant's name.
        """
        if response_mode == "text":
            return self.text_prompt or DEFAULT_TEXT_PROMPT.format(name=self.assistant_name)
        return self.voice_prompt or DEFAULT_VOICE_PROMPT.format(name=self.assistant_name)

    @property
    def ha_enabled(self) -> bool:
        return bool(self.ha_url and self.ha_token)

    @property
    def ha_control_domain_set(self) -> set[str]:
        return {d.strip() for d in self.ha_control_domains.split(",") if d.strip()}

    # Where the built PWA lives, served as static files by the same app.
    frontend_dir: str = field(default_factory=lambda: _env(
        "NOVA_VOICE_FRONTEND_DIR",
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend"),
    ))

    # Where user-editable settings (e.g. the chosen voice) are persisted. These
    # are runtime overrides on top of the env defaults above - see
    # settings_store.py. In Docker this points at a mounted volume so the
    # choice survives image rebuilds; the local-dev default is a gitignored
    # data/ dir at the repo root.
    settings_file: str = field(default_factory=lambda: _env(
        "NOVA_VOICE_SETTINGS_FILE",
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "settings.json"),
    ))

    # Conversation history (SQLite). Persists every turn from every surface (web
    # + satellite) so past chats are listable and resumable. Same data/ dir /
    # mounted volume as the settings file. History older than history_days is
    # pruned so the log stays a rolling window.
    history_db: str = field(default_factory=lambda: _env(
        "NOVA_VOICE_HISTORY_DB",
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "nova.db"),
    ))
    history_days: int = field(default_factory=lambda: int(_env("NOVA_VOICE_HISTORY_DAYS", "7")))

    # The link dashboard document (Homepage replacement, #78). A JSON file of the
    # user's edited groups/tiles; absent ⇒ the packaged seed is served. Same
    # data/ dir / mounted volume as the settings + history stores.
    dashboard_file: str = field(default_factory=lambda: _env(
        "NOVA_VOICE_DASHBOARD_FILE",
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "dashboard.json"),
    ))

    # Where uploaded manual PDFs are kept (the originals; their vectors live in
    # Qdrant). Same data/ dir / mounted volume as the history DB.
    manuals_dir: str = field(default_factory=lambda: _env(
        "NOVA_MANUALS_DIR",
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "manuals"),
    ))

    # Verbose activity log (SQLite ring buffer) exposed at /api/log for the
    # Settings-screen viewer. log_level is the capture threshold; the buffer
    # is bounded by both an age cutoff (log_retention_hours) and a hard row
    # cap (log_max_rows). Shares the history DB / /data volume.
    log_level: str = field(default_factory=lambda: _env("NOVA_VOICE_LOG_LEVEL", "INFO"))
    log_retention_hours: int = field(default_factory=lambda: int(_env("NOVA_VOICE_LOG_RETENTION_HOURS", "48")))
    # A safety cap on top of the age cutoff. Verbose capture (every request) is
    # high-volume, so this is generous enough that the 48h window is the binding
    # bound in normal use, not the row count.
    log_max_rows: int = field(default_factory=lambda: int(_env("NOVA_VOICE_LOG_MAX_ROWS", "50000")))

    # Timezone the assistant reasons about clock times in (for reminders like "at
    # 5pm"). IANA name, e.g. "America/New_York"; empty = the server's local time
    # (the container's TZ). Countdown timers are timezone-independent.
    timezone: str = field(default_factory=lambda: _env("NOVA_VOICE_TIMEZONE", ""))

    # Persistent memory (durable facts about the user, injected into context). A
    # "mid"-term memory is kept this many days; "short" expires at end of day and
    # "long" never expires. At most memory_max_context facts are injected into
    # the prompt (newest per tier) to bound its size. Shares the history DB.
    memory_mid_days: int = field(default_factory=lambda: int(_env("NOVA_VOICE_MEMORY_MID_DAYS", "30")))
    memory_max_context: int = field(default_factory=lambda: int(_env("NOVA_VOICE_MEMORY_MAX_CONTEXT", "60")))

    # A satellite counts as "online" if it has sent a heartbeat within this many
    # seconds. Must comfortably exceed the satellite's poll interval plus a turn's
    # duration (it doesn't heartbeat mid-turn), so it doesn't flicker offline.
    satellite_ttl: int = field(default_factory=lambda: int(_env("NOVA_VOICE_SATELLITE_TTL", "30")))

    # Upstream request timeout (seconds). STT of a long clip or a slow first
    # token can take a while, so keep this generous.
    request_timeout: float = field(default_factory=lambda: float(_env("NOVA_VOICE_TIMEOUT", "120")))


settings = Settings()
