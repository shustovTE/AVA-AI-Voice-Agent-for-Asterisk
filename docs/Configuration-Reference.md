# Configuration Reference

This document explains every major option in `config/ai-agent.yaml`, the precedence model for greeting/persona, and the impact of fine‑tuning parameters across AudioSocket/ExternalMedia, VAD, Barge‑In, Streaming, and Providers.

## Local Override File (`ai-agent.local.yaml`)

Operator customizations are stored in `config/ai-agent.local.yaml` (git-ignored). At startup the engine deep-merges this file on top of the base `config/ai-agent.yaml`:

- **Base file** (`config/ai-agent.yaml`) — shipped golden defaults, git-tracked. Updated by upstream releases.
- **Local override** (`config/ai-agent.local.yaml`) — operator changes only. All Admin UI saves, CLI wizard writes, and `agent setup` output go here.

Keys in the local file win over the base file (deep merge — nested dicts are merged recursively, scalars are replaced). If the local file does not exist, the base file is used as-is.

This separation means `git pull` during updates will never conflict with operator config, eliminating the merge-conflict problem on `ai-agent.yaml`.

### ARI WebSocket silent-failure detection

The engine uses protocol-level WebSocket pings to detect an ARI connection that
has stopped carrying packets without a clean TCP close:

```yaml
asterisk:
  ws_ping_interval_sec: 10
  ws_ping_timeout_sec: 10
```

The nominal detection bound is the interval plus the timeout, so the shipped
defaults make `/ready` fail in approximately 20 seconds before the existing
reconnect supervisor takes over. Both values accept 5–60 seconds and require an
AI Engine restart. Values near the lower bound should be used only after load
testing because a severely stalled event loop can delay pong processing and
cause an unnecessary reconnect.

## Configuration Architecture

Starting in v4.0, the project added a **modular pipeline architecture** alongside monolithic provider support:

### Monolithic Providers
- **Single provider** handles STT, LLM, and TTS internally
- Examples: `openai_realtime`, `deepgram` Voice Agent
- Configuration: Set `default_provider: "openai_realtime"` or `default_provider: "deepgram"`
- **Best for**: Simplicity, fastest response times

### Pipeline Configurations
- **Separate providers** for STT, LLM, and TTS
- Examples: Local Hybrid (Vosk STT + OpenAI LLM + Piper TTS)
- Configuration: Define under `pipelines:` block and set `active_pipeline: "pipeline_name"`
- **Best for**: Flexibility, privacy (local audio processing), cost control

### Pipeline LLM Hangup Guardrail (hangup_call)

Some pipeline LLMs can be overly eager to emit `hangup_call`. A per-pipeline guardrail can require explicit end-of-call intent in the user's transcript before honoring `hangup_call`.

- `pipelines.<name>.options.llm.hangup_call_guardrail`: `true`/`false` (unset = auto; enabled by default for specific adapters)
- `pipelines.<name>.options.llm.hangup_call_guardrail_mode`: `relaxed`/`normal`/`strict` (unset = use global hangup policy mode)
- `pipelines.<name>.options.llm.hangup_call_guardrail_markers.end_call`: list of caller phrases that count as end-of-call intent (unset/empty = use global hangup policy defaults)

Streaming modular STT uses an engine-managed raw PCM16-LE mono 16 kHz bus.
`stream_format`, `encoding`, `sample_rate`, and `channels` are written by the
Admin UI for clarity but are not independently negotiable. Conflicting legacy
values are normalized at runtime and logged. Audio Profiles continue to control
wire/full-agent negotiation and do not change the modular STT bus format.

### Pipeline End of Turn (caller turn-taking)

Streaming modular STT returns a result at every phrase boundary, so the number
of results says nothing about whether the caller has finished a sentence. The
caller's turn therefore ends on silence: each new result restarts a window, and
the accumulated text goes to the LLM once the caller has been quiet for it.

- `pipelines.<name>.options.llm.end_of_turn_silence_ms`: silence that ends the
  caller's turn. Defaults to `700`. Raise it (900-1200) when callers are cut off
  working through a long sentence; lower it (400-600) for snappier replies. `0`
  answers every STT result immediately.
- `pipelines.<name>.options.llm.end_of_turn_max_wait_ms`: optional hard cap
  measured from the caller's first pending result, so someone who never pauses
  still gets an answer. Unset or `0` disables the cap, which is the default.

A window measured from the recognizer's result has a blind spot: a streaming
recognizer emits a result only after its own silence gate (T-one holds 600 ms),
so a caller who pauses and then goes on can never be bridged — the
continuation's result arrives only after they pause again. With Asterisk
`TALK_DETECT` enabled for the pipeline (`barge_in.pipeline_talk_detect_enabled`),
the same events that trigger barge-in decide the end of the turn instead: the
turn is held while Asterisk reports the caller talking and released a short
grace after it reports them quiet, so the silence that ends a turn is measured
from the caller's last sound. The silence itself is then tuned with
`barge_in.pipeline_talk_detect_silence_ms` (800-1000 ms suits a caller who
thinks aloud; every value is also the latency before an answer).

- `pipelines.<name>.options.llm.end_of_turn_source`: `auto` (default; the
  first detector available in this order: Silero VAD when `vad.silero_enabled`,
  Asterisk talk detection when it is enabled for the pipeline, the result
  window otherwise), `vad`, `talk_detect`, or `final` to pin one. A pinned
  `vad` without the model loaded falls back the same way as `auto`.
- `pipelines.<name>.options.llm.end_of_turn_talk_detect_grace_ms`: grace after
  Asterisk reports the caller quiet, or after a result that lands while they
  already are. Defaults to `250`: long enough for a result that is about to
  arrive to join the turn, short enough not to be felt.
- `pipelines.<name>.options.llm.end_of_turn_talk_detect_hold_ms`: how long a
  pending result is held while Asterisk keeps reporting speech without any
  newer result. Defaults to `8000`. A `ChannelTalkingFinished` is not
  guaranteed to arrive, and a caller still talking produces a result every
  phrase, so a long quiet hold means the end event was lost.

With Silero VAD enabled (see [Silero VAD](#silero-vad-pipelines) below) the
engine's own neural detector takes that role, and with
[Smart Turn](#smart-turn-pipelines) on top an incomplete thought holds the
turn beyond the stop: the turn is held while Silero
reports the caller talking and released `end_of_turn_talk_detect_grace_ms`
after it reports them quiet, so the pause a caller may take is
`vad.silero_stop_ms` plus that grace. Because the detector runs in the engine,
the recognizer is also told to finalize the moment the caller stops instead of
waiting out its own gate in real time, and the turn waits for that result:

- `pipelines.<name>.options.llm.end_of_turn_vad_final_wait_ms`: how long a
  turn waits for the result the recognizer was told to produce when Silero
  reported the caller quiet. Defaults to `1000`. The result normally lands
  well inside this; the bound only matters when it never comes (the caller's
  sound carried no words). Not applied when a result already arrived after
  the caller's last speech.

Only the resolved detector drives the turn: with Silero in charge, Asterisk
talk-detect events still serve barge-in and the inactivity watchdog but no
longer touch the end of turn, so the two cannot disagree about it.

Every `Caller turn ended on silence` line reports which `source` decided it
(`vad`, `talk_detect` or `final`), and `Pipeline end-of-turn policy resolved`
at call start reports whether Silero is tracking the call. The line's
`waited_sec` spans the whole turn from its first result, so read the wait a
caller actually felt from `quiet_ms` (how long the detector had reported them
quiet when the turn was released: the stop window plus the grace, plus any
wait for a result) and `since_result_ms` (the time since the last result).

Which fields apply depends on the resolved source, and the pipeline editor
greys out the rest: `end_of_turn_silence_ms` only ends a turn measured from
results (`final`); the grace and `end_of_turn_vad_final_wait_ms` only follow
a detector. The pause a caller may take is never set here with a detector in
charge: it is `vad.silero_stop_ms` for Silero VAD and
`barge_in.pipeline_talk_detect_silence_ms` for talk detection.

Length no longer decides anything, so a one-word "yes" is answered as promptly
as a paragraph. Every turn is logged as `Caller turn ended on silence` with the
number of merged results and how long the caller was given.

Superseded options in existing configs keep working with a warning:
`aggregation_silence_sec` and `aggregation_timeout_sec` are read as the silence
window and `aggregation_max_wait_sec` as the cap, all converted from seconds.
`aggregation_min_words`, `aggregation_min_chars`, and
`aggregation_wait_for_silence` are ignored and logged once per call as
`Ignoring superseded transcript aggregation options` — the word and character
thresholds are what answered callers mid-sentence.

### Pipeline Reply Length

A reply that stops mid-sentence or mid-word without any barge-in in the log
was cut by the LLM's token limit, not by turn-taking: the endpoint returns
`finish_reason: length` and the engine speaks what it got. Every
OpenAI-compatible LLM adapter (`openai`, `native_llm`, Mistral, Groq and the
other Chat Completions endpoints) now logs `LLM reply cut by max_tokens` with
the `max_tokens` in force and the length of the cut reply, and `finish_reason`
on every completed request. Raise `pipelines.<name>.options.llm.max_tokens`
(the sample config ships `200`, which is short for a Russian reply with a list
in it), or have the prompt ask for shorter, list-free answers, which also read
better through TTS.

### Pipeline Gated Audio

While the agent speaks, the caller's frames are withheld so the agent cannot
hear itself. Dropping them outright glues the audio either side of the gap
together, and a word straddling a short gap reaches the recognizer garbled. The
engine therefore feeds silence in their place, which keeps the recognizer's
timeline continuous.

- `pipelines.<name>.options.stt.gated_silence_ms`: how long silence is fed into
  one gated stretch. Defaults to `3000`. Each silent frame costs the recognizer
  one more inference, and past a few seconds the caller has long stopped talking
  and a splice is harmless, so the budget caps what an agent turn costs. `0`
  restores the previous behaviour of dropping the frames. The budget is renewed
  as soon as real audio flows again.

### Golden Baselines
See the validated configurations in `config/`:
- `ai-agent.golden-openai.yaml` - OpenAI Realtime (monolithic, fastest)
- `ai-agent.golden-deepgram.yaml` - Deepgram Voice Agent (monolithic, enterprise)
- `ai-agent.golden-google-live.yaml` - Google Live (monolithic, lowest latency)
- `ai-agent.golden-elevenlabs.yaml` - ElevenLabs Agent (monolithic, premium voice)
- `ai-agent.golden-local-hybrid.yaml` - Local Hybrid (pipeline, privacy-focused)

### Additional Pipeline Providers (v6.4.0+)
- **Azure Speech Service** — Modular STT (`azure_stt`) and TTS (`azure_tts`) pipeline adapters. See [Provider-Azure-Setup.md](Provider-Azure-Setup.md).
- **MiniMax LLM** — Pipeline LLM adapter (`minimax_llm`). See [Provider-MiniMax-Setup.md](Provider-MiniMax-Setup.md).
- **Telnyx AI Inference** — Pipeline LLM adapter. See [Provider-Telnyx-Setup.md](Provider-Telnyx-Setup.md).

For comprehensive inline documentation, refer to the golden baseline YAML files directly.

### Local AI Server Backends (v4.4.2+)

Environment variables for selecting local STT/TTS backends:

| Variable | Options | Default | Description |
|----------|---------|---------|-------------|
| `LOCAL_STT_BACKEND` | `vosk`, `sherpa`, `kroko`, `tone`, `onnx_asr`, `faster_whisper`, `whisper_cpp` | `vosk` | Speech-to-text engine |
| `LOCAL_TTS_BACKEND` | `piper`, `kokoro`, `melotts`, `silero` | `piper` | Text-to-speech engine |

**STT Backends**:
- **Vosk**: Offline ASR with good accuracy, multiple language models
- **Sherpa-ONNX**: Low-latency streaming ASR using ONNX runtime
- **Kroko**: High-quality streaming ASR with 12+ languages (requires API key for hosted mode)
- **T-one**: Native Russian telephony STT using the upstream streaming CTC pipeline
- **onnx-asr** (`onnx_asr`): GigaAM v3 (Sber) and NeMo FastConformer RU through ONNX Runtime, offline models behind the Silero VAD gate (a phrase is recognized once the caller pauses; CUDA when the GPU image and a GPU are present). `ONNX_ASR_MODEL` picks the model (`gigaam-v3-e2e-ctc` by default), fetched from Hugging Face on first start; requires `INCLUDE_ONNX_ASR=true`. See [LOCAL_ONLY_SETUP.md](LOCAL_ONLY_SETUP.md#supported-stt-models).
- **Faster-Whisper**: Whisper inference via `faster-whisper` (model IDs like `base`, `small`, etc., or a local model directory depending on your install)
- **Whisper.cpp**: Local GGML Whisper inference with multilingual language hints

**TTS Backends**:
- **Piper**: Fast local TTS with multiple voices
- **Kokoro**: High-quality neural TTS with natural prosody (voices: af_heart, af_bella, am_michael)
- **MeloTTS**: High-quality multilingual TTS with voice IDs (depends on installed voices/models)

**Model/voice identifiers**:
- `providers.local.stt_model` and `providers.local.tts_voice` are treated as **backend-specific identifiers** (paths for some backends, IDs for others) and are used by the Admin UI “switch/rebuild” flows.

See [LOCAL_ONLY_SETUP.md](LOCAL_ONLY_SETUP.md) for detailed configuration.

---

## Call Selection & Precedence (Provider / Pipeline / Agent)

On each call, the engine selects:
- an **agent** (greeting/prompt/tools/profile)
- a **provider mode** (full agent provider vs pipeline)

This selection is intentionally flexible so you can keep safe defaults while still overriding behavior per extension.

### Agent selection

> **v7.4 resolution is Agent-only.** After the one-time YAML → `agents.db`
> migration (see [OPERATOR_MIGRATION.md](OPERATOR_MIGRATION.md)), the engine resolves
> the agent for a call from `agents.db`, **not** from `ai-agent.yaml` at runtime. Editing
> a context directly in `ai-agent.yaml` or `config/contexts/*.yaml` after migration has
> **no runtime effect** — it only triggers a drift warning. Edit agents in
> **Admin UI → Agents**, or reconcile YAML changes via **Agents → Migration Status**.

Resolution order on each call:

1. `AI_AGENT` channel var (preferred) → agent looked up by slug in `agents.db`.
2. `AI_CONTEXT` channel var (deprecated compatibility alias) → agent looked up display-name-first.
3. Otherwise, the default agent in `agents.db`.

`AI_AGENT` wins if both `AI_AGENT` and `AI_CONTEXT` are set on the channel.

Before accepting calls, v7.4 atomically imports legacy YAML Contexts when `agents.db` is
empty. Runtime routing then reads Agents only and fails closed if the database is absent
or unreadable. There is no live YAML persona fallback in v7.4.

### Audio profile selection

Audio profiles control the call’s negotiated sample rates/encodings (telephony wire format, provider input/output format, and internal pacing). They are defined under `profiles:` in `config/ai-agent.yaml`.

#### Audio format resolution — single source of truth

Each leg of the media path has exactly one owner; everything else is derived
per call at setup:

| Leg | Owner (edit point) | Notes |
| --- | --- | --- |
| Asterisk wire (in/out) | Audio Profile → `transport_out` / `transport_in` | On AudioSocket, companded selections (μ-law/A-law) ride the lossless `slin@8000` signed-linear carrier; Asterisk transcodes to the trunk codec. On ExternalMedia the wire is owned by the Audio Transport RTP settings instead: `external_media.codec` — one codec, both directions, shared by every profile and Agent. |
| Internal processing rate | Audio Profile → `internal_rate_hz` | |
| Provider API boundary | Provider card → `provider_input_*` / `output_*` | Constrained by the adapter's declared capabilities; must match the provider-side (dashboard) audio settings. For pipeline Agents the profile's `provider_pref` is the preference instead. |
| Wire-facing provider fields (`input_encoding`, `input_sample_rate_hz`, `target_encoding`, `target_sample_rate_hz`) | Derived — not an edit point | The engine overwrites them per call from the resolved profile (`session.provider_overrides`); YAML values are legacy fallbacks only. |

The Admin UI shows the resolved paths on the dedicated **Audio Path** page
(Core Configuration → Audio Path, backed by `GET /api/config/audio/alignment`):
one diagram per Agent — Asterisk ═ transport wire ═ AI Engine ═ provider API ═
Provider (monolithic or pipeline, with the pipeline's STT/TTS components) —
plus findings whenever a stored value disagrees with what a call will actually
use — e.g. a provider `target_encoding` that the profile overrides, a provider
rate outside the adapter's supported set, or a profile whose transport legs
are ignored because the ExternalMedia RTP settings own the wire. The Agent editor renders the same diagram
live under its Audio Profile selector, and the per-call log line `Audio
profile resolved and applied` records the same chain at call setup.

Profile audio pairs are validated at load/save time: G.711 codecs (`ulaw`,
`mulaw`, `alaw`) are fixed at 8000 Hz and `slin`/`slin16` at 8000/16000 Hz — a
mismatched pair is rejected instead of breaking the stream mid-call. The Admin
UI locks the rate fields accordingly. A-law is accepted alongside μ-law for
`provider_pref` encodings, `transport_out`, and the ExternalMedia codec.

`profiles.<name>.transport_in` is an optional asymmetric inbound leg
(`encoding` + `sample_rate_hz`, same fixed-rate rules). When omitted, the
inbound leg mirrors `transport_out`, which is the telephony norm. AudioSocket
announces the actual inbound format per frame, so `transport_in` acts as the
declared expectation used for RTP decode fallbacks and diagnostics — set it
only when the inbound side genuinely differs from the outbound side.

Audio profiles are parsed once at call setup and never change mid-call. The
Admin UI therefore allows editing a profile that Agents are using after an
explicit confirmation: active calls keep the settings they connected with, and
new calls pick up the change after apply (hot reload or restart). Deleting a
profile that is still assigned to an Agent remains blocked.

The shipped `telephony_ulaw_8k` profile retains compatibility downsampling.
`telephony_enhanced_8k` is an opt-in profile with the same stable 8 kHz wire
contract plus alias-safe provider-output downsampling. It reduces sibilant hiss
when a provider emits 16 or 24 kHz PCM; native 8 kHz output is unchanged. To
roll back a test, assign the Agent back to `telephony_ulaw_8k`.

`wideband_pcm_16k` is the opt-in end-to-end wideband profile for AudioSocket.
It originates the media channel as `c(slin16)`, receives PCM16 at 16 kHz in
AudioSocket type `0x12`, and stamps outbound frames with the same type. This is
per call, so 8 and 16 kHz Agents can run concurrently. Full-agent providers use
their declared native PCM boundary (16 or 24 kHz), and supported modular TTS
adapters request PCM and convert once to the 16 kHz AudioSocket boundary. It
requires Asterisk 20.17+, 21.12+, 22.7+, or 23.1+ and a wideband caller leg
(for example G.722). Older or unrecognized Asterisk versions fail the call
closed with a remediation message. Use
`telephony_ulaw_8k` or `telephony_enhanced_8k` for PSTN/G.711 calls;
upsampling an 8 kHz trunk cannot restore frequencies that the trunk discarded.

ExternalMedia RTP supports the shipped `telephony_ulaw_8k` and
`telephony_enhanced_8k` profiles only. Do not assign `wideband_pcm_16k` while
`audio_transport: externalmedia` is active. RTP ExternalMedia has no SDP offer
and therefore cannot negotiate the dynamic payload mapping used by Asterisk for
`slin16`; returning payload type 118 produced no caller audio in live testing.
Asterisk Media over WebSocket can carry `slin16`, but AAVA does not currently
implement that separate transport. Use AudioSocket for supported wideband audio.

The wideband provider boundary is call-scoped and does not rewrite provider
defaults. OpenAI Realtime uses PCM at 24 kHz in both directions; Google Live
uses 16 kHz input and fixed 24 kHz output; ElevenLabs, Deepgram, and Grok use
their declared 16 kHz PCM route. Their output is converted to the 16 kHz wire
rate where necessary. Local TTS negotiates native `linear16@16000` output for
Piper and Kokoro (local or API mode); MeloTTS, Matcha, and Silero retain a
truthful μ-law/8 kHz fallback. The Local Hybrid Piper path is live-validated;
the local Kokoro CPU canary preserved correct 16 kHz media but failed the
interactive-latency gate because whole-utterance synthesis was approximately
real-time or slower. Kokoro API mode and the local full-agent path remain
untested. Use Piper for CPU deployments unless local Kokoro performance has
been measured on the target host. Switching the Agent back to an 8 kHz profile
is an immediate per-call rollback.

The profile field is `output_resampler: linear | bandlimited`. Provider and
pipeline fields default to `inherit`. Narrow overrides resolve in this order:
full-agent environment override, pipeline override, provider override, audio
profile, then the compatibility default. Existing profiles without this field
continue to use `linear`.

Highest priority first:

1. **Dialplan override**: `AI_AUDIO_PROFILE` (if set)
2. **Agent mapping**: the selected Agent's `audio_profile` value in `agents.db`
3. **Global default**: `profiles.default` (fallback is `telephony_ulaw_8k` if unset)

### Provider selection

Highest priority first:

1. **Dialplan override**: `AI_PROVIDER` (if set)
2. **Agent override**: the selected Agent's provider or pipeline in `agents.db`
3. **Global default**: `default_provider`

### Pipeline selection

If the selected provider path is a pipeline-based configuration, the engine uses:

- `active_pipeline` to determine which pipeline to run.
- If `active_pipeline` is unset/null, the engine falls back to the first available pipeline in `pipelines:`.

### Recommended approach

- Keep `default_provider` + `active_pipeline` set to a known-good baseline.
- Use `AI_AGENT` (preferred; `AI_CONTEXT` is the legacy equivalent) for persona/tool scoping.
- Use `AI_PROVIDER` only when you want an explicit per-extension override.

See also:
- [Installation Guide](INSTALLATION.md)
- [Transport Compatibility](Transport-Mode-Compatibility.md)

---

## Provider-start failure recovery

Provider and explicit pipeline startup failures use these root-level settings:

| Key | Default | Description |
|---|---|---|
| `on_provider_failure` | `announce_hangup` | `announce_hangup`, opt-in `dialplan_redirect`, or legacy `leave_open` |
| `provider_failure_prompt` | `sorry-youre-having-problems` | Asterisk sound played before the safe fallback hangup |
| `provider_failure_redirect_context` | unset | Required Asterisk context for `dialplan_redirect` |
| `provider_failure_redirect_extension` | `s` | Extension within the recovery context |
| `provider_failure_redirect_priority` | `1` | Priority within the recovery extension |

`dialplan_redirect` marks the caller as transferred before issuing ARI
`continueInDialplan`, so Stasis cleanup removes AAVA's bridge/media channels but
does not hang up the caller. The continuation is attempted once. A missing
context, ARI error, or rejected continuation falls back to the configured error
prompt and hangup; it never leaves the caller in silent dead air.

Example:

```yaml
on_provider_failure: dialplan_redirect
provider_failure_redirect_context: aava-provider-failure
provider_failure_redirect_extension: s
provider_failure_redirect_priority: 1
```

```asterisk
[aava-provider-failure]
exten => s,1,NoOp(AAVA provider startup failed)
 same => n,Playback(sorry-youre-having-problems)
 same => n,Goto(from-internal,6000,1)
```

The recovery route must not send the same channel directly back to the failing
AAVA agent, or the dialplan can create a retry loop.

---

## Outbound Campaign Dialer (Milestone 22)

Outbound calling is implemented as an **engine-driven scheduler + SQLite + ARI originate**, with **dialplan-assisted AMD** for voicemail detection.

### Assumptions (MVP)

- Your outbound **trunk(s) and outbound routes** are already configured in Asterisk/FreePBX.
- Outbound calls originate as **extension identity `6789`** (default), and routing happens via your existing FreePBX dialplan patterns.
- Campaign defaults and lead overrides should contain active Agent slugs; outbound channels set canonical `AI_AGENT` and retain `AI_CONTEXT` only as a v7.4 compatibility alias.
- Persistence reuses the existing Call History SQLite DB path (`CALL_HISTORY_DB_PATH`) by adding outbound tables.

### Key env vars

| Variable | Default | Description |
|----------|---------|-------------|
| `AAVA_OUTBOUND_EXTENSION_IDENTITY` | `6789` | Extension identity for FreePBX routing (sets `AMPUSER` + `CALLERID(num)` on originate). A lead's Caller ID override replaces it for that lead's calls. |
| `AAVA_OUTBOUND_AMD_CONTEXT` | `aava-outbound-amd` | Dialplan context name used for AMD hop (`continueInDialplan`) |
| `AAVA_OUTBOUND_PBX_TYPE` | `freepbx` | PBX-specific AAVA Campaign channel vars: `freepbx` \| `generic`. Legacy `vicidial` remains readable for one migration release but cannot be newly selected in the UI; use Call Scheduling → VICIdial Remote Agents. |
| `AAVA_OUTBOUND_DIAL_CONTEXT` | `from-internal` | Asterisk dialplan context for `Local/` channel origination |
| `AAVA_OUTBOUND_DIAL_PREFIX` | (empty) | Dial prefix prepended to phone number for AAVA Campaign carrier selection |
| `AAVA_OUTBOUND_CHANNEL_TECH` | `auto` | Channel tech for extension probing: `auto` \| `pjsip` \| `sip` \| `local_only` |
| `AAVA_OUTBOUND_ATTEMPT_STALE_SECONDS` | `120` | Shared startup/runtime timeout for attempts that never reach a live session; minimum `10` seconds |
| `AAVA_OUTBOUND_LEAD_IMPORT_MAX_BYTES` | `10485760` | Maximum uploaded CSV or `.xlsx` lead file size in bytes |
| `AAVA_OUTBOUND_LEAD_IMPORT_MAX_ROWS` | `10000` | Maximum data rows read from the first `.xlsx` worksheet; hard-capped at `100000` |
| `AAVA_MEDIA_DIR` | `/mnt/asterisk_media/ai-generated` | Where the Admin UI uploads voicemail drop `.ulaw` files |

### Dialplan requirements

- Create a custom dialplan context (default: `[aava-outbound-amd]`) that runs `AMD(${AAVA_AMD_OPTS})` and returns to Stasis with:
  - `outbound_amd` (action)
  - `${AAVA_ATTEMPT_ID}` (attempt correlation)
  - `${AMDSTATUS}` and `${AMDCAUSE}`

See `docs/contributing/milestones/milestone-22-outbound-campaign-dialer.md` for the full snippet and smoke test checklist.

### Originate-time channel variables and lead context

AAVA sends outbound channel variables in the ARI `POST /channels` JSON
`variables` object. This includes Agent/provider routing, outbound correlation,
FreePBX identity, and AMD/consent controls. Local-channel routes receive both
ordinary and inherited variants so the metadata survives the `;1`/`;2`
boundary.

Lead `custom_vars` never travel through channel variables and have no size
limit: the engine keeps them in attempt metadata and the durable SQLite store
and re-reads them by attempt id when the answered call returns to Stasis. If
the engine restarts while an originate is still ringing, it recovers the
unfinished attempt and lead from SQLite before starting the AI session;
unavailable or corrupt authoritative metadata fails closed. The value is
intentionally excluded from logs; use attempt, campaign, lead, and channel
identifiers when troubleshooting.

## VICIdial Remote Agents (Alpha)

VICIdial Remote Agents are separate from AAVA's native Outbound Campaign Dialer. VICIdial owns
campaign origination, customer channels, reports, dispositions, DNC, callbacks, and transfers;
AAVA supplies the mapped AI conversation. Configure connections and mappings under
**Call Scheduling → VICIdial Remote Agents** and follow the
[VICIdial Remote Agent Setup](Vicidial-Setup.md) acceptance sequence.

The connection stores environment-variable names rather than credential values. The UI defaults
to the following names, but a mapping may reference other valid environment-variable names:

| Variable | Default | Description |
|----------|---------|-------------|
| `VICIDIAL_API_USER` | (empty) | Dedicated least-privilege VICIdial API username referenced by the connection |
| `VICIDIAL_API_PASS` | (empty) | Dedicated VICIdial API password referenced by the connection |
| `VICIDIAL_DB_PATH` | `/app/data/operator/vicidial.db` | Shared SQLite store for connections, mappings, and readiness evidence |

Recreate `ai_engine` and `admin_ui` after changing credential environment variables so both
services receive the same values. Back up `vicidial.db` with the rest of `data/operator/`; it
contains configuration and verification evidence, but never the referenced API password.

## Canonical persona and greeting

- llm.initial_greeting: Text the agent speaks first (if provider supports explicit greeting or the engine plays via TTS).
- llm.prompt: The agent persona/instructions used by LLMs.
- Precedence at runtime:
  1) Provider/pipeline overrides (if explicitly set, e.g., `providers.openai_realtime.instructions`, `providers.deepgram.greeting`)
  2) `llm.prompt` and `llm.initial_greeting` in YAML
  3) Env defaults `AI_ROLE`, `GREETING`

## Voice (v7.3.0+)

- Voice precedence per call: per-call override → agent voice (agents.db, or the optional
  `voice:` key on a YAML context) → the provider's configured voice (default/fallback).
- Supported for per-agent override: OpenAI Realtime (validated against the GA voice list —
  unknown values fall back with a warning, never failing the call), Grok, Google Live, and
  Deepgram Voice Agent. ElevenLabs Agent voices are platform-managed; pipeline TTS voices
  come from the pipeline's TTS provider configuration.
- The resolved voice and its source are logged (`Session voice resolved`) and recorded in
  Call History per call. See [VOICE_SELECTION.md](VOICE_SELECTION.md).

## Transports

- audio_transport: `audiosocket` | `externalmedia`
  - **audiosocket**: TCP-based audio transport.
  - **externalmedia**: RTP/UDP-based audio transport.
  - **Selection**: Use the validated combinations in **[Transport & Playback Mode Compatibility Guide](Transport-Mode-Compatibility.md)**. Transport selection depends on provider mode and playback method (not a single strict rule).
- downstream_mode: `stream` | `file`
  - **stream**: Real-time streaming (20ms frames). Best UX. Works with full agents.
  - **file**: File-based playback via bridge. Most robust for pipelines; streaming-first is supported with automatic fallback to file when enabled.

## AudioSocket

- audiosocket.host: Bind address for AudioSocket listener.
- audiosocket.advertise_host: Address Asterisk connects to (optional; defaults to `audiosocket.host`). Use for NAT/VPN.
- audiosocket.port: TCP port.
- audiosocket.format: legacy fallback only — the audio profile is the source of truth for the wire format (`profiles.<name>.transport_out`, resolved per call; companded profiles ride the signed-linear carrier automatically). This value is used only when a call resolves no valid profile wire format (e.g. configs predating Audio Profiles). It is intentionally not exposed in the Admin UI Transport page; keep it at `slin` if present, and select wideband on the Agent or with `AI_AUDIO_PROFILE` rather than changing it globally.

## ExternalMedia

- external_media.rtp_host: Bind address for RTP server.
- external_media.advertise_host: Address Asterisk sends RTP to (optional; defaults to `external_media.rtp_host`). Use for NAT/VPN.
- external_media.rtp_port: Port for inbound RTP.
- external_media.port_range: Optional range (`start:end`) for dynamic per-call RTP allocation; defaults to `rtp_port`.
- external_media.codec: Asterisk RTP wire codec, BOTH directions, one codec shared by every profile and Agent (audio-profile Transport Output/Input apply to AudioSocket only; engine restart applies changes). Supported: `ulaw`, `alaw`, `slin16`. The release baseline is `ulaw`/`alaw` at 8 kHz; `slin16`/16 kHz RTP is outside the supported baseline — use AudioSocket with `wideband_pcm_16k` instead.
- external_media.sample_rate: engine-side transit rate — inbound RTP is decoded to PCM16 and resampled to this rate for the engine (default 16000, inferred from `external_media.format` when unset). Set it to the provider's native rate (8000 Hz for G.711 providers) to avoid an extra resample round trip; editable on the Audio Transport page (the page also keeps `external_media.format` coherent: `slin` at 8 kHz, `slin16` at 16 kHz). The Audio Path view shows the effective transit rate on the AI Engine node.
- external_media.direction: expert YAML key, default `both` (two-way media — the only mode a voice agent works in). `sendonly`/`recvonly` are one-way monitoring modes that leave the agent mute or deaf; not exposed in the Admin UI.
- external_media.lock_remote_endpoint: When true (default), **do not** accept mid-call changes to the inbound RTP source `(ip,port)` for that call.
- external_media.allowed_remote_hosts: Optional list of **IP addresses** allowed as inbound RTP sources. When set, packets from other sources are dropped (recommended when the RTP source IP is stable).
  - Note: if `asterisk.host` is an IP literal, the engine may default `allowed_remote_hosts` to `[asterisk.host]` unless explicitly configured.
  - If `asterisk.host` is a **hostname**, set `external_media.allowed_remote_hosts` explicitly (the platform does not auto-allowlist hostnames).
- Note: `external_media.jitter_buffer_ms` is no longer used (RTP buffering is not configurable here). Use `streaming.jitter_buffer_ms` for downstream playback pacing.

## Barge‑In

Controls interruption of TTS playback when the caller speaks.

- barge_in.enabled: true/false
- barge_in.initial_protection_ms: 200–600 ms. Drop inbound immediately after TTS starts to avoid self‑echo.
- barge_in.min_ms: 250–600 ms. Minimum sustained speech before a barge‑in is acknowledged (de‑bounce).
- barge_in.energy_threshold: 1000–3000. RMS energy threshold; raise on noisy lines.
- barge_in.cooldown_ms: 500–1500 ms. Ignore new barge‑ins after one triggers.
- barge_in.post_tts_end_protection_ms: 250–500 ms. Short guard to avoid clipping the start of the next caller utterance.
- barge_in.pipeline_min_ms: 80–250 ms. Pipeline-only (local file playback) minimum talk duration before triggering barge-in.
- barge_in.pipeline_energy_threshold: 200–1200. Pipeline-only RMS threshold (more sensitive than full-agent mode).
- barge_in.pipeline_talk_detect_enabled: true/false. Pipeline-only; uses Asterisk `TALK_DETECT` (ARI `ChannelTalkingStarted`) to trigger barge-in during channel playback.
- barge_in.pipeline_talk_detect_silence_ms: 800–2000. Pipeline-only; `TALK_DETECT(set)` silence window.
- barge_in.pipeline_talk_detect_talking_threshold: 1–32768 (default 256). Pipeline-only; global Asterisk `TALK_DETECT(set)` DSP magnitude threshold. Audio profiles can override it with `profiles.<name>.talk_detect_talking_threshold`; `wideband_pcm_16k` uses the live-validated value 1000 to reject wideband playback echo while retaining caller barge-in.

Notes (pipelines / `local_hybrid`):

- With `vad.silero_enabled` (and `vad.silero_barge_in`, the default), Silero VAD scores every caller frame in the engine, gated or not, and triggers pipeline barge-in on `vad.silero_start_ms` of sustained speech, using the same `talk_detect_initial_protection_ms`, `greeting_protection_ms` and `cooldown_ms` guards as talk detection. The local energy check is then skipped; `TALK_DETECT` may stay enabled alongside it.
- Pipelines play TTS locally (file playback), so the platform can flush playback on barge-in without colliding with provider-owned VAD/cancellation.
- With ExternalMedia, Asterisk channel playback may pause/alter the inbound RTP stream; `TALK_DETECT` is the preferred trigger source for pipeline barge-in.
- Prereqs: Asterisk must have talk detection available (`app_talkdetect.so` / `func_talkdetect.so`). Verify with `asterisk -rx 'module show like talkdetect'` and `asterisk -rx 'core show function TALK_DETECT'`.

Notes (OpenAI Realtime / AudioSocket):

- OpenAI can emit `input_audio_buffer.speech_started` when the provider has no cancellable response but the platform is still draining buffered audio; the platform treats this as a barge-in trigger to flush local output immediately.

Tuning guidance:

- Noisy lines: raise `energy_threshold` and `min_ms`.
- Fast, chatty interactions: lower `min_ms` and `post_tts_end_protection_ms` cautiously.

## Streaming (downstream_mode=stream)

Controls the pacing and robustness of streamed agent audio.

- streaming.sample_rate: Output sample rate (typically 8000 for telephony).
- streaming.jitter_buffer_ms: 80–150 ms. Higher = more robust to jitter, slightly higher latency.
- streaming.keepalive_interval_ms: TCP keepalive interval for streaming connections.
- streaming.connection_timeout_ms: Time to consider a streaming connection dead.
- streaming.fallback_timeout_ms: No audio for this long triggers fallback to file playback.
- streaming.chunk_size_ms: 20 ms recommended for telephony cadence.
- streaming.min_start_ms: 250–400 ms. Warm‑up buffer before first frame; too low risks underruns.
- streaming.low_watermark_ms: Brief pause/guard band; increase if underruns occur.
- streaming.provider_grace_ms: Absorb late provider chunks to avoid tail-chop artifacts.
- streaming.logging_level: Verbosity for the streaming manager.
- streaming.egress_force_mulaw: When true, converts outbound streaming audio to μ-law 8 kHz regardless of provider encoding.
- streaming.greeting_rtp_wait_ms: ExternalMedia-only. How long to wait (ms) for the remote RTP endpoint to be discovered during the initial greeting before falling back to file playback (prevents “dead air until caller speaks” in some Asterisk setups).

## VAD (Voice Activity Detection)

Defines how inbound speech is segmented into utterances for STT.

- vad.webrtc_aggressiveness: 0–3. 0=least aggressive (best for 8 kHz telephony), 3=most aggressive (may clip speech).
- vad.webrtc_start_frames: Consecutive frames above threshold to start recording.
- vad.webrtc_end_silence_frames: Silence frames to finalize an utterance (e.g., 50 → ~1000 ms at 20 ms frames).
- vad.min_utterance_duration_ms: Lower bound on utterance length. Raise if STT returns empty.
- vad.max_utterance_duration_ms: Hard cap to prevent runaway capture.
- vad.utterance_padding_ms: Padding around detected speech.
- vad.fallback_enabled: When true, sends audio at a fixed interval if VAD fails to detect speech.
- vad.fallback_interval_ms: Interval between fallback sends.
- vad.fallback_buffer_size: Bytes to accumulate at fallback thresholds.
- vad.upstream_squelch_enabled: When true, replaces low-energy/noise frames with silence for continuous-audio providers that have native VAD (improves end-of-turn detection in noisy environments; may suppress quiet callers if too aggressive).
- vad.upstream_squelch_base_rms: Minimum RMS threshold (PCM16 space) before audio is treated as “speech”.
- vad.upstream_squelch_noise_factor: Dynamic threshold multiplier relative to estimated noise floor.
- vad.upstream_squelch_noise_ema_alpha: EMA smoothing factor (0–1) for noise floor estimation.
- vad.upstream_squelch_min_speech_frames: Hysteresis: speech frames required to enter “speaking”.
- vad.upstream_squelch_end_silence_frames: Hysteresis: silence frames required to exit “speaking”.

Common pitfalls:

- Too-short utterances (e.g., 20 ms) cause empty STT transcripts → raise `min_utterance_duration_ms` and ensure `webrtc_end_silence_frames` is not too low.
- Overly aggressive VAD (aggressiveness=2/3) may clip 8 kHz speech; prefer 0–1 for telephony.

### Silero VAD (pipelines)

Asterisk `TALK_DETECT`, WebRTC VAD and the energy checks all decide "speech"
from signal energy, so breathing, line noise or a television count as the
caller while a quiet trailing syllable counts as silence. Silero VAD v6 is a
small neural network (about 2 MB, ONNX) that scores every 32 ms of audio with
a speech probability, natively at 8 or 16 kHz, in well under a millisecond per
chunk on one CPU core (about half a percent of a core per call). Enabled, it
runs in the engine on the very frames that reach the recognizer for every
modular-pipeline call and becomes the one detector behind barge-in, the
inactivity watchdog and the end of the caller's turn (see
[Pipeline End of Turn](#pipeline-end-of-turn-caller-turn-taking)). Full-agent
providers are not affected.

- `vad.silero_enabled`: `true`/`false` (default `false`). Requires
  `onnxruntime` in the engine image (in `requirements.txt`; rebuild the
  `ai_engine` image after upgrading) and the model file below. When either is
  missing the engine logs `Silero VAD unavailable` at start and pipelines fall
  back to talk detection or the result window exactly as before.
- `vad.silero_model_path`: path of `silero_vad.onnx` inside the engine
  container (default `models/vad/silero_vad.onnx`; `./models` is mounted at
  `/app/models`).
- `vad.silero_auto_download`: fetch the pinned release (v6.2.1, SHA-256
  verified) into that path on first start when the file is missing (default
  `true`). On hosts without outbound access run `scripts/fetch_silero_vad.sh`
  instead, or place the file from the `silero-vad` 6.2.1 wheel
  (`silero_vad/data/silero_vad.onnx`) there.
- `vad.silero_threshold`: speech probability at or above which a chunk counts
  as speech, 0–1 (default `0.5`). Raise it on noisy trunks, lower it for quiet
  callers.
- `vad.silero_stop_threshold`: probability below which speech ends. Defaults
  to `silero_threshold − 0.15`, Silero's own margin; chunks between the two
  thresholds neither end speech nor restart it.
- `vad.silero_start_ms`: sustained speech before the caller counts as talking
  (default `96`, three chunks). Holds the turn and triggers barge-in.
- `vad.silero_stop_ms`: silence after the caller's last speech before they
  count as quiet (default `300`). This is the pause a caller may take
  mid-sentence; the answer follows it by the pipeline's grace plus the
  recognizer's finalization, so 300 ms is snappy and 600–800 ms tolerates a
  caller who thinks aloud.
- `vad.silero_stt_finalize_ms`: silence fed to the recognizer the moment the
  caller is quiet (default `900`, `0` disables). A streaming recognizer only
  closes a phrase after its own silence gate (T-one: 600 ms, at 300 ms chunk
  boundaries); the burst satisfies it at once instead of in real time, so the
  result arrives within a recognizer round trip of the stop rather than 600–900
  ms later. Plain zeros through the normal audio path, so no recognizer change
  is needed; a recognizer that already emitted the phrase simply scores a
  little more silence.
- `vad.silero_barge_in`: let Silero speech during agent playback trigger
  barge-in (default `true`). Turn off to leave barge-in to `TALK_DETECT` while
  Silero still decides the end of turn.

Every call logs `Silero VAD tracking caller speech` at start; each end of
speech logs `Silero VAD: caller quiet` at debug level with whether a finalize
burst was sent and whether a result is still expected.

### Smart Turn (pipelines)

A voice-activity detector knows that the caller's sound stopped, not whether
their thought did: «я хочу…» followed by a pause reads exactly like «да».
[Smart Turn v3](https://github.com/pipecat-ai/smart-turn) (pipecat-ai,
BSD-2) is an audio-native turn detector, a Whisper Tiny encoder with a linear
head (8M parameters, 8 MB int8 ONNX) that scores the caller's own audio for
whether the turn is complete from prosody and content rather than from a
transcript, for 23 languages including Russian, in about 50 ms on one CPU
core. The engine runs it in-process, on the caller audio Silero VAD already
sees, whenever Silero reports the caller quiet; feature extraction is a numpy
re-implementation of the Whisper front end validated against
`transformers`, so nothing beyond `onnxruntime` is needed.

The verdict slots into the end-of-turn policy: `complete` (probability at or
above `vad.smart_turn_threshold`) changes nothing, the turn is released after
the grace as before; `incomplete` holds it up to
`vad.smart_turn_incomplete_hold_ms` past the stop, and a caller who goes on
is judged again on the whole turn at the next stop, as in the Pipecat
reference integration. A verdict still being computed holds the turn at most
`vad.smart_turn_timeout_ms`. The recognizer is finalized at the stop either
way, so the transcript is ready the moment the turn is released.

- `vad.smart_turn_enabled`: `true`/`false` (default `false`). Requires
  `vad.silero_enabled`; without Silero it stays off with an error at start.
- `vad.smart_turn_model_path`: path of the ONNX file inside the engine
  container (default `models/turn/smart-turn-v3.2-cpu.onnx`).
- `vad.smart_turn_auto_download`: fetch the pinned v3.2 CPU build (SHA-256
  verified) from the pipecat-ai Hugging Face release on first start when the
  file is missing (default `true`); `scripts/fetch_smart_turn.sh` does the
  same from the host.
- `vad.smart_turn_threshold`: completion probability that releases the turn
  at once (default `0.5`).
- `vad.smart_turn_incomplete_hold_ms`: how long an incomplete verdict may
  hold the turn past the Silero stop (default `3000`).
- `vad.smart_turn_trailing_silence_ms`: how much of the silence after the
  caller's last speech the model is shown (default `200`); the rest of the
  Silero stop window is trimmed, matching the reference integration's VAD.
- `vad.smart_turn_timeout_ms`: how long the turn waits for a verdict at most
  (default `500`).
- `vad.smart_turn_threads`: CPU threads for one inference (default `1`).

Every judged stop logs `Smart Turn verdict` with `complete`, `probability`,
`audio_ms` and `inference_ms`, and `Caller turn ended on silence` carries
`turn_verdict` (`complete`, `incomplete` when the hold ran out, `pending`
when the timeout did) and `turn_probability`. Telephony audio is 8 kHz
upsampled with a proper low-pass to the model's 16 kHz; the model was trained
on wideband speech, so judge its accuracy on your own recordings before
raising the hold, and lower `vad.silero_stop_ms` only once it earns trust.

## Caller inactivity (`no_input`)

The engine-level watchdog prevents an answered call from remaining open indefinitely when the caller stops responding. It is independent of provider endpointing/VAD: timing runs only while the conversation is ready and the caller, agent, and model are all idle.

- `no_input.enabled`: Enables the policy. Defaults to `true`.
- `no_input.inbound_enabled`: Applies the policy to inbound calls. Defaults to `true`.
- `no_input.outbound_enabled`: Compatibility/default field; outbound calls still require the equivalent per-Agent option under **Agents → Caller Inactivity Overrides**.
- `no_input.initial_timeout_sec`: Idle time before the first check-in. Defaults to 30 seconds.
- `no_input.grace_timeout_sec`: Reply window after each check-in. Defaults to 15 seconds.
- `no_input.max_check_ins`: Number of check-ins before the final message and hangup. Defaults to 1; `0` skips directly to the final message.
- `no_input.check_in_message`: Check-in text, spoken by the active provider/pipeline in the agent's configured voice.
- `no_input.final_message`: Non-empty text spoken before the ARI hangup. Blank or whitespace-only per-agent values inherit the safe default.
- `no_input.stall_timeout_sec`: Hang up once nothing has been exchanged for this long. Defaults to `0` (off); `90`–`120` is a sensible value. See *Stalled conversations* below.
- `no_input.max_call_duration_sec`: Hard cap on a call's length, counted from its start. Defaults to `0` (off). See *Maximum call duration* below.

Caller media activity, provider speech-start events, and user transcripts reset the window. The timer pauses during greetings, agent TTS, LLM processing, sustained caller speech, and other output gating. A terminal watchdog hangup is stored in Call History as `no_input_timeout`, not as a provider error.

### Stalled conversations (`no_input.stall_timeout_sec`)

The check-ins count only while the caller is quiet. A line that carries sound but no conversation therefore never triggers them: hold music, a noisy room, an IVR or a radio keep the speech detectors (Silero VAD, `TALK_DETECT`, provider VAD) reporting caller speech, the recognizer returns nothing the model could answer, and the call stays open until the trunk drops it. For outbound calls the check-ins are off by default anyway.

`stall_timeout_sec` is the safety net for that case. It counts from the last *exchange*, which is either a caller turn handed to the model (a transcript accepted for an LLM turn) or an utterance the agent finished (a reply, the greeting, a check-in), and it ignores the speech detectors entirely: caller sound on its own never restarts it. It pauses while the agent is speaking, so a long reply is never cut, and while the caller is on hold or in a transfer; a hosted agent's reply to its own silence pseudo-turn does not count as an exchange. When it expires the engine hangs up without an announcement and records the call as `no_input_timeout`, with `timed_out_reason: stall` in the session's `no_input_state` and the log line `Conversation stalled; hanging up` (`since_exchange_sec`, `last_exchange_source`, `caller_sound_active`), followed by `Hanging up stalled conversation`.

It applies to inbound and outbound calls alike: only `no_input.enabled` turns it off, not `inbound_enabled`/`outbound_enabled`, so an outbound campaign can have it without the check-ins. Per-agent overrides set their own value (`0` disables it for that agent). In the Admin UI it is **Stall Timeout** under **Advanced Settings → Voice Activity Detection → Caller Inactivity** and under **Agents → Edit Agent → Caller Inactivity Overrides**. A value below the check-in flow (`initial_timeout_sec` plus `grace_timeout_sec` per check-in) ends a quiet caller's call before the final message; keep it above that flow for inbound calls, for example `90` with the default `30` + `15`.

### Maximum call duration (`no_input.max_call_duration_sec`)

`max_call_duration_sec` is a hard cap on a call's length. It counts from the call's start (for an outbound call, the time already spent in the AMD hop before the agent's session began is included), and nothing pauses it: not the caller talking, not the agent speaking, not a turn being processed. When it is reached the engine hangs up at once, mid-sentence if need be, without an announcement; a transfer or hold in progress only delays the hangup, which is retried every few seconds until the engine can end the call. The log line is `Call reached its maximum duration; hanging up` (`duration_sec`, `agent_speaking`, `caller_sound_active`), followed by `Hanging up call at its maximum duration`.

A call ended this way is recorded as `max_duration` (Call History shows *Max duration reached*; the outbound attempt and the post-call webhook's `{call_outcome}` carry `max_duration`), as a policy outcome and not a provider failure. Like the stall timeout it applies to inbound and outbound calls alike, only `no_input.enabled` turns it off, and per-agent overrides set their own value (`0` disables it for that agent). In the Admin UI it is **Max Call Duration** next to the stall timeout, globally and per agent.

Provider generation completion is not treated as caller playback completion. Before terminal hangup, the engine also drains provider/coalescing queues, jitter frames, frame remainder, active ARI playback, and a short transport-specific post-roll. This applies uniformly to AudioSocket and ExternalMedia/RTP; pipeline/file announcements use Asterisk `PlaybackFinished` as their authoritative boundary.

Per-agent overrides are partial and inherit every omitted global field. In the Admin UI, configure global defaults under **Advanced Settings → Voice Activity Detection → Caller Inactivity** and agent-specific values under **Agents → Edit Agent → Caller Inactivity Overrides**.

For ElevenLabs full-agent deployments, also follow the provider-side client-event and silence ownership settings in [Provider-ElevenLabs-Setup.md](Provider-ElevenLabs-Setup.md#ava-caller-inactivity-compatibility-v731).

## LLM block

- llm.initial_greeting: First message spoken by the agent (if provider supports explicit greeting or engine plays via TTS).
- llm.prompt: Persona/system instruction used by LLMs.
- llm.api_key: Optional API key for LLMs that require it.

## Providers

### OpenAI Realtime (monolithic agent)

- providers.openai_realtime.api_key: injected from `OPENAI_API_KEY` (env-only; do not commit secrets to YAML).
- providers.openai_realtime.api_version: `ga` (default) or `beta` (legacy payload/header behavior).
- providers.openai_realtime.model, voice, base_url: Model and voice.
- providers.openai_realtime.instructions: Persona override. Leave empty to inherit `llm.prompt`.
- providers.openai_realtime.greeting: Explicit greeting. Leave empty to inherit `llm.initial_greeting`.
- providers.openai_realtime.response_modalities: list of modalities, typically `[\"audio\"]` or `[\"audio\", \"text\"]`.
- providers.openai_realtime.provider_input_encoding/provider_input_sample_rate_hz: Format sent to OpenAI (typically PCM16); prefer matching this to the engine’s internal PCM rate to avoid extra resampling.
- providers.openai_realtime.input_encoding/input_sample_rate_hz: Inbound format; use `ulaw` at 8 kHz when AudioSocket() is invoked with `,ulaw` (engine converts to PCM before sending to OpenAI).
- providers.openai_realtime.output_encoding/output_sample_rate_hz: Provider output; for telephony, prefer `mulaw` at 8 kHz (for example: `output_encoding: mulaw`, `output_sample_rate_hz: 8000`) to avoid mid-stream PCM → μ-law conversion artifacts.
- providers.openai_realtime.target_encoding/target_sample_rate_hz: Downstream transport expectations (e.g., μ‑law at 8 kHz).
- providers.openai_realtime.egress_pacer_enabled: When true, OpenAI provider emits fixed 20 ms audio cadence (silence on underrun); prefer `false` when downstream playback already paces reliably.
- providers.openai_realtime.turn_detection: Server‑side VAD (type, silence_duration_ms, threshold, prefix_padding_ms); improves turn handling.
  - Metrics: `ai_agent_openai_assumed_output_sample_rate_hz`, `ai_agent_openai_provider_output_sample_rate_hz`, and `ai_agent_openai_measured_output_sample_rate_hz` are **low-cardinality gauges** (latest observed across calls). Use Call History for per-call debugging.

### xAI Grok Voice Agent (monolithic agent, NEW in v6.5.2)

- providers.grok.api_key: injected from `XAI_API_KEY` (env-only legacy fallback for single-instance setups). Multi-instance deployments should use per-instance `api_key_file: /app/project/secrets/providers/<provider_key>/api-key` instead — never commit secrets to YAML.
- providers.grok.model: `grok-voice-latest` (default; xAI's only published Voice Agent model as of v6.5.2).
- providers.grok.voice: One of `eve`, `ara`, `rex`, `sal`, `leo`, or a custom cloned-voice ID from your xAI workspace.
- providers.grok.instructions / greeting: Persona override + explicit greeting. Leave empty to inherit `llm.prompt` / `llm.initial_greeting`.
- providers.grok.input_encoding / input_sample_rate_hz: AudioSocket inbound format. Default `ulaw` @ 8 kHz (matches Asterisk telephony format with no resampling).
- providers.grok.provider_input_encoding / provider_input_sample_rate_hz: Format actually sent to xAI. Default `ulaw` @ 8 kHz (xAI accepts `audio/pcmu` natively). Set to `linear16` @ 24 kHz for wideband AudioSocket (`slin16`) setups.
- providers.grok.output_encoding / output_sample_rate_hz: Provider output observed from xAI. Default `linear16` @ 24 kHz; AAVA converts it to the configured transport target.
- providers.grok.target_encoding / target_sample_rate_hz: Caller-facing Asterisk transport target. Default `ulaw` @ 8 kHz.
- providers.grok.turn_detection: Server-side VAD (default `server_vad` with `threshold: 0.5`, `silence_duration_ms: 600`, `prefix_padding_ms: 300`). Named Grok instances inherit the canonical `grok` runtime block before applying instance overrides.
- providers.grok.session_warn_after_seconds: Conservative long-session warning threshold (default `1680` = 28 minutes; set to `0` to disable). xAI's current model page lists a 120-minute maximum; the earlier warning remains for compatibility and is not a statement of the current provider limit.
- providers.grok.extra_tools: YAML escape hatch for xAI-native tools (`web_search`, `x_search`, `file_search`, `mcp`) not exposed in the Admin UI. Forwarded verbatim into `session.update.tools`. See [docs/Provider-Grok-Setup.md](Provider-Grok-Setup.md).
- providers.grok.display_name / customer: Free-text labels surfaced in the Admin UI and call-history attribution. Used to identify multi-instance deployments.

xAI does not consistently send `session.updated` ACK; the provider waits ~2s and proceeds either way. See [docs/Provider-Grok-Setup.md](Provider-Grok-Setup.md) for current audio, VAD, interruption, announcement, and long-session behavior, and [docs/Multi-Instance-Full-Agent-Providers.md](Multi-Instance-Full-Agent-Providers.md) for multi-instance routing.

### OpenAI (pipelines)

Modular OpenAI pipeline components use `type: openai` provider blocks:

- `openai_llm`: Chat Completions (`chat_base_url`, `chat_model`)
- `openai_stt`: Speech-to-Text via `audio/transcriptions` (`stt_base_url`, `stt_model`)
- `openai_tts`: Text-to-Speech via `audio/speech` (`tts_base_url`, `tts_model`, `voice`, `response_format`)

**Vendor fields (LLM).** A provider block describes one endpoint, so any key the engine has no meaning for is treated as a parameter for that endpoint and is forwarded verbatim in the Chat Completions body. Write the vendor's own parameter name directly in the block:

```yaml
providers:
  native_llm:
    type: openai
    chat_base_url: https://api.mistral.ai/v1
    chat_model: mistral-small-latest
    prompt_cache_key: prompt-1        # or ${CACHE_KEY:-prompt-1}
    safe_prompt: true
```

Three groups never leave the engine: fields of the typed provider config (`chat_base_url`, `chat_model`, `response_timeout_sec`, `voice`, …), routing and identity metadata (`type`, `name`, `enabled`, `capabilities`, `base_url`, `model`, `timeout_sec`, …), and anything shaped like a credential (a key containing `api_key`, `secret`, `password` or `credential`, or named `token` / ending in `_token`). `max_tokens` is a request parameter, not a credential, and is forwarded.

The flip side is that a typo now reaches the endpoint, and an API that rejects unknown parameters will fail the request. Every call that carries forwarded fields logs their names (never their values) as `Forwarding extra_body fields to the LLM`, so check that line first when an endpoint starts returning 400.

Keys the engine owns in the request itself (`model`, `messages`, `stream`, `tools`, `tool_choice`) cannot be overridden and are ignored with a warning. An explicit `extra_body:` mapping is still supported, in the provider block, in a pipeline's `options.llm`, or in runtime options, shallow-merged in that order; it wins over a bare key of the same name and is the only way to set a field per pipeline rather than per provider.

For prompt caching (`prompt_cache_key` on OpenAI and Mistral, where cached input tokens bill at a fraction of the normal rate), pick a value that stays the same across calls sharing a system prompt and bump it when that prompt changes. A per-call value defeats the cache, and the key must not contain secrets or personal data.

**Transport (LLM).** One Chat Completions request is made per caller turn, and the pause between turns (the reply's playback plus the caller's next words) usually outlasts aiohttp's 15 s idle window, so by default almost every reply pays a fresh TCP+TLS handshake. Two typed keys on the provider block, both also accepted per pipeline under `options.llm` where they override the block, change that; neither is ever forwarded to the endpoint as a body field:

```yaml
providers:
  native_llm:
    type: openai
    chat_base_url: https://api.mistral.ai/v1
    keepalive_timeout_sec: 240        # idle window for connection reuse; absent = aiohttp's 15 s
    proxy: "http://xray:8080"         # optional; empty or absent = direct connection
```

- `keepalive_timeout_sec`: how long an idle connection to the endpoint is kept for reuse. It stands on its own and does not need a proxy; a window of 120-300 s outlasts a turn, so the next request reuses the connection. `0` or an unparsable value falls back to the default.
- `proxy`: an HTTP proxy URL for this adapter's Chat Completions requests alone, with the same rules as the ElevenLabs adapter (`http://` and `https://` only, inline credentials moved into a `Proxy-Authorization` header and never logged, a malformed value fails the adapter rather than connecting directly). The container's `HTTPS_PROXY` is ignored either way. The OpenAI STT and TTS adapters keep their direct path.

After `data: [DONE]` the adapter now reads the streamed body to its end before releasing the connection: aiohttp closes a connection whose body was left unread, and the chunk terminator can land a packet after the `[DONE]` line, which silently cost the next turn a handshake.

Whether a request reused a connection is visible in the log: `OpenAI chat completion received` and the new `OpenAI streaming completed` line (with `finish_reason`, `chars`, `first_token_ms`, `total_ms`) carry `connection=reused` or `connection=new` with `connect_ms` (TCP, TLS and, through a proxy, the CONNECT round trip) and `headers_ms` (until the response headers). The ElevenLabs TTS `synthesis completed` lines carry the same fields. `OpenAI-compatible LLM transport ready` at the first request of a session reports the proxy and keepalive in force.

**Prompt warm-up (LLM).** Each call gets its own adapter and connection pool, so the first Chat Completions request of every call pays a new TCP+TLS connection (through a proxy, the CONNECT round trip on top) and the prefill of the whole system prompt, and both land on the caller's first reply. `warm_up: true` on the provider block (or per pipeline under `options.llm`, where it overrides the block; never forwarded to the endpoint) sends one `max_tokens: 1` request the moment the prompt, the tools and the greeting are resolved, while the greeting is synthesized and played. It carries exactly the prefix the first turn will carry: the system prompt with its variables substituted, the tool schemas, the greeting as the assistant's first message, and any vendor field such as `prompt_cache_key`, followed by a one-character user message, through the adapter's own session. The reply is discarded and never enters the history. By the first real turn the connection is open in the pool (`connection=reused` on `OpenAI streaming completed`) and, on an endpoint with prefix caching (vLLM with prefix caching on, OpenAI, Mistral), the prompt prefix is cached, so that turn prefills only the caller's words.

```yaml
providers:
  native_llm:
    type: openai
    chat_base_url: https://ai-api.example.com/v1
    keepalive_timeout_sec: 150
    warm_up: true                     # default false
```

The log shows `Pipeline LLM warm-up started` (debug) and `LLM prompt warm-up completed` with `total_ms`, `messages_count`, `tools_count`, the connection fields, and `prompt_tokens` / `cached_tokens` where the endpoint reports them (`cached_tokens` on the warm-up itself says whether the previous call's prefix was still cached). A failed warm-up logs `LLM prompt warm-up failed` and costs the call nothing else; hangup cancels one still in flight; a caller who interrupts the greeting before it completes simply gets a second connection. Off by default, because a metered API bills one extra prompt per call (the warm-up at the full input price, the first turn at the cached price). The prefix cache only helps where the chat template puts the system prompt at the start of the token stream; a template that attaches it to the last user message (the older Mistral ones do) still gets the handshake out of the way, but not the prefill.

Requirements:

- `OPENAI_API_KEY` must be set in the environment.

### Telnyx AI Inference (pipelines)

Telnyx AI Inference is supported as a modular LLM component:

- `telnyx_llm`: OpenAI-compatible Chat Completions (`chat_base_url`, `chat_model`, `temperature`, `max_tokens`, `response_timeout_sec`, `api_key_ref`)

Requirements:

- `TELNYX_API_KEY` must be set in the environment.

Notes:

- Telnyx supports many model IDs. Use the exact model ID returned by Telnyx `/models`.
- Some model IDs represent **external providers** (for example `openai/gpt-4o`). Those require `providers.telnyx_llm.api_key_ref` to be set (Integration Secret identifier) or Telnyx will return `400` with "OpenAI API key required…".
- For pipeline selection, set `AI_PROVIDER=telnyx_hybrid` (pipeline name) in your dialplan when forcing a per-extension pipeline.

### Deepgram Voice Agent

- providers.deepgram.api_key: injected from `DEEPGRAM_API_KEY` (env-only; do not commit secrets to YAML).
- The default Think stage uses Deepgram-managed reasoning and does not consume or require `OPENAI_API_KEY`. Custom/BYO reasoning endpoints are not currently exposed by this provider configuration.
- `providers.deepgram.model`: Deepgram Voice Agent listen model. The verified UI surface is `nova-3`, `flux-general-en`, and `flux-general-multi`; `nova-2-phonecall` is a legacy English telephony compatibility option. Existing custom values are preserved.
- `providers.deepgram.tts_model`: Aura voice model. Its language suffix must match `agent_language`; unknown or mismatched voices fail the Deepgram call before Settings are sent rather than silently falling back.
- `providers.deepgram.agent_language`: Full Voice Agent conversation language (default: `en`). Supported end-to-end languages are `en`, `es`, `de`, `fr`, `it`, `nl`, and `ja`; BCP-47 variants are normalized to their base language. This is separate from modular pipeline `stt_language`.
- providers.deepgram.greeting: Agent greeting. Leave empty to inherit `llm.initial_greeting`.
- providers.deepgram.instructions: Persona override for the “think” stage; leave empty to inherit `llm.prompt`.
- providers.deepgram.input_encoding/input_sample_rate_hz: Keep `input_encoding=ulaw` at 8 kHz when AudioSocket runs μ-law transport.
- providers.deepgram.continuous_input: true to stream audio continuously.
  - Metrics: `ai_agent_deepgram_input_sample_rate_hz` and `ai_agent_deepgram_output_sample_rate_hz` are **low-cardinality gauges** (latest observed across calls). Use Call History for per-call debugging.

### Google (pipelines)

- google_llm.system_instruction/system_prompt: Persona; if missing, adapter falls back to `llm.prompt`.
- google_tts/tts fields: voice, language, audio encoding/sample rate, target format.
- google_stt/stt fields: encoding, language, model, sampleRateHertz.

### Groq Speech (pipelines)

Groq Speech uses OpenAI-compatible REST endpoints:

- STT: `https://api.groq.com/openai/v1/audio/transcriptions`
- TTS: `https://api.groq.com/openai/v1/audio/speech` (Orpheus, WAV-only)

Requirements:

- `GROQ_API_KEY` must be set in the environment.

Config notes:

- `groq_stt` options: `stt_model` (`whisper-large-v3-turbo`, `whisper-large-v3`), plus optional `language`, `prompt`, `response_format` (`json|verbose_json|text`), `temperature`, `timestamp_granularities`.
- `groq_tts` options: `tts_model` (`canopylabs/orpheus-v1-english`, `canopylabs/orpheus-arabic-saudi`), `voice` (Orpheus voice IDs), `response_format` (`wav` only), and output format controls (`target_encoding`/`target_sample_rate_hz` or pipeline `tts.format`).

### Local provider (pipelines)

- Local STT/LLM/TTS parameters live under pipeline `options`. The engine plays `llm.initial_greeting` first if configured.

### Google Live (monolithic agent)

- `providers.google_live.api_key`: injected from `GOOGLE_API_KEY` (env-only; do not commit secrets to YAML).
- `providers.google_live.llm_model`: Live LLM model name (see `config/ai-agent.yaml` for shipped defaults).
- `providers.google_live.tts_voice_name`: Live voice name (provider-specific).
- `providers.google_live.response_modalities`: `audio`, `text`, or `audio_text` (provider behavior varies by model generation).
- `providers.google_live.hangup_fallback_audio_idle_sec`: idle-audio timeout after hangup is armed.
- `providers.google_live.hangup_fallback_min_armed_sec`: minimum armed duration before fallback can fire.
- `providers.google_live.hangup_fallback_no_audio_timeout_sec`: timeout when provider emits no farewell audio.
- `providers.google_live.hangup_fallback_turn_complete_timeout_sec`: grace period waiting for `turnComplete` before fallback hangup.
- `providers.google_live.hangup_markers_enabled`: enable/disable marker-based hangup heuristics (end_call / assistant_farewell) used to arm `cleanup_after_tts`. Recommended `false` for production (prefer tool-driven hangup via `hangup_call`).
- `providers.google_live.ws_keepalive_enabled`: enable protocol-level WebSocket ping keepalive (pings only fire when the connection is idle).
- `providers.google_live.ws_keepalive_interval_sec`: ping interval when keepalive is enabled.
- `providers.google_live.ws_keepalive_idle_sec`: minimum idle time (no `realtimeInput`) before sending a ping.

### Deepgram Voice Agent (monolithic agent)

- `providers.deepgram.voice_agent_base_url`: WebSocket endpoint for Deepgram Voice Agent.
  - Default: `wss://agent.deepgram.com/v1/agent/converse`
  - YAML overrides allow regional endpoints or proxy URLs.

### ElevenLabs Agent (monolithic agent)

Full agent provider using ElevenLabs Conversational AI for premium voice quality.

> **Scope Note**: ElevenLabs is supported as a full agent (`elevenlabs_agent`) and as a TTS-only pipeline adapter (`elevenlabs_tts`).

- `providers.elevenlabs_agent.api_key`: injected from `ELEVENLABS_API_KEY` (env-only; do not commit secrets to YAML).
- `providers.elevenlabs_agent.agent_id`: injected from `ELEVENLABS_AGENT_ID` (env-only).
- `providers.elevenlabs_agent.voice_id`: Voice ID for TTS output (configured in agent dashboard).
- `providers.elevenlabs_agent.model_id`: Model ID (e.g., `eleven_flash_v2_5`).
- `providers.elevenlabs_agent.voice_settings`: Optional object with `stability`, `similarity_boost`, `style` (0.0-1.0).

**Tool Calling**: ElevenLabs tools must be defined in the ElevenLabs dashboard. The engine executes tool calls locally based on matching function names. See [ElevenLabs Implementation Guide](contributing/references/Provider-ElevenLabs-Implementation.md) for tool schema format.

**Audio Format**: ElevenLabs uses PCM16 at 16kHz. The engine automatically resamples from telephony μ-law 8kHz.

Example:
```yaml
providers:
  elevenlabs_agent:
    enabled: true
    voice_id: "pNInz6obpgDQGcFmaJgB"
    model_id: "eleven_flash_v2_5"
```

## Precedence summary

- Provider/pipeline explicit overrides (instructions/greeting) take priority.
- Otherwise providers/pipelines inherit `llm.prompt` / `llm.initial_greeting`.
- Env `AI_ROLE`/`GREETING` act as defaults when YAML does not specify values.

## Health & Contexts

- Health endpoint:
  - `health.host`: Bind address for `/live`, `/ready`, `/health`, and `/metrics` (default `127.0.0.1`).
  - `health.port`: Port for the health/metrics HTTP server (default `15000`).
  - Environment variables `HEALTH_BIND_HOST` / `HEALTH_BIND_PORT` override the YAML values when set.
- Contexts:
  - Inline: `contexts:` block in `config/ai-agent.yaml` defines named contexts (prompt, greeting, profile, provider, tools).
  - External: YAML files in `config/contexts/*.yaml` are also loaded.
    - Each file must define a `name` field; that becomes the context key.
    - `system_prompt` in external files is treated as `prompt` if `prompt` is not present.
    - If the same context `name` exists both inline and in an external file, the inline definition in `ai-agent.yaml` wins.

### Context Options

Each context supports the following fields:

- `prompt`: System prompt/persona instructions for the AI.
- `greeting`: Initial greeting spoken when call connects.
- `profile`: Audio profile name to use for this context.
- `provider`: Provider override for this context.
- `tools`: List of **in-call** tool names to enable for this context.
- `pre_call_tools`: List of pre-call tool names to run after answer, before the AI speaks (HTTP lookups/enrichment).
- `in_call_http_tools`: List of in-call HTTP tool names to allowlist for this context (defined under `in_call_tools:`).
- `post_call_tools`: List of post-call tool names to run after the call ends (webhooks/automation).
- `disable_global_pre_call_tools`: Disable specific global pre-call tools for this context.
- `disable_global_in_call_tools`: Disable specific global in-call tools for this context.
- `disable_global_post_call_tools`: Disable specific global post-call tools for this context.
- `connection_audio`: Optional caller-only Asterisk ARI media URI played after answer while the provider or pipeline initializes. `tone:ring` repeats until stopped; omit or leave empty to disable it.
- `background_music`: Music On Hold class name for ambient music during calls (see below).

### Connection Audio / Ringback

Connection audio removes silent wait time between answer and the initial greeting without injecting audio into the provider, STT, or VAD path. The engine targets the caller channel and stops playback immediately before the first provider or pipeline greeting audio. Cleanup and provider-start failures also stop it.

- Recommended value: `tone:ring` (continuous local ringback until the greeting is ready).
- Optional locale: `tone:ring;tonezone=fr` (replace `fr` with an installed Asterisk tone zone).
- Local prompt: `sound:custom/please-wait` (plays once; the sound must exist on Asterisk).
- Remote URLs and `file:` URIs are rejected. Leave the field empty to disable connection audio.

In the Admin UI, edit an agent and enable **Play ringback while connecting**. The value is stored in the agent's `extra_json`, exported as `connection_audio`, and works identically for full-agent providers and modular pipelines.

Configure these overrides under **Agents → Edit Agent → Caller Inactivity Overrides**.
Agent persona and policy fields live in `agents.db`; legacy `contexts:` YAML is accepted
only as one-time migration input and is not a live v7.4 configuration surface.

### HTTP Tools (Phase Tools)

HTTP tools are configured in YAML and/or via the Admin UI:

- **Pre-call HTTP lookups**: live under `tools:<name>` with `kind: generic_http_lookup` and `phase: pre_call`.
- **Post-call webhooks**: live under `tools:<name>` with `kind: generic_webhook` and `phase: post_call`.
- **In-call HTTP tools**: live under `in_call_tools:<name>` with `kind: in_call_http_lookup` (AI-invoked during conversation).

See `docs/TOOL_CALLING_GUIDE.md` for full examples and variable substitution details.

### Background Music

Play ambient music during AI conversations. Music is mixed into the call audio.

- Agent `background_music`: MOH class name stored in the Agent's advanced fields (e.g., `default`, `ambient`).
  - When set, a snoop channel with Music On Hold starts when the call begins.
  - Music continues until the call ends.
  - Leave empty/omit to disable background music.

**Setup Requirements**:

1. Place audio files in `/var/lib/asterisk/moh/<class-name>/`
2. For FreePBX: Configure via **Settings → Music On Hold**
3. Supported formats: WAV, ulaw, alaw, sln, mp3

**Best Practices**:

- Use low-volume (15-20%) ambient/instrumental music
- Music is heard by the AI (affects VAD); loud music reduces accuracy
- Test with real calls before production

Example:
```yaml
contexts:
  support:
    greeting: "Hello, how can I help?"
    prompt: "You are a helpful support agent."
    background_music: "ambient"  # MOH class name
```

## Environment Variable Resolution

Environment variable placeholders (`${VAR}`, `${VAR:-default}`) are expanded for the **entire YAML file** when `config/ai-agent.yaml` is loaded.

Example (supported):
```yaml
providers:
  local:
    base_url: ${LOCAL_WS_URL:-ws://127.0.0.1:8765}  # ✅ Resolved
```

Notes:
- Expansion happens **before YAML parsing**. Use `${VAR:-default}` to avoid empty-string surprises.
- Avoid putting secrets directly in YAML; prefer `.env` + `${VAR}` placeholders.

## Admin UI HTTP Tool Testing (Security)

The Admin UI includes an HTTP tool **Test** feature that makes real outbound HTTP requests.

By default, the Admin UI blocks test requests to localhost/private targets to reduce SSRF risk if the UI is exposed beyond a trusted network.

> **Security note (accepted limitation):** the private-target check resolves DNS separately from the outbound HTTP connection, so a residual DNS-rebinding TOCTOU window exists. This is acceptable because the endpoint is admin-only and authenticated — do not rely on it as a hard SSRF boundary on an untrusted network.

- `AAVA_HTTP_TOOL_TEST_ALLOW_PRIVATE=1`: allow private/localhost targets (trusted network only).
- `AAVA_HTTP_TOOL_TEST_ALLOW_HOSTS=host1,host2`: allow specific hostnames.
- `AAVA_HTTP_TOOL_TEST_FOLLOW_REDIRECTS=1`: allow redirects (default is disabled).

**v6.5.0+:** the guard reads `.env` before `os.environ`, so changes made through the Admin UI Environment page take effect on the next test request without recreating the `ai_engine` container. `.env` read failures fail closed (helpers fall back to the default rather than silently consulting `os.environ`).


## Tips

- For noisy trunks, start with:
  - `barge_in.energy_threshold=2200`, `barge_in.min_ms=450`, `vad.webrtc_aggressiveness=1`.
- For lowest latency, start with:
  - `streaming.min_start_ms=250`, `streaming.jitter_buffer_ms=80`, `barge_in.min_ms=300` (expect more sensitivity to jitter).

## MCP (Experimental)

MCP-backed tools (Model Context Protocol) can be exposed through the existing tool calling system.

- Design + configuration guide: `docs/MCP_INTEGRATION.md`
