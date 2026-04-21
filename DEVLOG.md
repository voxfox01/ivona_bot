# ivona_bot — Development Log

Running record of what was built, changed, and why. Newest entries at the top.

---

## Phase 3 — Conversational UX Improvements *(in progress)*

### 2026-04-21 — Power and thermal monitoring in --debug-stt mode
**Commit:** *(this commit)*

**Problem observed:** Jetson reporting OC3 over-current throttling events (42 events on
`/sys/class/hwmon/hwmon3/oc3_event_cnt`). OC3 is the VDD_IN system input rail — total board
draw exceeding the PSU's current limit under peak LLM load.

**Change:** Extended the `--debug-stt` output box with a live hardware line sampled from
`tegrastats` after each utterance (just before LLM inference begins — near peak draw):

```
│  Hardware : 8,012mW  CPU 51°C  GPU 52°C
```

- `_sample_tegrastats()` spawns tegrastats, reads one line, terminates it immediately
- Parses VDD_IN (total board power), cpu@ temperature, and gpu@ temperature
- Falls back to "n/a" silently if tegrastats is unavailable
- No sudo required — tegrastats is readable by normal users on JetPack

**Diagnosis from OC event counters:**
```
oc1_event_cnt: 0   (VDD_CPU — fine)
oc2_event_cnt: 0   (VDD_GPU — fine)
oc3_event_cnt: 42  (VDD_IN  — PSU current limit hit)
```
Root cause: generic 5V/4A (20W) barrel-jack PSU insufficient for simultaneous LLM GPU inference
+ CPU + NVMe + USB ReSpeaker + Bluetooth A2DP. A 5V/5A+ supply should eliminate OC3 events.

---

### Objective
Close the experience gap between ivona_bot and commercial conversational robots (e.g., Reachy Mini).
Reachy Mini achieves ~2–4s end-to-end latency using the OpenAI Realtime API (cloud). ivona_bot
currently takes ~12–15s and requires "hey jarvis" before every question. The goal is to reduce
perceived latency and make multi-turn conversation feel natural — all while staying fully offline.

### Improvements tracker

| # | Improvement | Status |
|---|---|---|
| 3.1 | Continuous VAD conversation mode — no wake word for follow-up questions | ✅ Complete |
| 3.2 | In-process STT via faster-whisper — eliminate per-call model reload | ✅ Complete |
| 3.3 | Persistent Piper TTS process — eliminate ~0.5s subprocess startup per utterance | ✅ Complete |
| 3.4 | GStreamer audio pipeline — replace parecord polling with low-latency C pipeline | 🔲 Planned |
| 3.5 | Evaluate smaller/faster LLM (Qwen2.5-3B or TensorRT-LLM) | 🔲 Planned |

---

### 2026-04-21 — VAD multi-turn sessions, persistent TTS, mic muting
**Commit:** `b1caa14`

#### 3.1 — Continuous VAD conversation mode ✅
**Implemented in:** `app/main.py`, `services/stt/transcriber.py`, `config/settings.yaml`, `requirements.txt`

Replaced the single-turn wake→STT→LLM→TTS→wake loop with a session model:
- `run_conversation_session()` opens one persistent `parecord` stream per session
- `webrtcvad` (Google's WebRTC VAD) detects each utterance within the session
- After Ivona responds, listening resumes immediately — no wake word required for follow-ups
- Session ends after 30s of silence (`session_timeout_seconds`); control returns to wake word mode
- Session start/end spoken phrases configurable in `settings.yaml`

**VAD tuning to eliminate false triggers:**
- Required 3 consecutive voiced frames (~90ms) before confirming speech onset
- Added RMS gate (reject if RMS < 0.008) to filter ambient noise bursts that pass VAD
- Pre-roll ring buffer (300ms) captures audio before detected onset so utterance start isn't clipped
- End-of-utterance requires 900ms of consecutive silence
- `webrtcvad-wheels` used instead of `webrtcvad` — avoids `pkg_resources` import error on Python 3.12+

**Obstacle — VAD false-triggering on ambient noise:**
First test captured 0.99s of silence (RMS=0.0084) as speech. Fixed by raising aggressiveness
from 2 → 3 and adding the 3-frame onset + RMS gate combination.

#### 3.2 — In-process STT via faster-whisper ✅
**Implemented in:** `services/stt/transcriber.py`, `config/settings.yaml`

Switched active backend from `whisper_cpp` (subprocess) to `faster_whisper` (in-process):
- `WhisperModel` loaded once at startup (1.2s), stays warm for the session lifetime
- Per-utterance inference: ~0.1s on CPU for the base model (vs ~2–3s subprocess startup overhead)
- `local_files_only=True` added to prevent a HuggingFace network call at startup — required for
  offline conference deployment (was silently making an HTTP GET on every run)
- `beam_size=1` (greedy decoding) for ~3× speed vs default beam_size=5 with negligible quality loss
- `vad_filter=True` in transcribe() provides a second-pass noise filter inside faster-whisper

#### 3.3 — Persistent Piper TTS process ✅
**Implemented in:** `services/tts/speaker.py`

Replaced the per-utterance Piper subprocess with a single persistent process:
- Piper launched once with `--json-input` flag; stays alive between calls
- Each synthesis request writes a JSON line to stdin: `{"text": "...", "output_file": "/tmp/x.wav"}`
- Piper writes the output path to stdout as a completion signal; `readline()` blocks until done
- WAV file read with soundfile and played via sounddevice; temp file cleaned up after each call
- Eliminates ~0.5s subprocess startup cost per utterance

**Obstacle — `--output-raw` approach broken:**
First implementation used `--output-raw` with `select.select([stdout], [], [], 0.05)` to read PCM.
The 50ms select timeout fired before Piper wrote any output, returning 0 bytes every time.
Confirmed `--output-raw` worked in shell (`echo "text" | piper --output_raw | wc -c` → 67700 bytes)
but the Python-side timing was unreliable. Switched to `--json-input` which provides a clean
synchronization signal (stdout line) instead of a timeout-based read.

#### Mic muting during TTS playback
**Implemented in:** `app/main.py`

**Problem observed:** After a long response, VAD immediately captured 3+ seconds of the bot's own
voice as a new utterance. Root cause: `parecord` runs continuously and buffers audio into the pipe
during TTS playback. When `record_with_vad()` is called next, it reads buffered echo instantly.

**Fix:** Mute the PulseAudio mic source at the OS level during all TTS output:
- `pactl set-source-mute <pulse_source> 1` before speaking
- `pactl set-source-mute <pulse_source> 0` after speaking (in `finally` block)
- `_drain_mic_pipe()` flushes any residual frames buffered at the moment of unmute
- Applied to both the streaming response (`speak_streaming_muted`) and session phrases (`speak_muted`)

**Session timeout fix:**
`last_speech_time` was reset when the *user* spoke, not when Ivona finished. A long response
(34s LLM + TTS) would trigger the 30s inactivity timeout immediately after speaking. Fixed by
resetting `last_speech_time` after `speak_streaming_muted()` returns.

#### 3.4 — GStreamer audio pipeline
**Why:** parecord + pipe is working well; GStreamer would reduce audio capture latency further
and remove the parecord subprocess dependency. Deferred — current latency is acceptable.

**Files to change:** `services/stt/transcriber.py`, `services/wake_word/detector.py`
**New dependency:** `PyGObject` (gi.repository.Gst — available on Jetson via apt)

#### 3.5 — Smaller/faster LLM
**Why:** LLM generation is the largest single latency contributor (~10–15s for longer answers).
Options:
- **Qwen2.5-3B Q4KM** (~2.2GB, ~4–5s generation) — straightforward model swap
- **More GPU layers** — with faster-whisper using less memory than whisper-cli, may allow raising
  from 20 → 28 layers
- **TensorRT-LLM** — NVIDIA's Jetson-optimized runtime, potentially 2–3x faster than llama.cpp;
  significant setup effort, Phase 4 candidate

---

## Phase 2 — Runtime Hardening *(complete)*

### 2026-04-20 — OpenAI code review: harden runtime and fix config accuracy
**Commit:** `b0fe167`
**Trigger:** External code review identified four reliability risks before conference deployment.

**Changes made:**

**Startup preflight check** (`app/main.py: preflight_check()`)
- Added validation of all required binaries and model files before loading any heavy models
- Clear error message and clean exit if anything is missing — avoids cryptic failures mid-demo
- Checks: whisper-cli binary, whisper model, Piper binary, Piper voice model, LLM GGUF file, parecord

**Per-turn exception recovery** (`app/main.py: main() while loop`)
- Wrapped every interaction turn in `try/except Exception`
- Bad STT result, TTS subprocess error, or audio glitch now logs and loops back to wake word
- `KeyboardInterrupt` re-raised so Ctrl-C still exits cleanly

**Audio config truthfulness** (`config/settings.yaml`, `services/wake_word/detector.py`, `services/stt/transcriber.py`)
- Added `pulse_source` key to config — the explicit PulseAudio source name for the ReSpeaker
- Both `parecord` calls now pass `--device=<pulse_source>` instead of relying on PulseAudio default
- Removed misleading `input_device_index` and `input_channel` fields that implied sounddevice routing
  but had no effect on the actual parecord-based audio path

**Removed runtime network dependency** (`services/wake_word/detector.py`)
- Removed `openwakeword.utils.download_models()` from `__init__`
- `hey_jarvis_v0.1.onnx` is bundled inside the openwakeword package at
  `.venv/lib/.../openwakeword/resources/models/` — no download ever needed at runtime
- This was a silent offline risk: on a fresh machine the app would attempt a network fetch on startup

**Requirements cleanup** (`requirements.txt`)
- Removed `faster-whisper` from default install (it is not the active backend)
- Removed unused `rich` and `requests` packages
- Added comments explaining optional backends and why llama-cpp-python is built from source

**README correction** (`README.md`)
- Updated architecture diagram: STT stage now correctly labeled CPU-only with an explanatory note
  about CUDA being disabled to prevent GPU memory conflict with the resident LLM

---

### 2026-04-20 — Comprehensive README
**Commit:** `7027dd8`

Added full `README.md` covering:
- ASCII architecture diagram with per-stage latency estimates
- Complete hardware table
- Step-by-step setup: venv, llama-cpp-python CUDA build, llama.cpp/whisper.cpp cmake builds,
  model downloads, PulseAudio config, Bluetooth A2DP profile switch
- Runtime commands including `--debug-stt` flag
- Configuration reference table
- Memory budget breakdown with n_gpu_layers warning
- Troubleshooting table covering all issues encountered during development

---

### 2026-04-20 — Audio pipeline overhaul + streaming LLM→TTS + STT debug mode
**Commit:** `e3da8a5`

**Audio capture: sounddevice/ALSA → parecord (PulseAudio)**
- `sounddevice` opened ALSA devices directly, bypassing the PulseAudio DSP layer
- ReSpeaker XVF3800 requires PulseAudio for proper initialization — direct ALSA gave constant
  background noise (RMS=0.015, identical across all recordings regardless of speech)
- Switched both STT recorder and wake word detector to `parecord` subprocess
- `parecord` routes through PulseAudio and properly activates the ReSpeaker XMOS chip
- Added `_keep_mic_active()` at startup: calls `pactl suspend-source <index> 0` to prevent
  PulseAudio from suspending the ReSpeaker between turns (suspension caused 1–2s silence on
  each new recording as the source woke up)

**Device index correction**
- ReSpeaker had moved from sounddevice index 24 to index 0 (hw:0,0) after a system restart
- Index 24 was a Jetson APE audio output device (0 input channels) — correct device is index 0

**STT model downgrade: ggml-medium (1.5GB) → ggml-base (142MB)**
- ggml-medium.en.bin was 1.5GB, not 176MB as initially recorded in CLAUDE.md
- With LLM holding ~2.7GB GPU/RAM, swap already at 2.5GB (VS Code SSH + GNOME), and Piper
  loading at TTS time, whisper-cli adding another 1.5GB triggered the Linux OOM killer (SIGKILL)
- Switched to ggml-base.en.bin (142MB) — fits comfortably alongside LLM with ~1.9GB headroom

**whisper-cli forced CPU-only**
- Added `CUDA_VISIBLE_DEVICES=""` to whisper-cli subprocess environment
- Without this, whisper-cli competed with the resident LLM for GPU memory → SIGABRT (exit -6)
- Whisper base on CPU: ~2–3s inference; acceptable given the LLM is the dominant latency

**Streaming LLM→TTS pipeline**
- Previously: wait for full LLM response (~9s) then speak it all
- Now: LLM generates in a background thread; TTS speaks each sentence as it arrives
- Implemented via `Responder.stream_sentences()` using llama-cpp-python `stream=True`
- Sentence boundary detection via regex `r'[.!?][\s"\')\]]*(?=\s|$)'`
- First spoken word heard ~2–3s after question vs ~9s previously

**Minimum recording period**
- Added `min_seconds=2.0` to `record_until_silence()` — silence detection cannot fire until
  2 full seconds have been recorded; gives PulseAudio source time to unsuspend if needed
- Previously, silence detection would trigger immediately on a just-woken source, returning
  near-empty audio that caused whisper-cli to SIGABRT

**openWakeWord model state flush**
- After wake word detection, feed ~1s of silence frames through the model to reset internal state
- Without this, the model retained activation from the wake word phrase and would immediately
  re-trigger on the next `wait_for_wake_word()` call (detected wake word every ~5 seconds in loop)

**`--debug-stt` flag** (`app/main.py`)
- Added `--debug-stt` argument to `app/main.py`
- When enabled, prints audio RMS, peak, duration, and raw transcript after each recording turn
- Used to diagnose the silent-mic and whisper-crash issues

**STT countdown in test_pipeline.py**
- Added 3-2-1 countdown before recording in `test_stt()`
- Prints audio diagnostics and a prominent transcript box so pipeline issues are immediately visible

---

## Phase 1 — Environment & Pipeline Setup *(complete)*

### 2026-04-18 — Fix openWakeWord: tflite → ONNX backend
**Commit:** `057987a`

**Problem:** `tflite_runtime` installed from PyPI was compiled against NumPy 1.x. The project
venv has NumPy 2.2.6 which changed the C ABI (`_ARRAY_API not found` at import time). The
runtime crashed immediately when initializing the wake word detector.

**Fix:** Changed `inference_framework` from `"tflite"` to `"onnx"` in `WakeWordDetector.__init__`.
Both `.tflite` and `.onnx` model variants are downloaded by openWakeWord — `onnxruntime` has no
NumPy 2.x ABI constraint.

---

### 2026-04-18 — Phase 1 & 2 initial implementation
**Commit:** `19d906c`

Full offline voice pipeline built from scratch for NVIDIA Jetson Orin Nano Super.

**Wake word** (`services/wake_word/detector.py`)
- openWakeWord 0.6.0, built-in `hey_jarvis` model, ONNX inference
- sounddevice InputStream (later replaced — see Phase 2 audio overhaul)
- 6-channel ReSpeaker capture, channel 4 (beamformed output)

**STT** (`services/stt/transcriber.py`)
- Dual-backend: `whisper_cpp` (default) or `faster_whisper`
- `whisper_cpp`: subprocess calling whisper-cli built with CUDA (sm_87)
- `faster_whisper`: WhisperModel in-process (CPU-only on aarch64 — no CUDA via PyPI)
- Record-until-silence with RMS threshold 0.01

**LLM** (`services/llm/responder.py`)
- Dual-backend: `llama_cpp_python` (default) or `llama_cli` subprocess
- `llama_cpp_python`: Qwen2.5-7B-Instruct Q4_K_M, 20 GPU layers, warm in memory
- `llama_cli`: subprocess fallback for debugging
- ChatML prompt format (`<|im_start|>` / `<|im_end|>`)
- Qwen2.5-7B chosen for quality; Q4_K_M quantization for 8GB memory budget

**TTS** (`services/tts/speaker.py`)
- Piper TTS aarch64 binary with `en_US-amy-medium` voice
- Subprocess: text → Piper → temp WAV file → sounddevice playback
- Temp WAV approach chosen over ALSA pipe to avoid ALSA underrun errors

**Main loop** (`app/main.py`)
- Sequential: wake → record → transcribe → generate → speak → repeat
- SIGINT/SIGTERM handlers for clean shutdown

**Test script** (`scripts/test_pipeline.py`)
- Per-service smoke tests with `--skip-wake/stt/llm/tts` flags

**Key obstacles resolved during Phase 1:**
- Qwen2.5 GGUF is split across two files — llama.cpp auto-loads split GGUFs by pointing to file 1
- LLM OOM with 33 GPU layers → reduced to 20 (~2.7GB); safe limit confirmed at 20
- Bluetooth speaker on HFP 8kHz → switched profile to A2DP: `pactl set-card-profile bluez_card.DD_03_78_D7_0D_1A a2dp_sink`
- ReSpeaker not visible in GNOME sound settings → accessible via pavucontrol (PulseAudio multichannel source)
- CTranslate2 PyPI wheel has no CUDA on aarch64 → built whisper.cpp from source as GPU STT alternative
- llama-cli `--simple-io` flag polluted subprocess stdout → removed flag, redirected stderr to DEVNULL
- Piper TTS ALSA underrun → synthesize to temp WAV first, then play via sounddevice

---

### 2026-04-14 — Initial project setup
**Commit:** `f748e4e`

- Jetson Orin Nano Super provisioned with JetPack 6.1 (L4T 36.5.0), CUDA 12.6 verified
- Python 3.10 venv created at `.venv/`
- Project directory structure established
- Qwen2.5-7B-Instruct Q4_K_M GGUF downloaded (split: `...-00001-of-00002` + `...-00002-of-00002`)
- llama.cpp cloned and built: `cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87`
- whisper.cpp cloned and built with same CUDA flags → `whisper-cli` binary
- llama-cpp-python 0.3.20 built from source with CUDA support (~45 min compile)
- faster-whisper 1.2.1 installed (CPU-only — PyPI aarch64 wheel has no CUDA)
- openWakeWord 0.6.0 installed; built-in models downloaded
- Piper TTS binary (aarch64) + `en_US-amy-medium` voice installed via `scripts/install_piper.sh`
- ReSpeaker XVF3800 detected via PulseAudio (not GNOME sound panel — use pavucontrol)
- Bluetooth BM4D speaker configured to A2DP profile

---

*Log format: newest at top. Each entry covers what changed, why, and any constraints that shaped the decision.*
