# ivona_bot — Project Context for Claude Code

## Who is the user
David Rivers — data scientist, comfortable with Python and VS Code, new to robotics and embedded Linux. Works on a Windows laptop connected to the Jetson via VS Code SSH extension. Needs guidance on hardware integration, Linux system services, and audio pipelines.

---

## Project objective
Deploy a fully **offline** LLM-powered conversational robot on an NVIDIA Jetson Orin Nano Super to answer visitor questions at the **AI Conference for Good** (deadline: **July 2026**). The robot must operate with zero internet connectivity on the conference floor.

---

## Hardware
| Component | Detail |
|---|---|
| Compute | Jetson Orin Nano Super — 8GB unified RAM/VRAM, Ampere GPU (sm_87), CUDA 12.6, JetPack 6.1 (L4T 36.5.0) |
| Storage | 1TB NVMe SSD on M.2 2280 (OS installed here) |
| Microphone | ReSpeaker XMOS XVF3800 4-Mic Array — sounddevice index 24, channel 4 (beamformed output), 16kHz |
| Speaker | Bluetooth BM4D — A2DP profile (44.1kHz stereo), auto-connects on boot |
| Dev machine | Windows laptop → VS Code SSH → Jetson |

---

## Pipeline architecture
```
openWakeWord (CPU)  →  whisper.cpp (GPU)  →  llama-cpp-python (GPU)  →  Piper TTS (CPU)
"hey jarvis"            ggml-medium.en         Qwen2.5-7B Q4KM           en_US-amy-medium
ReSpeaker ch4          ~0.5s latency          ~10s / 50 tokens           ~2s / sentence
```

---

## Key file locations
| Path | Purpose |
|---|---|
| `app/main.py` | Main wake→STT→LLM→TTS loop |
| `config/settings.yaml` | All runtime config (devices, models, backends, system prompt) |
| `services/wake_word/detector.py` | openWakeWord listener |
| `services/stt/transcriber.py` | Dual-backend STT (faster_whisper or whisper_cpp) |
| `services/llm/responder.py` | Dual-backend LLM (llama_cli subprocess or llama_cpp_python) |
| `services/tts/speaker.py` | Piper TTS → sounddevice playback |
| `scripts/test_pipeline.py` | Per-service smoke tests (`--skip-wake/stt/llm/tts`) |
| `models/LLM/qwen_2_5/` | Qwen2.5-7B Q4KM split GGUF (2 files, ~4.4GB total) |
| `models/STT/ggml-medium.en.bin` | Whisper medium GGML model (176MB) |
| `models/TTS/en_US-amy-medium.onnx` | Piper voice model (61MB, not in git) |
| `llama.cpp/build/bin/` | Built llama-cli + libggml-cuda.so |
| `whisper.cpp/build/bin/whisper-cli` | Built whisper-cli binary |
| `services/piper/piper/` | Piper binary + espeak-ng data (not in git) |

---

## What has been completed

### Phase 1 — Environment & component setup ✅
- JetPack 6.1 / CUDA 12.6 verified
- Qwen2.5-7B-Instruct Q4_K_M downloaded (split: `...-00001-of-00002.gguf` + `...-00002-of-00002.gguf`)
- faster-whisper 1.2.1 installed (CPU only — PyPI aarch64 wheel has no CUDA)
- openWakeWord 0.6.0 installed; built-in models downloaded (`hey_jarvis`, `alexa`, etc.)
- llama.cpp cloned and built with `GGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87`
- whisper.cpp cloned and built with same CUDA flags → `whisper-cli` binary
- llama-cpp-python 0.3.20 built from source with CUDA support (~45 min compile)
- Piper TTS binary (aarch64) + `en_US-amy-medium` voice downloaded
- Bluetooth speaker switched from HFP (8kHz mono) to A2DP (44.1kHz stereo)
- ReSpeaker XVF3800 detected via PulseAudio (not GNOME sound panel — use pavucontrol)

### Phase 2 — Service modules & integration ✅
- All four service modules written and importable
- `test_pipeline.py` validates each service independently
- TTS → BT speaker: working
- STT (whisper.cpp GPU): working
- LLM (llama-cpp-python, 20 GPU layers): working
- Repo committed and pushed to GitHub

---

## Known constraints & decisions

**Memory budget (8GB unified):**
- OS + idle processes: ~2.5GB
- LLM (20 GPU layers, Q4KM): ~2.7GB
- STT (whisper medium): ~1.5GB
- Headroom: ~1.3GB
- `n_gpu_layers: 20` is the safe limit. Do not increase above 28.

**Wake word:** Using `hey_jarvis` (built-in openWakeWord model) as placeholder. Custom `hey_ivona` model needs training — Phase 3 task.

**STT backend:** `whisper_cpp` (GPU) is active. `faster_whisper` fallback available in config but CTranslate2 PyPI wheel has no CUDA for aarch64.

**LLM backend:** `llama_cpp_python` is active (model stays warm in memory). `llama_cli` subprocess fallback available — use when Python bindings unavailable or for debugging.

---

## Obstacles encountered and solutions

| # | Problem | Solution |
|---|---|---|
| 1 | `huggingface-cli` deprecated | Use `hf` CLI or `huggingface_hub.hf_hub_download()` in Python |
| 2 | Qwen2.5 GGUF is split (2 files, not 1) | Download both `...-00001-of-00002` and `...-00002-of-00002`; llama.cpp auto-loads split files |
| 3 | Bluetooth speaker on HFP (8kHz mono) | `pactl set-card-profile bluez_card.DD_03_78_D7_0D_1A a2dp_sink` |
| 4 | ReSpeaker not in GNOME sound settings | Use `pavucontrol`; device appears as PulseAudio source 14 (multichannel 6ch 16kHz) |
| 5 | CTranslate2 PyPI wheel has no CUDA on aarch64 | Built whisper.cpp from source with CUDA instead; used as drop-in STT backend |
| 6 | LLM OOM with 33 GPU layers (only 2GB free during builds) | Reduced to `n_gpu_layers: 20` (~2.7GB); free memory after builds complete |
| 7 | `llama-cli` status messages polluting subprocess stdout | Removed `--simple-io` flag; redirected stderr to `DEVNULL`; check `returncode` for errors |
| 8 | `test_pipeline.py` hardcoded `WhisperModel` ignoring config | Replaced with `Transcriber` service call so backend config is respected |
| 9 | Piper TTS ALSA underrun warning | Synthesise to temp WAV file first, then play via `sounddevice` (not raw pipe to aplay) |
| 10 | VS Code showing 117 pending items after push | False positive — VS Code git cache not refreshed after `.gitignore` update; run Git: Refresh in VS Code |
| 11 | `tflite_runtime` crashes with NumPy 2.x (`_ARRAY_API not found`) | Switch openWakeWord to ONNX backend: `inference_framework="onnx"`. Both `.tflite` and `.onnx` models are downloaded — ONNX works with NumPy 2.x via onnxruntime. |

---

## What is NOT in the git repo (download/build separately)
Run these scripts after cloning:
```bash
bash scripts/install_piper.sh              # Piper binary + voice model
bash scripts/install_llama_cpp_python.sh   # llama-cpp-python with CUDA
# Then manually clone and build:
git clone --depth=1 https://github.com/ggerganov/llama.cpp.git
git clone --depth=1 https://github.com/ggerganov/whisper.cpp.git
# Build both with: cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87
# Download models from HuggingFace (see config/settings.yaml for paths)
```

---

## Remaining phases

### Phase 3 — Custom wake word + RAG (May–June 2026)
- Train custom `hey_ivona` wake word model with openWakeWord
- Build RAG layer for conference-specific knowledge (agenda, speakers, AI topics)
- Fine-tune system prompt for concise, conversational conference responses

### Phase 4 — Conference hardening (June–July 2026)
- Noise stress test in a loud environment (conference hall acoustics)
- `systemd` service for auto-start on power-on
- LED/display state indicator (listening / thinking / speaking)
- 8-hour offline endurance test (memory leaks, thermal throttling)
- Physical enclosure and power supply finalized
