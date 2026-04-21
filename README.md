# ivona_bot

A fully **offline** LLM-powered conversational voice assistant running on an NVIDIA Jetson Orin Nano Super. Built for the **AI Conference for Good** — visitors ask questions by voice and Ivona answers using on-device AI with zero internet connectivity.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ivona_bot pipeline                           │
│                                                                     │
│  ReSpeaker XVF3800                                                  │
│  4-Mic Array (USB)                                                  │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────┐    ┌──────┐ │
│  │  Wake Word  │───▶│     STT      │───▶│    LLM     │───▶│ TTS  │ │
│  │openWakeWord │    │ whisper.cpp  │    │llama-cpp-  │    │Piper │ │
│  │   (CPU)     │    │   (CPU)*     │    │python(GPU) │    │(CPU) │ │
│  │             │    │              │    │            │    │      │ │
│  │ "hey jarvis"│    │ggml-base.en  │    │Qwen2.5-7B  │    │amy-  │ │
│  │  ONNX model │    │  ~2-3s       │    │Q4KM ~9s    │    │medium│ │
│  └─────────────┘    └──────────────┘    └────────────┘    └──────┘ │
│                           * CUDA disabled so whisper-cli doesn't    │
│                             compete with LLM for unified memory     │
│                                                │                    │
│                                         Streaming:                  │
│                                    sentence 1 → TTS                 │
│                                    sentence 2 → TTS  (parallel)     │
│                                         ...                         │
│                                                     │               │
│                                                     ▼               │
│                                          Bluetooth Speaker          │
│                                          BM4D (A2DP 44.1kHz)        │
└─────────────────────────────────────────────────────────────────────┘

Audio I/O:  PulseAudio  (parecord for capture, sounddevice for playback)
```

**Key design decisions:**
- All inference is fully offline — no network calls at runtime
- LLM and STT share the 8GB unified RAM/VRAM (Jetson has no separate VRAM)
- Streaming LLM→TTS pipeline: first sentence is spoken while the model generates the rest, reducing perceived latency from ~9s to ~2-3s
- Wake word uses ONNX backend (not tflite) for NumPy 2.x compatibility
- Audio capture uses PulseAudio (`parecord`) instead of direct ALSA to work correctly with the ReSpeaker's DSP

---

## Hardware

| Component | Detail |
|-----------|--------|
| Compute | NVIDIA Jetson Orin Nano Super — 8GB unified RAM/VRAM, Ampere GPU (sm_87), CUDA 12.6, JetPack 6.1 |
| Storage | 1TB NVMe SSD (M.2 2280) |
| Microphone | ReSpeaker XMOS XVF3800 4-Mic Array — USB, 6ch 16kHz, sounddevice index 0 |
| Speaker | Bluetooth BM4D — A2DP profile (44.1kHz stereo) |

---

## Repository layout

```
ivona_bot/
├── app/
│   └── main.py                  # Main wake→STT→LLM→TTS loop
├── config/
│   └── settings.yaml            # All runtime config (devices, models, backends)
├── models/                      # Downloaded separately — not in git
│   ├── LLM/qwen_2_5/            # Qwen2.5-7B-Instruct Q4_K_M split GGUF
│   ├── STT/ggml-base.en.bin     # Whisper base GGML model (142MB)
│   ├── TTS/en_US-amy-medium.onnx
│   └── wake_word/               # openWakeWord built-in models (auto-downloaded)
├── scripts/
│   ├── install_piper.sh         # Download Piper TTS binary + voice model
│   ├── install_llama_cpp_python.sh  # Build llama-cpp-python with CUDA
│   └── test_pipeline.py         # Per-service smoke tests
├── services/
│   ├── wake_word/detector.py    # openWakeWord listener (PulseAudio)
│   ├── stt/transcriber.py       # whisper.cpp subprocess STT
│   ├── llm/responder.py         # llama-cpp-python LLM with sentence streaming
│   └── tts/speaker.py           # Piper TTS → sounddevice playback
└── requirements.txt
```

---

## Setup

### Prerequisites

- Jetson Orin Nano Super running JetPack 6.1 (L4T 36.5.0)
- CUDA 12.6, Python 3.10
- ReSpeaker XVF3800 USB mic connected
- Bluetooth speaker paired and set to A2DP profile

### 1 — Clone and create virtual environment

```bash
git clone https://github.com/voxfox01/ivona_bot.git
cd ivona_bot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2 — Install llama-cpp-python with CUDA

This compiles from source (~45 minutes on Jetson):

```bash
bash scripts/install_llama_cpp_python.sh
```

### 3 — Build llama.cpp and whisper.cpp

```bash
# llama.cpp
git clone --depth=1 https://github.com/ggerganov/llama.cpp.git
cmake -B llama.cpp/build -S llama.cpp -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build llama.cpp/build -j4

# whisper.cpp
git clone --depth=1 https://github.com/ggerganov/whisper.cpp.git
cmake -B whisper.cpp/build -S whisper.cpp -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build whisper.cpp/build -j4
```

### 4 — Install Piper TTS

```bash
bash scripts/install_piper.sh
```

### 5 — Download models

**LLM — Qwen2.5-7B-Instruct Q4_K_M** (download both split files, ~4.4GB total):

```bash
mkdir -p models/LLM/qwen_2_5
python3 -c "
from huggingface_hub import hf_hub_download
for i in ['00001-of-00002', '00002-of-00002']:
    hf_hub_download(
        repo_id='Qwen/Qwen2.5-7B-Instruct-GGUF',
        filename=f'qwen2.5-7b-instruct-q4_k_m-{i}.gguf',
        local_dir='models/LLM/qwen_2_5'
    )
"
```

**STT — Whisper base** (142MB):

```bash
mkdir -p models/STT
whisper.cpp/models/download-ggml-model.sh base.en
mv whisper.cpp/models/ggml-base.en.bin models/STT/
```

**Wake word** — downloaded automatically on first run by openWakeWord.

### 6 — Audio setup

The ReSpeaker must be selected as the PulseAudio default input source. Open `pavucontrol`, go to **Input Devices**, and set the ReSpeaker XVF3800 as the default. This persists across reboots.

For the Bluetooth speaker, switch it from HFP (8kHz) to A2DP (44.1kHz stereo):

```bash
pactl set-card-profile bluez_card.<MAC_ADDRESS> a2dp_sink
```

Replace `<MAC_ADDRESS>` with your speaker's MAC (find it with `pactl list cards short`).

---

## Running

### Full pipeline

```bash
source .venv/bin/activate
python app/main.py
```

Say **"hey jarvis"** to activate, then ask your question. Ivona will respond via the speaker.

### Debug mode (shows mic levels and transcript)

```bash
python app/main.py --debug-stt
```

After each "Yes?", the terminal prints the audio RMS level and the raw STT transcript — useful for diagnosing mic or transcription issues.

### Test individual services

```bash
python scripts/test_pipeline.py                        # test all
python scripts/test_pipeline.py --skip-wake            # skip wake word
python scripts/test_pipeline.py --skip-wake --skip-stt # LLM + TTS only
```

---

## Configuration

All runtime settings are in `config/settings.yaml`:

| Setting | Default | Notes |
|---------|---------|-------|
| `wake_word.wake_word` | `hey jarvis` | Built-in openWakeWord model |
| `wake_word.threshold` | `0.5` | Lower = more sensitive |
| `stt.whisper_cpp_model` | `models/STT/ggml-base.en.bin` | Swap for `ggml-small.en.bin` for better accuracy |
| `llm.n_gpu_layers` | `20` | Max safe value is 28; stay at 20 with STT co-resident |
| `llm.max_tokens` | `512` | Keep short for conference floor responses |
| `llm.system_prompt` | See file | Edit to change Ivona's persona |

---

## Memory budget (8GB unified)

| Component | RAM/VRAM |
|-----------|----------|
| OS + GNOME + idle processes | ~2.5GB |
| LLM (Qwen 7B Q4KM, 20 GPU layers) | ~2.7GB |
| whisper-cli subprocess (base model) | ~0.4GB |
| Python runtime + models | ~0.5GB |
| **Headroom** | **~1.9GB** |

> Do not increase `n_gpu_layers` above 28. With `ggml-base` STT and 20 GPU layers the system runs comfortably. Switching back to `ggml-medium` (1.5GB) will cause OOM.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Wake word never triggers | Open `pavucontrol` and confirm ReSpeaker is the default input source |
| `PortAudioError: Invalid number of channels` | Device index changed; run `python -c "import sounddevice as sd; print(sd.query_devices())"` and update `input_device_index` in `settings.yaml` |
| `Killed` during STT | OOM — reduce `n_gpu_layers` or switch to `ggml-base.en.bin` |
| Transcript is `(buzzing)` or empty | Mic capturing noise only; check `input_channel` in `settings.yaml` (use channel with highest RMS from the channel test below) |
| Bluetooth speaker sounds like 8kHz mono | Switch profile: `pactl set-card-profile bluez_card.<MAC> a2dp_sink` |
| onnxruntime GPU warning on startup | Harmless — wake word runs on CPU intentionally |

**Channel RMS test** (find the best mic channel):

```bash
python -c "
import sounddevice as sd, numpy as np
audio = sd.rec(3*16000, samplerate=16000, channels=6, dtype='float32', device=0)
sd.wait()
for ch in range(6):
    rms = float(np.sqrt(np.mean(audio[:,ch]**2)))
    print(f'ch{ch}: RMS={rms:.4f}')
"
```

---

## Roadmap

- **Phase 3** (May–June 2026): Custom `hey ivona` wake word model + RAG layer for conference knowledge
- **Phase 4** (June–July 2026): systemd auto-start, noise stress testing, 8-hour endurance test, LED state indicator
