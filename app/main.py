"""
ivona_bot — main pipeline
Wake word → STT → LLM → TTS loop

Usage:
    python app/main.py              # normal mode
    python app/main.py --debug-stt  # show mic levels and raw transcript each turn
"""

import argparse
import logging
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "settings.yaml"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def setup_logging(cfg: dict) -> None:
    log_cfg = cfg.get("logging", {})
    log_file = ROOT / log_cfg.get("file", "logs/ivona.log")
    log_file.parent.mkdir(exist_ok=True)
    logging.basicConfig(
        level=log_cfg.get("level", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )


def preflight_check(cfg: dict) -> None:
    """Fail fast with a clear message if any required binary or model is missing."""
    log = logging.getLogger("ivona.preflight")
    errors = []

    checks = [
        (ROOT / cfg["stt"]["whisper_cpp_binary"],  "whisper-cli binary (build whisper.cpp)"),
        (ROOT / cfg["stt"]["whisper_cpp_model"],   "Whisper model (run: whisper.cpp/models/download-ggml-model.sh base.en)"),
        (ROOT / cfg["tts"]["binary_path"],          "Piper binary (run: bash scripts/install_piper.sh)"),
        (ROOT / cfg["tts"]["model_path"],           "Piper voice model (run: bash scripts/install_piper.sh)"),
    ]

    # LLM: split GGUF — check the first file
    llm_path = ROOT / cfg["llm"]["model_path"]
    checks.append((llm_path, f"LLM model {llm_path.name} (download from HuggingFace — see README)"))

    for path, description in checks:
        if not path.exists():
            errors.append(f"  MISSING: {path}\n    → {description}")

    # Check parecord is available
    if subprocess.run(["which", "parecord"], capture_output=True).returncode != 0:
        errors.append("  MISSING: parecord\n    → Install PulseAudio: sudo apt install pulseaudio-utils")

    if errors:
        log.error("Preflight check failed — cannot start:\n%s", "\n".join(errors))
        sys.exit(1)

    log.info("Preflight check passed.")


def _keep_mic_active(pulse_source: str | None) -> None:
    """Prevent PulseAudio from suspending the ReSpeaker source between turns."""
    log = logging.getLogger("ivona")

    if pulse_source:
        # Use the configured source name directly
        result = subprocess.run(["pactl", "list", "sources", "short"],
                                capture_output=True, text=True)
        for line in result.stdout.splitlines():
            if pulse_source in line:
                source_index = line.split()[0]
                subprocess.run(["pactl", "suspend-source", source_index, "0"],
                               capture_output=True)
                log.info("Mic source '%s' kept active (suspend disabled)", pulse_source)
                return
        log.warning("Configured pulse_source '%s' not found in pactl output", pulse_source)
        return

    # Auto-detect ReSpeaker
    result = subprocess.run(["pactl", "list", "sources", "short"],
                            capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if "respeaker" in line.lower() or "xvf3800" in line.lower():
            source_index = line.split()[0]
            subprocess.run(["pactl", "suspend-source", source_index, "0"],
                           capture_output=True)
            log.info("ReSpeaker source %s kept active (suspend disabled)", source_index)
            return
    log.warning("ReSpeaker PulseAudio source not found — mic may suspend between turns")


def stt_debug_print(audio: np.ndarray, transcript: str) -> None:
    duration = len(audio) / 16000
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    bar = "#" * min(40, int(rms * 800))
    print()
    print("┌─── STT DEBUG ──────────────────────────────────────┐")
    print(f"│  Duration : {duration:.2f}s")
    print(f"│  RMS      : {rms:.4f}  Peak: {peak:.4f}")
    print(f"│  Level    : [{bar:<40}]")
    print(f"│  Transcript: {transcript!r}")
    print("└────────────────────────────────────────────────────┘")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="ivona_bot voice assistant")
    parser.add_argument(
        "--debug-stt",
        action="store_true",
        help="Print mic audio levels and raw transcript after each recording",
    )
    args = parser.parse_args()

    cfg = load_config()
    setup_logging(cfg)
    log = logging.getLogger("ivona")

    log.info("Starting ivona_bot%s...", " [STT debug ON]" if args.debug_stt else "")

    preflight_check(cfg)

    pulse_source = cfg.get("wake_word", {}).get("pulse_source") or cfg.get("audio", {}).get("pulse_source")
    _keep_mic_active(pulse_source)

    from services.wake_word.detector import WakeWordDetector
    from services.stt.transcriber import Transcriber
    from services.llm.responder import Responder
    from services.tts.speaker import Speaker

    wake = WakeWordDetector(cfg["wake_word"])
    stt = Transcriber(cfg["stt"])
    llm = Responder(cfg["llm"])
    tts = Speaker(cfg["tts"])

    def shutdown(sig, frame):
        log.info("Shutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    log.info("Ready. Listening for wake word '%s'...", cfg["wake_word"]["wake_word"])

    while True:
        try:
            wake.wait_for_wake_word()
            log.info("Wake word detected. Listening for question...")
            tts.speak("Yes?")
            time.sleep(0.4)  # let speaker audio settle before mic opens

            audio = stt.record_until_silence()

            if audio is None or len(audio) < 8000:
                if args.debug_stt:
                    print("\n[STT DEBUG] No audio captured (< 0.5s)\n")
                log.warning("No question captured, resuming wake word detection.")
                continue

            transcript = stt.transcribe(audio)

            if args.debug_stt:
                stt_debug_print(audio, transcript)

            if not transcript.strip():
                log.info("Empty transcript — resuming wake word detection.")
                continue

            log.info("User said: %s", transcript)

            # Streaming pipeline: LLM generates sentences in a background thread;
            # TTS speaks each sentence as it arrives so the first word is heard
            # ~2-3s after the question instead of waiting for the full response.
            sentence_q: queue.Queue[str | None] = queue.Queue()
            full_response: list[str] = []

            def _generate():
                try:
                    for sentence in llm.stream_sentences(transcript):
                        log.debug("LLM sentence: %s", sentence)
                        sentence_q.put(sentence)
                except Exception as exc:
                    log.error("LLM generation error: %s", exc)
                finally:
                    sentence_q.put(None)

            gen_thread = threading.Thread(target=_generate, daemon=True)
            gen_thread.start()

            while True:
                sentence = sentence_q.get()
                if sentence is None:
                    break
                full_response.append(sentence)
                try:
                    tts.speak(sentence)
                except Exception as exc:
                    log.error("TTS error: %s", exc)

            gen_thread.join()
            log.info("Response: %s", " ".join(full_response))

        except KeyboardInterrupt:
            raise
        except Exception as exc:
            log.error("Unexpected error during turn (recovering): %s", exc, exc_info=True)
            time.sleep(1.0)  # brief pause before resuming wake word detection


if __name__ == "__main__":
    main()
