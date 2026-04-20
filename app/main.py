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


def _keep_mic_active() -> None:
    """Prevent PulseAudio from suspending the ReSpeaker source.
    Without this, the source suspends between recordings and takes ~1-2s to
    wake up, causing parecord to return silence until it's fully active."""
    import subprocess
    result = subprocess.run(["pactl", "list", "sources", "short"],
                            capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if "respeaker" in line.lower() or "xvf3800" in line.lower():
            source_index = line.split()[0]
            subprocess.run(["pactl", "suspend-source", source_index, "0"],
                           capture_output=True)
            logging.getLogger("ivona").info(
                "ReSpeaker source %s kept active (suspend disabled)", source_index)
            return
    logging.getLogger("ivona").warning(
        "ReSpeaker PulseAudio source not found — mic may suspend between turns")


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
    _keep_mic_active()

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
            for sentence in llm.stream_sentences(transcript):
                log.debug("LLM sentence: %s", sentence)
                sentence_q.put(sentence)
            sentence_q.put(None)  # sentinel

        gen_thread = threading.Thread(target=_generate, daemon=True)
        gen_thread.start()

        while True:
            sentence = sentence_q.get()
            if sentence is None:
                break
            full_response.append(sentence)
            tts.speak(sentence)

        gen_thread.join()
        log.info("Response: %s", " ".join(full_response))


if __name__ == "__main__":
    main()
