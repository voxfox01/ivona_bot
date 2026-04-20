"""
Smoke-tests each pipeline service in isolation.
Run from the project root:
    source .venv/bin/activate
    python scripts/test_pipeline.py [--skip-wake] [--skip-stt] [--skip-llm] [--skip-tts]
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("test_pipeline")

with open(ROOT / "config" / "settings.yaml") as f:
    cfg = yaml.safe_load(f)


def test_tts(cfg):
    log.info("=== TTS TEST ===")
    from services.tts.speaker import Speaker
    tts = Speaker(cfg["tts"])
    tts.speak("Hello! I am Ivona. The text to speech system is working correctly.")
    log.info("TTS OK")


def test_stt(cfg):
    import time
    import numpy as np
    log.info("=== STT TEST ===")
    from services.stt.transcriber import Transcriber
    stt = Transcriber(cfg["stt"])

    # Fixed 5-second recording with countdown so mic captures real speech
    log.info("Recording in 3...")
    time.sleep(1)
    log.info("Recording in 2...")
    time.sleep(1)
    log.info("Recording in 1...")
    time.sleep(1)
    log.info("RECORDING NOW — speak your question (5 seconds)...")
    audio = stt.record_until_silence(max_seconds=5.0)
    log.info("Recording complete.")

    if audio is None or len(audio) == 0:
        log.warning("No audio captured.")
        return ""

    duration = len(audio) / 16000
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    log.info("Audio: %.2f seconds | RMS=%.4f | Peak=%.4f", duration, rms, peak)
    if rms < 0.005:
        log.warning("Audio level very low — check mic selection (input_device_index/input_channel in settings.yaml)")

    log.info("Sending to STT model...")
    text = stt.transcribe(audio)

    print("\n" + "=" * 60)
    print(f"  TRANSCRIPT: {text!r}")
    print("=" * 60 + "\n")
    log.info("STT OK")
    return text


def test_llm(cfg, prompt="What is artificial intelligence in two sentences?"):
    log.info("=== LLM TEST ===")
    from services.llm.responder import Responder
    llm = Responder(cfg["llm"])
    response = llm.generate(prompt)
    log.info("Response: %s", response)
    log.info("LLM OK")
    return response


def test_wake_word(cfg):
    log.info("=== WAKE WORD TEST (say the wake word) ===")
    from services.wake_word.detector import WakeWordDetector
    wake = WakeWordDetector(cfg["wake_word"])
    log.info("Listening... say '%s'", cfg["wake_word"]["wake_word"])
    wake.wait_for_wake_word()
    log.info("Wake word detected! WAKE WORD OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-wake", action="store_true")
    parser.add_argument("--skip-stt", action="store_true")
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--skip-tts", action="store_true")
    args = parser.parse_args()

    if not args.skip_tts:
        test_tts(cfg)

    if not args.skip_stt:
        transcript = test_stt(cfg)
    else:
        transcript = "What can you tell me about AI for social good?"

    if not args.skip_llm:
        response = test_llm(cfg, transcript)
        if not args.skip_tts:
            log.info("=== SPEAKING LLM RESPONSE ===")
            from services.tts.speaker import Speaker
            Speaker(cfg["tts"]).speak(response)

    if not args.skip_wake:
        test_wake_word(cfg)

    log.info("=== ALL TESTS PASSED ===")
