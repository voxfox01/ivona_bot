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
    log.info("=== STT TEST (5 seconds — speak now) ===")
    from services.stt.transcriber import Transcriber
    stt = Transcriber(cfg["stt"])
    audio = stt.record_until_silence(max_seconds=5.0)
    if audio is None:
        log.warning("No audio captured.")
        return ""
    text = stt.transcribe(audio)
    log.info("Transcript: '%s'", text)
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
