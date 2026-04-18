"""
ivona_bot — main pipeline
Wake word → STT → LLM → TTS loop
"""

import logging
import signal
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "settings.yaml"

# Add project root to path so `services.*` imports resolve from anywhere
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


def main() -> None:
    cfg = load_config()
    setup_logging(cfg)
    log = logging.getLogger("ivona")

    log.info("Starting ivona_bot...")

    # Lazy imports so startup logging appears before heavy model loads
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
        log.info("Wake word detected.")

        audio = stt.record_until_silence()
        if audio is None:
            continue

        transcript = stt.transcribe(audio)
        if not transcript.strip():
            continue

        log.info("User said: %s", transcript)

        response = llm.generate(transcript)
        log.info("Response: %s", response)

        tts.speak(response)


if __name__ == "__main__":
    main()
