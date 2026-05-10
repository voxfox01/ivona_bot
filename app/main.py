"""
ivona_bot — main pipeline
Wake word → conversation session (VAD multi-turn) → wake word

Usage:
    python app/main.py              # normal mode
    python app/main.py --debug-stt  # show mic levels and raw transcript each turn
"""

import argparse
import logging
import queue
import re
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
        (ROOT / cfg["tts"]["binary_path"],  "Piper binary (run: bash scripts/install_piper.sh)"),
        (ROOT / cfg["tts"]["model_path"],   "Piper voice model (run: bash scripts/install_piper.sh)"),
        (ROOT / cfg["llm"]["model_path"],   f"LLM model (download from HuggingFace — see README)"),
    ]
    if cfg["stt"]["backend"] == "whisper_cpp":
        checks += [
            (ROOT / cfg["stt"]["whisper_cpp_binary"], "whisper-cli binary (build whisper.cpp)"),
            (ROOT / cfg["stt"]["whisper_cpp_model"],  "Whisper model"),
        ]

    for path, description in checks:
        if not Path(path).exists():
            errors.append(f"  MISSING: {path}\n    → {description}")

    if subprocess.run(["which", "parecord"], capture_output=True).returncode != 0:
        errors.append("  MISSING: parecord\n    → sudo apt install pulseaudio-utils")

    if errors:
        log.error("Preflight check failed:\n%s", "\n".join(errors))
        sys.exit(1)
    log.info("Preflight check passed.")


def _keep_mic_active(pulse_source: str | None) -> None:
    """Prevent PulseAudio from suspending the ReSpeaker source between turns."""
    log = logging.getLogger("ivona")
    result = subprocess.run(["pactl", "list", "sources", "short"],
                            capture_output=True, text=True)
    for line in result.stdout.splitlines():
        match = pulse_source and pulse_source in line
        auto = not pulse_source and ("respeaker" in line.lower() or "xvf3800" in line.lower())
        if match or auto:
            idx = line.split()[0]
            subprocess.run(["pactl", "suspend-source", idx, "0"], capture_output=True)
            log.info("Mic source %s kept active (suspend disabled)", idx)
            return
    log.warning("ReSpeaker PulseAudio source not found — mic may suspend between turns")


def _sample_tegrastats() -> str:
    """Return a one-line power+thermal summary from tegrastats."""
    try:
        proc = subprocess.Popen(
            ["tegrastats"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        line = proc.stdout.readline()
        proc.terminate()
        proc.wait()
        vdd = re.search(r"VDD_IN (\d+)mW", line)
        gpu_temp = re.search(r"gpu@([\d.]+)C", line)
        cpu_temp = re.search(r"cpu@([\d.]+)C", line)
        parts = []
        if vdd:
            parts.append(f"{int(vdd.group(1)):,}mW")
        if cpu_temp:
            parts.append(f"CPU {float(cpu_temp.group(1)):.0f}°C")
        if gpu_temp:
            parts.append(f"GPU {float(gpu_temp.group(1)):.0f}°C")
        return "  ".join(parts) if parts else "n/a"
    except Exception:
        return "n/a"


def stt_debug_print(audio: np.ndarray, transcript: str) -> None:
    duration = len(audio) / 16000
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    bar = "#" * min(40, int(rms * 800))
    hw = _sample_tegrastats()
    print()
    print("┌─── STT DEBUG ──────────────────────────────────────┐")
    print(f"│  Duration : {duration:.2f}s")
    print(f"│  RMS      : {rms:.4f}  Peak: {peak:.4f}")
    print(f"│  Level    : [{bar:<40}]")
    print(f"│  Hardware : {hw}")
    print(f"│  Transcript: {transcript!r}")
    print("└────────────────────────────────────────────────────┘")
    print()


def _clean_for_tts(text: str) -> str:
    """Strip markdown and model-format tokens so Piper speaks natural text."""
    # Model format tokens — defensive in case stop tokens miss a boundary
    text = re.sub(r'<\|?(?:im_start|im_end)\|?>', '', text)
    text = re.sub(r'<(?:start|end)_of_turn>', '', text)
    # Markdown links: [label](url) → label
    text = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', text)
    # Headings: leading # symbols
    text = re.sub(r'^\s*#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Bold / italic: **text**, *text*, __text__, _text_
    text = re.sub(r'\*{1,2}([^*]+?)\*{1,2}', r'\1', text)
    text = re.sub(r'_{1,2}([^_]+?)_{1,2}', r'\1', text)
    # Inline code: `text`
    text = re.sub(r'`([^`]*)`', r'\1', text)
    # Bullet / list markers at line start
    text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)
    # Emoji and miscellaneous Unicode symbols
    text = re.sub(r'[\U0001F000-\U0001FFFF]', '', text)
    text = re.sub(r'[☀-➿]', '', text)
    # Collapse internal newlines to a space
    text = re.sub(r'\s*\n\s*', ' ', text)
    return text.strip()


def _set_mic_mute(pulse_source: str | None, mute: bool) -> None:
    """Mute or unmute the PulseAudio mic source via pactl."""
    if not pulse_source:
        return
    state = "1" if mute else "0"
    subprocess.run(["pactl", "set-source-mute", pulse_source, state],
                   capture_output=True)


def _drain_mic_pipe(stream) -> None:
    """Discard audio buffered in the parecord pipe while the mic was unmuted.

    Even with OS-level muting, a few frames may be buffered at unmute time.
    Draining ensures record_with_vad() only sees live post-response audio.
    """
    import select
    while True:
        ready, _, _ = select.select([stream.stdout], [], [], 0)
        if not ready:
            break
        chunk = stream.stdout.read1(8192)  # type: ignore[attr-defined]
        if not chunk:
            break


def speak_streaming(llm, tts, transcript: str, log) -> str:
    """Run LLM in background thread, speak each sentence as it arrives."""
    sentence_q: queue.Queue[str | None] = queue.Queue()
    full_response: list[str] = []

    def _generate():
        try:
            for sentence in llm.stream_sentences(transcript):
                sentence_q.put(sentence)
        except Exception as exc:
            log.error("LLM generation error: %s", exc)
        finally:
            sentence_q.put(None)

    threading.Thread(target=_generate, daemon=True).start()

    while True:
        sentence = sentence_q.get()
        if sentence is None:
            break
        full_response.append(sentence)
        try:
            tts.speak(_clean_for_tts(sentence))
        except Exception as exc:
            log.error("TTS error: %s", exc)

    return " ".join(full_response)


def run_conversation_session(stt, llm, tts, cfg, pulse_source: str | None,
                             debug_stt: bool, log) -> None:
    """Open a multi-turn conversation session after wake word activation.

    Keeps a single parecord stream open. Uses webrtcvad to detect each new
    utterance without requiring the wake word again. The session ends after
    `session_timeout_seconds` of silence and control returns to wake word mode.
    """
    conv_cfg = cfg.get("conversation", {})
    timeout_seconds = conv_cfg.get("session_timeout_seconds", 30)
    end_phrase = conv_cfg.get("session_end_phrase", "Goodbye!")

    llm.reset_session()

    def speak_muted(text: str) -> None:
        """Mute mic, speak, unmute, then drain any residual pipe audio."""
        _set_mic_mute(pulse_source, mute=True)
        try:
            tts.speak(text)
        finally:
            _set_mic_mute(pulse_source, mute=False)
        _drain_mic_pipe(stream)

    def speak_streaming_muted(transcript: str) -> str:
        """Mute mic for the full LLM→TTS streaming response."""
        _set_mic_mute(pulse_source, mute=True)
        try:
            result = speak_streaming(llm, tts, transcript, log)
        finally:
            _set_mic_mute(pulse_source, mute=False)
        return result

    log.info("Conversation session started (timeout=%ds).", timeout_seconds)
    stream = stt.open_stream()

    try:
        last_speech_time = time.time()

        while True:
            # Check session timeout
            if time.time() - last_speech_time > timeout_seconds:
                log.info("Session timed out after %ds of silence.", timeout_seconds)
                speak_muted(end_phrase)
                break

            # Wait for next utterance (VAD), with a max wait = remaining timeout
            remaining = timeout_seconds - (time.time() - last_speech_time)
            audio = stt.record_with_vad(stream, max_seconds=min(remaining, 15.0))

            if audio is None:
                continue  # no speech detected in this window, loop to check timeout

            last_speech_time = time.time()

            transcript = stt.transcribe(audio)

            if debug_stt:
                stt_debug_print(audio, transcript)

            if not transcript.strip():
                log.debug("Empty transcript, continuing session.")
                continue

            log.info("User said: %s", transcript)
            response = speak_streaming_muted(transcript)
            log.info("Response: %s", response)

            # Drain any frames buffered at the moment of unmute, then reset timer.
            _drain_mic_pipe(stream)
            last_speech_time = time.time()

    except Exception as exc:
        log.error("Session error (recovering): %s", exc, exc_info=True)
    finally:
        stream.terminate()
        stream.wait()
        log.info("Conversation session ended.")


def main() -> None:
    parser = argparse.ArgumentParser(description="ivona_bot voice assistant")
    parser.add_argument("--debug-stt", action="store_true",
                        help="Print mic levels and raw transcript after each utterance")
    args = parser.parse_args()

    cfg = load_config()
    setup_logging(cfg)
    log = logging.getLogger("ivona")

    log.info("Starting ivona_bot%s...", " [STT debug ON]" if args.debug_stt else "")

    preflight_check(cfg)

    pulse_source = (cfg.get("wake_word", {}).get("pulse_source")
                    or cfg.get("audio", {}).get("pulse_source"))
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

    start_phrase = cfg.get("conversation", {}).get("session_start_phrase", "Yes?")
    wake_word = cfg["wake_word"]["wake_word"]
    log.info("Ready. Say '%s' to start a conversation.", wake_word)

    while True:
        try:
            wake.wait_for_wake_word()
            log.info("Wake word detected — starting session.")
            tts.speak(start_phrase)
            time.sleep(0.3)  # let speaker audio settle before opening mic stream
            run_conversation_session(stt, llm, tts, cfg, pulse_source, args.debug_stt, log)
            log.info("Returned to wake word listening.")
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            log.error("Unexpected error (recovering): %s", exc, exc_info=True)
            time.sleep(1.0)


if __name__ == "__main__":
    main()
