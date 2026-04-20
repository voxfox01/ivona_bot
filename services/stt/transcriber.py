import logging
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000


class Transcriber:
    def __init__(self, cfg: dict):
        self._input_channel = cfg.get("input_channel", 0)
        self._language = cfg.get("language", "en")
        self._silence_threshold = cfg.get("silence_threshold_seconds", 1.5)
        self._backend = cfg.get("backend", "faster_whisper")

        if self._backend == "whisper_cpp":
            self._whisper_bin = Path(cfg["whisper_cpp_binary"]).resolve()
            self._whisper_model = Path(cfg["whisper_cpp_model"]).resolve()
            log.info("STT using whisper.cpp backend: %s", self._whisper_model.name)
        else:
            log.info(
                "Loading Whisper %s on %s (%s)...",
                cfg["model_size"], cfg["device"], cfg["compute_type"],
            )
            self._model = WhisperModel(
                cfg["model_size"],
                device=cfg["device"],
                compute_type=cfg["compute_type"],
                download_root=str(Path(cfg["model_dir"]).resolve()),
            )
            log.info("Whisper loaded.")

    def record_until_silence(self, max_seconds: float = 15.0,
                             min_seconds: float = 2.0) -> np.ndarray | None:
        """Record via PulseAudio (parecord) until silence, up to max_seconds.

        min_seconds: always record at least this long before silence detection
        fires — gives the PulseAudio source time to unsuspend after wake word.
        """
        tmp = tempfile.NamedTemporaryFile(suffix=".raw", delete=False)
        tmp_path = tmp.name
        tmp.close()

        cmd = [
            "parecord",
            "--channels=1",
            f"--rate={SAMPLE_RATE}",
            "--format=s16le",
            "--raw",
            tmp_path,
        ]
        proc = subprocess.Popen(cmd, stderr=subprocess.DEVNULL)

        chunk_size = 512
        silence_limit_chunks = int(self._silence_threshold * SAMPLE_RATE / chunk_size)
        max_chunks = int(max_seconds * SAMPLE_RATE / chunk_size)
        min_chunks = int(min_seconds * SAMPLE_RATE / chunk_size)

        silence_chunks = 0
        total_chunks = 0
        bytes_per_chunk = chunk_size * 2  # s16le = 2 bytes per sample

        try:
            while total_chunks < max_chunks:
                time.sleep(chunk_size / SAMPLE_RATE)
                file_size = Path(tmp_path).stat().st_size
                available = file_size // bytes_per_chunk
                if available <= total_chunks:
                    continue
                with open(tmp_path, "rb") as f:
                    f.seek(total_chunks * bytes_per_chunk)
                    raw = f.read((available - total_chunks) * bytes_per_chunk)
                chunk_audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768
                total_chunks = available
                if total_chunks < min_chunks:
                    continue  # don't check silence until minimum time elapsed
                rms = float(np.sqrt(np.mean(chunk_audio ** 2))) if len(chunk_audio) > 0 else 0.0
                if rms < 0.005:
                    silence_chunks += 1
                    if silence_chunks >= silence_limit_chunks:
                        break
                else:
                    silence_chunks = 0
        finally:
            proc.terminate()
            proc.wait()

        try:
            raw_bytes = Path(tmp_path).read_bytes()
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            return None

        if not raw_bytes:
            return None

        audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768
        log.debug("Recorded %.2f seconds via PulseAudio.", len(audio) / SAMPLE_RATE)
        return audio

    def transcribe(self, audio: np.ndarray) -> str:
        if self._backend == "whisper_cpp":
            return self._transcribe_whisper_cpp(audio)
        return self._transcribe_faster_whisper(audio)

    def _transcribe_faster_whisper(self, audio: np.ndarray) -> str:
        segments, _ = self._model.transcribe(
            audio,
            beam_size=5,
            language=self._language,
            vad_filter=True,
        )
        return " ".join(s.text for s in segments).strip()

    def _transcribe_whisper_cpp(self, audio: np.ndarray) -> str:
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < 0.005:
            log.debug("Audio RMS %.4f too low — skipping transcription", rms)
            return ""

        # whisper.cpp requires at least ~1s of audio; pad if needed
        min_samples = SAMPLE_RATE  # 1 second
        if len(audio) < min_samples:
            audio = np.pad(audio, (0, min_samples - len(audio)))

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            sf.write(tmp_path, audio, SAMPLE_RATE, subtype="PCM_16")
            # CUDA_VISIBLE_DEVICES="" forces CPU-only so whisper-cli doesn't
            # compete with the resident LLM for GPU memory (would cause SIGABRT).
            import os
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ""
            result = subprocess.run(
                [
                    str(self._whisper_bin),
                    "-m", str(self._whisper_model),
                    "-f", tmp_path,
                    "-l", self._language,
                    "--no-timestamps",
                    "-t", "4",
                ],
                capture_output=True,
                text=True,
                env=env,
            )
            if result.returncode != 0:
                log.warning("whisper-cli exited %d — audio too short or corrupt", result.returncode)
                return ""
            lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
            return " ".join(lines).strip()
        finally:
            Path(tmp_path).unlink(missing_ok=True)
