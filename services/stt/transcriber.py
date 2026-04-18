import logging
import queue
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000


class Transcriber:
    def __init__(self, cfg: dict):
        self._device_index = cfg.get("input_device_index")
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

    def record_until_silence(self, max_seconds: float = 15.0) -> np.ndarray | None:
        """Record from the mic until the speaker stops, up to max_seconds."""
        audio_q: queue.Queue[np.ndarray] = queue.Queue()

        def _callback(indata, frames, time_info, status):
            channel = min(self._input_channel, indata.shape[1] - 1)
            audio_q.put(indata[:, channel].copy())

        frames: list[np.ndarray] = []
        silence_frames = 0
        max_frames = int(max_seconds * SAMPLE_RATE / 512)
        silence_limit = int(self._silence_threshold * SAMPLE_RATE / 512)

        log.debug("Recording...")
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=6,
            dtype="float32",
            device=self._device_index,
            blocksize=512,
            callback=_callback,
        ):
            while len(frames) < max_frames:
                try:
                    chunk = audio_q.get(timeout=0.5)
                except queue.Empty:
                    continue
                frames.append(chunk)
                rms = np.sqrt(np.mean(chunk ** 2))
                if rms < 0.01:
                    silence_frames += 1
                    if silence_frames >= silence_limit:
                        break
                else:
                    silence_frames = 0

        if not frames:
            return None
        audio = np.concatenate(frames)
        log.debug("Recorded %.2f seconds.", len(audio) / SAMPLE_RATE)
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
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            sf.write(tmp_path, audio, SAMPLE_RATE, subtype="PCM_16")
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
                check=True,
            )
            lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
            return " ".join(lines).strip()
        finally:
            Path(tmp_path).unlink(missing_ok=True)
