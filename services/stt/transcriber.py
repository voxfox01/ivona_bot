import collections
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import webrtcvad
from faster_whisper import WhisperModel

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
VAD_FRAME_MS = 30                          # webrtcvad supports 10/20/30ms
VAD_FRAME_SAMPLES = SAMPLE_RATE * VAD_FRAME_MS // 1000   # 480 samples
VAD_FRAME_BYTES = VAD_FRAME_SAMPLES * 2    # s16le = 2 bytes/sample


class Transcriber:
    def __init__(self, cfg: dict):
        self._language = cfg.get("language", "en")
        self._silence_threshold = cfg.get("silence_threshold_seconds", 1.5)
        self._backend = cfg.get("backend", "faster_whisper")
        self._pulse_source = cfg.get("pulse_source")

        # VAD config
        vad_cfg = cfg.get("vad", {})
        self._vad_aggressiveness = vad_cfg.get("aggressiveness", 2)  # 0–3
        self._vad_speech_pad_ms = vad_cfg.get("speech_pad_ms", 300)  # pre-roll buffer
        self._vad_end_silence_ms = vad_cfg.get("end_silence_ms", 900) # silence to end utterance

        self._vad = webrtcvad.Vad(self._vad_aggressiveness)

        if self._backend == "whisper_cpp":
            self._whisper_bin = Path(cfg["whisper_cpp_binary"]).resolve()
            self._whisper_model = Path(cfg["whisper_cpp_model"]).resolve()
            log.info("STT using whisper.cpp backend: %s", self._whisper_model.name)
        else:
            log.info("Loading faster-whisper '%s' on %s (%s)...",
                     cfg.get("model_size", "base"), cfg.get("device", "cpu"),
                     cfg.get("compute_type", "int8"))
            self._fw_model = WhisperModel(
                cfg.get("model_size", "base"),
                device=cfg.get("device", "cpu"),
                compute_type=cfg.get("compute_type", "int8"),
                download_root=str(Path(cfg["model_dir"]).resolve()),
                local_files_only=True,
            )
            log.info("faster-whisper loaded.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open_stream(self) -> subprocess.Popen:
        """Start a persistent parecord stream. Caller owns the process."""
        cmd = ["parecord", "--channels=1", f"--rate={SAMPLE_RATE}",
               "--format=s16le", "--raw"]
        if self._pulse_source:
            cmd += [f"--device={self._pulse_source}"]
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def record_with_vad(self, stream: subprocess.Popen,
                        max_seconds: float = 15.0) -> np.ndarray | None:
        """Read from an open parecord stream using VAD to find speech boundaries.

        Returns float32 audio of the spoken utterance, or None if max_seconds
        elapses without detecting speech.  The stream is left open for reuse.

        Speech onset requires ONSET_FRAMES consecutive voiced frames to avoid
        triggering on transient noise.  A minimum RMS gate additionally filters
        out low-level ambient triggers.
        """
        end_silence_frames = int(self._vad_end_silence_ms / VAD_FRAME_MS)
        pre_roll_frames = int(self._vad_speech_pad_ms / VAD_FRAME_MS)
        max_frames = int(max_seconds * 1000 / VAD_FRAME_MS)
        onset_frames = 3   # require 3 consecutive voiced frames (~90ms) to start

        ring = collections.deque(maxlen=pre_roll_frames)
        speech_frames: list[bytes] = []
        in_speech = False
        silence_count = 0
        onset_count = 0         # consecutive voiced frames seen so far
        onset_buffer: list[bytes] = []  # frames accumulated during onset check

        for _ in range(max_frames):
            raw = stream.stdout.read(VAD_FRAME_BYTES)
            if not raw or len(raw) < VAD_FRAME_BYTES:
                break

            is_speech = self._vad.is_speech(raw, SAMPLE_RATE)

            if not in_speech:
                ring.append(raw)
                if is_speech:
                    onset_count += 1
                    onset_buffer.append(raw)
                    if onset_count >= onset_frames:
                        # Confirmed speech onset — include pre-roll + onset frames
                        in_speech = True
                        speech_frames = list(ring)[:-len(onset_buffer)] + onset_buffer
                        silence_count = 0
                else:
                    onset_count = 0
                    onset_buffer.clear()
            else:
                speech_frames.append(raw)
                if not is_speech:
                    silence_count += 1
                    if silence_count >= end_silence_frames:
                        break
                else:
                    silence_count = 0

        if not speech_frames:
            return None

        raw_bytes = b"".join(speech_frames)
        audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768

        # RMS gate: reject if audio is just ambient noise
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < 0.008:
            log.debug("VAD triggered but RMS %.4f too low — ignoring", rms)
            return None

        log.debug("VAD captured %.2fs speech (RMS=%.4f)", len(audio) / SAMPLE_RATE, rms)
        return audio

    def record_until_silence(self, max_seconds: float = 15.0,
                             min_seconds: float = 2.0) -> np.ndarray | None:
        """Fallback recorder (file-polling parecord). Used by test_pipeline.py."""
        tmp = tempfile.NamedTemporaryFile(suffix=".raw", delete=False)
        tmp_path = tmp.name
        tmp.close()

        cmd = ["parecord", "--channels=1", f"--rate={SAMPLE_RATE}",
               "--format=s16le", "--raw", tmp_path]
        if self._pulse_source:
            cmd += [f"--device={self._pulse_source}"]
        proc = subprocess.Popen(cmd, stderr=subprocess.DEVNULL)

        chunk_size = 512
        silence_limit = int(self._silence_threshold * SAMPLE_RATE / chunk_size)
        max_chunks = int(max_seconds * SAMPLE_RATE / chunk_size)
        min_chunks = int(min_seconds * SAMPLE_RATE / chunk_size)
        silence_chunks = 0
        total_chunks = 0
        bytes_per_chunk = chunk_size * 2

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
                    continue
                rms = float(np.sqrt(np.mean(chunk_audio ** 2))) if len(chunk_audio) > 0 else 0.0
                if rms < 0.005:
                    silence_chunks += 1
                    if silence_chunks >= silence_limit:
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
        log.debug("Recorded %.2fs via parecord polling.", len(audio) / SAMPLE_RATE)
        return audio

    def transcribe(self, audio: np.ndarray) -> str:
        if self._backend == "whisper_cpp":
            return self._transcribe_whisper_cpp(audio)
        return self._transcribe_faster_whisper(audio)

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------

    def _transcribe_faster_whisper(self, audio: np.ndarray) -> str:
        segments, _ = self._fw_model.transcribe(
            audio,
            beam_size=1,        # greedy decoding — ~3x faster than beam_size=5
            language=self._language,
            vad_filter=True,
        )
        return " ".join(s.text for s in segments).strip()

    def _transcribe_whisper_cpp(self, audio: np.ndarray) -> str:
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < 0.005:
            return ""
        if len(audio) < SAMPLE_RATE:
            audio = np.pad(audio, (0, SAMPLE_RATE - len(audio)))

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            sf.write(tmp_path, audio, SAMPLE_RATE, subtype="PCM_16")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ""
            result = subprocess.run(
                [str(self._whisper_bin), "-m", str(self._whisper_model),
                 "-f", tmp_path, "-l", self._language, "--no-timestamps", "-t", "4"],
                capture_output=True, text=True, env=env,
            )
            if result.returncode != 0:
                log.warning("whisper-cli exited %d", result.returncode)
                return ""
            lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
            return " ".join(lines).strip()
        finally:
            Path(tmp_path).unlink(missing_ok=True)
