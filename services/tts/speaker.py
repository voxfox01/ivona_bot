import json
import logging
import os
import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

log = logging.getLogger(__name__)


class Speaker:
    """Piper TTS with a persistent process for low-latency synthesis.

    Uses --json-input mode: each request writes a JSON line to stdin specifying
    the text and a temp output WAV path; Piper writes a JSON completion line to
    stdout when done. This gives clean synchronization without polling/select.
    """

    def __init__(self, cfg: dict):
        self._binary = Path(cfg["binary_path"]).resolve()
        self._model = Path(cfg["model_path"]).resolve()
        self._rate = cfg.get("speaking_rate", 1.0)
        self._lib_dir = str(self._binary.parent)

        if not self._binary.exists():
            raise FileNotFoundError(f"Piper binary not found: {self._binary}")
        if not self._model.exists():
            raise FileNotFoundError(f"Piper voice model not found: {self._model}")

        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

        log.info("TTS ready (model=%s)", self._model.name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak(self, text: str) -> None:
        """Synthesise text and play it. Blocks until audio finishes."""
        if not text.strip():
            return
        audio, sr = self._synthesise(text)
        sd.play(audio, samplerate=sr)
        sd.wait()

    def speak_async(self, text: str) -> threading.Thread:
        """Synthesise and play in a background thread. Returns the thread."""
        t = threading.Thread(target=self.speak, args=(text,), daemon=True)
        t.start()
        return t

    def close(self) -> None:
        """Terminate the persistent Piper process."""
        with self._lock:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
                self._proc.wait()
            self._proc = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_proc(self) -> subprocess.Popen:
        """Return the persistent Piper process, starting it if needed."""
        if self._proc is None or self._proc.poll() is not None:
            env = os.environ.copy()
            env["LD_LIBRARY_PATH"] = self._lib_dir + ":" + env.get("LD_LIBRARY_PATH", "")
            self._proc = subprocess.Popen(
                [
                    str(self._binary),
                    "--model", str(self._model),
                    "--json-input",
                    "--length_scale", str(1.0 / self._rate),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=env,
            )
            log.debug("Piper process started (pid=%d)", self._proc.pid)
        return self._proc

    def _synthesise(self, text: str) -> tuple[np.ndarray, int]:
        """Send text to Piper via --json-input and read the output WAV."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with self._lock:
                proc = self._get_proc()
                request = json.dumps({"text": text.strip(), "output_file": tmp_path})
                try:
                    proc.stdin.write(request.encode() + b"\n")
                    proc.stdin.flush()
                except BrokenPipeError:
                    self._proc = None
                    proc = self._get_proc()
                    proc.stdin.write(request.encode() + b"\n")
                    proc.stdin.flush()

                # Piper writes one JSON line to stdout when synthesis is complete.
                proc.stdout.readline()

            audio, sr = sf.read(tmp_path, dtype="float32")
            if audio.ndim > 1:
                audio = audio[:, 0]
            return audio, sr
        except Exception as exc:
            log.error("TTS synthesis failed: %s", exc)
            return np.zeros(1, dtype=np.float32), 22050
        finally:
            Path(tmp_path).unlink(missing_ok=True)
