import logging
import os
import subprocess
import tempfile
from pathlib import Path

import sounddevice as sd
import soundfile as sf

log = logging.getLogger(__name__)

PIPER_SAMPLE_RATE = 22050


class Speaker:
    def __init__(self, cfg: dict):
        self._binary = Path(cfg["binary_path"]).resolve()
        self._model = Path(cfg["model_path"]).resolve()
        self._rate = cfg.get("speaking_rate", 1.0)

        # Piper needs its bundled .so files on LD_LIBRARY_PATH
        self._lib_dir = str(self._binary.parent)

        if not self._binary.exists():
            raise FileNotFoundError(f"Piper binary not found: {self._binary}")
        if not self._model.exists():
            raise FileNotFoundError(f"Piper voice model not found: {self._model}")

        log.info("TTS ready (model=%s)", self._model.name)

    def speak(self, text: str) -> None:
        """Synthesise text via Piper and play through the default audio output."""
        if not text.strip():
            return

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = self._lib_dir + ":" + env.get("LD_LIBRARY_PATH", "")

        try:
            subprocess.run(
                [
                    str(self._binary),
                    "--model", str(self._model),
                    "--output_file", tmp_path,
                    "--length_scale", str(1.0 / self._rate),
                ],
                input=text.encode(),
                env=env,
                check=True,
                capture_output=True,
            )
            audio, sr = sf.read(tmp_path, dtype="float32")
            sd.play(audio, samplerate=sr)
            sd.wait()
        finally:
            Path(tmp_path).unlink(missing_ok=True)
