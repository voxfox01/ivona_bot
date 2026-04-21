import logging
import subprocess
import threading
from pathlib import Path

import numpy as np
from openwakeword.model import Model

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1280  # 80ms at 16kHz — openwakeword's expected frame size
BYTES_PER_SAMPLE = 2  # s16le


class WakeWordDetector:
    def __init__(self, cfg: dict):
        self._threshold = cfg.get("threshold", 0.5)
        self._wake_word = cfg.get("wake_word", "hey jarvis")
        self._pulse_source = cfg.get("pulse_source")  # None = PulseAudio default

        model_path = cfg.get("model_path")
        model_files = list(Path(model_path).glob("*.onnx")) if model_path else []

        if model_files:
            self._model = Model(wakeword_models=[str(model_files[0])], inference_framework="onnx")
        else:
            # Built-in models (hey_jarvis, alexa, etc.) are bundled with the
            # openwakeword package — no network access needed at runtime.
            ww_name = self._wake_word.lower().replace(" ", "_")
            self._model = Model(wakeword_models=[ww_name], inference_framework="onnx")

        log.info("Wake word detector loaded (threshold=%.2f)", self._threshold)

    def wait_for_wake_word(self) -> None:
        """Block until the configured wake word is detected, reading audio via PulseAudio."""
        detected = threading.Event()

        cmd = ["parecord", "--channels=1", f"--rate={SAMPLE_RATE}", "--format=s16le", "--raw"]
        if self._pulse_source:
            cmd += [f"--device={self._pulse_source}"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

        def _detect():
            buffer = np.array([], dtype=np.float32)
            chunk_bytes = CHUNK_SAMPLES * BYTES_PER_SAMPLE
            try:
                while not detected.is_set():
                    raw = proc.stdout.read(chunk_bytes)
                    if not raw:
                        break
                    frame_int16 = np.frombuffer(raw, dtype=np.int16)
                    chunk_f32 = frame_int16.astype(np.float32) / 32768.0
                    buffer = np.concatenate([buffer, chunk_f32])
                    while len(buffer) >= CHUNK_SAMPLES:
                        frame = buffer[:CHUNK_SAMPLES]
                        buffer = buffer[CHUNK_SAMPLES:]
                        frame_int16_oww = (frame * 32767).astype(np.int16)
                        prediction = self._model.predict(frame_int16_oww)
                        for label, score in prediction.items():
                            if score >= self._threshold:
                                log.debug("Wake word '%s' score: %.3f", label, score)
                                detected.set()

                # Flush ~1s of silence through the model to reset internal state
                # so it doesn't immediately re-trigger on the next call.
                flush_frames = SAMPLE_RATE // CHUNK_SAMPLES
                silence = np.zeros(CHUNK_SAMPLES, dtype=np.int16)
                for _ in range(flush_frames):
                    self._model.predict(silence)
            finally:
                proc.terminate()

        _thread = threading.Thread(target=_detect, daemon=True)
        _thread.start()
        detected.wait()
        proc.terminate()
        _thread.join(timeout=2.0)
