import logging
import queue
import threading
from pathlib import Path

import numpy as np
import openwakeword
import sounddevice as sd
from openwakeword.model import Model

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1280  # 80ms at 16kHz — openwakeword's expected frame size


class WakeWordDetector:
    def __init__(self, cfg: dict):
        self._device_index = cfg.get("input_device_index")
        self._threshold = cfg.get("threshold", 0.5)
        self._wake_word = cfg.get("wake_word", "hey jarvis")
        self._input_channel = cfg.get("input_channel", 0)

        model_path = cfg.get("model_path")
        model_files = list(Path(model_path).glob("*.onnx")) if model_path else []

        openwakeword.utils.download_models()

        if model_files:
            # Custom trained wake word model (.onnx file in models/wake_word/)
            self._model = Model(wakeword_models=[str(model_files[0])], inference_framework="onnx")
        else:
            # Built-in model — derive model name from wake_word string
            # e.g. "hey jarvis" → "hey_jarvis"
            ww_name = self._wake_word.lower().replace(" ", "_")
            self._model = Model(wakeword_models=[ww_name], inference_framework="onnx")

        log.info("Wake word detector loaded (threshold=%.2f)", self._threshold)

    def wait_for_wake_word(self) -> None:
        """Block until the configured wake word is detected."""
        detected = threading.Event()
        audio_q: queue.Queue[np.ndarray] = queue.Queue()

        def _audio_callback(indata, frames, time_info, status):
            if status:
                log.debug("Audio status: %s", status)
            # Extract the beamformed channel and convert to int16
            channel = min(self._input_channel, indata.shape[1] - 1)
            audio_q.put(indata[:, channel].copy())

        def _detect():
            buffer = np.array([], dtype=np.float32)
            while not detected.is_set():
                try:
                    chunk = audio_q.get(timeout=0.5)
                except queue.Empty:
                    continue
                buffer = np.concatenate([buffer, chunk])
                while len(buffer) >= CHUNK_SAMPLES:
                    frame = buffer[:CHUNK_SAMPLES]
                    buffer = buffer[CHUNK_SAMPLES:]
                    frame_int16 = (frame * 32767).astype(np.int16)
                    prediction = self._model.predict(frame_int16)
                    for label, score in prediction.items():
                        if score >= self._threshold:
                            log.debug("Wake word '%s' score: %.3f", label, score)
                            detected.set()

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=6,
            dtype="float32",
            device=self._device_index,
            blocksize=CHUNK_SAMPLES,
            callback=_audio_callback,
        ):
            _thread = threading.Thread(target=_detect, daemon=True)
            _thread.start()
            detected.wait()
            _thread.join(timeout=1.0)
