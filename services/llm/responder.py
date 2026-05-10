import logging
import re
import subprocess
from pathlib import Path
from typing import Iterator

log = logging.getLogger(__name__)

_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"

_LLAMA_CLI = Path(__file__).resolve().parents[2] / "llama.cpp" / "build" / "bin" / "llama-cli"
_LLAMA_LIB = str(_LLAMA_CLI.parent)


def _build_prompt(system: str, user: str, chat_format: str = "chatml") -> str:
    if chat_format == "gemma":
        # Gemma instruct format — system prompt folded into first user turn
        return (
            f"<start_of_turn>user\n{system}\n\n{user}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
    # Default: ChatML (Qwen, Mistral, etc.)
    return (
        f"{_IM_START}system\n{system}{_IM_END}\n"
        f"{_IM_START}user\n{user}{_IM_END}\n"
        f"{_IM_START}assistant\n"
    )


class Responder:
    def __init__(self, cfg: dict):
        self._model_path = Path(cfg["model_path"]).resolve()
        self._n_gpu_layers = cfg.get("n_gpu_layers", 20)
        self._context_length = cfg.get("context_length", 2048)
        self._max_tokens = cfg.get("max_tokens", 512)
        self._temperature = cfg.get("temperature", 0.7)
        self._system_prompt = cfg.get("system_prompt", "You are a helpful assistant.")
        self._backend = cfg.get("backend", "llama_cli")
        self._chat_format = cfg.get("chat_format", "chatml")

        if self._backend == "llama_cpp_python":
            from llama_cpp import Llama
            log.info("Loading LLM (llama-cpp-python) from %s (format=%s)...",
                     self._model_path.name, self._chat_format)
            self._llama = Llama(
                model_path=str(self._model_path),
                n_gpu_layers=self._n_gpu_layers,
                n_ctx=self._context_length,
                verbose=False,
            )
            log.info("LLM loaded.")
            from services.llm.conversation import ConversationGraph
            self._conversation = ConversationGraph(self._llama, cfg)
        else:
            self._conversation = None
            if not _LLAMA_CLI.exists():
                raise FileNotFoundError(f"llama-cli not found at {_LLAMA_CLI}")
            log.info("LLM using llama-cli subprocess: %s", self._model_path.name)

    def reset_session(self) -> None:
        """Start a fresh conversation thread (call at each wake-word activation)."""
        if self._conversation is not None:
            self._conversation.reset_session()

    def generate(self, user_text: str) -> str:
        if self._conversation is not None:
            return self._conversation.generate(user_text)
        prompt = _build_prompt(self._system_prompt, user_text, self._chat_format)
        return self._generate_cli(prompt)

    def stream_sentences(self, user_text: str) -> Iterator[str]:
        """Stream the LLM response one sentence at a time.

        Yields each sentence as soon as it is complete so the caller can
        pipe it to TTS while the model continues generating the next one.
        Falls back to a single chunk for the llama_cli backend.
        """
        if self._conversation is not None:
            yield from self._conversation.stream_sentences(user_text)
        else:
            prompt = _build_prompt(self._system_prompt, user_text, self._chat_format)
            yield self._generate_cli(prompt)

    def _stop_tokens(self) -> list[str]:
        if self._chat_format == "gemma":
            return ["<end_of_turn>", "<start_of_turn>"]
        return [_IM_END, _IM_START]

    def _stream_sentences_python(self, prompt: str) -> Iterator[str]:
        stream = self._llama(
            prompt,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            stop=self._stop_tokens(),
            echo=False,
            stream=True,
        )
        buffer = ""
        for chunk in stream:
            token = chunk["choices"][0]["text"]
            buffer += token
            # Yield complete sentences as they accumulate
            while True:
                m = re.search(r'[.!?][\s"\')\]]*(?=\s|$)', buffer)
                if not m:
                    break
                sentence = buffer[:m.end()].strip()
                buffer = buffer[m.end():]
                if sentence:
                    yield sentence
        if buffer.strip():
            yield buffer.strip()

    def _generate_python(self, prompt: str) -> str:
        output = self._llama(
            prompt,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            stop=self._stop_tokens(),
            echo=False,
        )
        return output["choices"][0]["text"].strip()

    def _generate_cli(self, prompt: str) -> str:
        import os
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = _LLAMA_LIB + ":" + env.get("LD_LIBRARY_PATH", "")

        result = subprocess.run(
            [
                str(_LLAMA_CLI),
                "--model", str(self._model_path),
                "--n-gpu-layers", str(self._n_gpu_layers),
                "--ctx-size", str(self._context_length),
                "--temp", str(self._temperature),
                "--prompt", prompt,
                "-n", str(self._max_tokens),
                "--no-display-prompt",
                "--log-disable",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,  # suppress spinner and status; errors checked via returncode
            text=True,
            env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"llama-cli exited with code {result.returncode}. "
                "Check GPU memory — try reducing n_gpu_layers in config/settings.yaml."
            )
        text = result.stdout.strip()
        for stop in self._stop_tokens():
            if stop in text:
                text = text[:text.index(stop)]
        return text.strip()
