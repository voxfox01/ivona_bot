"""
services/llm/conversation.py

LangGraph-based conversation memory for ivona_bot.
Maintains per-session message history using MemorySaver checkpointing.
Each wake-word activation gets a fresh thread_id via reset_session().
"""
import logging
import queue
import re
import threading
import uuid
from typing import Iterator

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

log = logging.getLogger(__name__)

_SENT_RE  = re.compile(r'[.!?][\s"\')\]]*(?=\s|$)')
_IM_START = "<|im_start|>"
_IM_END   = "<|im_end|>"


class ConversationGraph:
    """Wraps a llama_cpp.Llama instance in a LangGraph StateGraph with MemorySaver.

    Sentence streaming is preserved for real-time TTS: the LangGraph node pushes
    complete sentences to a queue.Queue as it generates tokens. The queue is passed
    through config["configurable"] so it travels to the node without being persisted
    by MemorySaver.
    """

    def __init__(self, llama, cfg: dict):
        self._llama         = llama
        self._system_prompt = cfg.get("system_prompt", "You are a helpful assistant.")
        self._chat_format   = cfg.get("chat_format", "chatml")
        self._max_tokens    = cfg.get("max_tokens", 512)
        self._temperature   = cfg.get("temperature", 0.7)
        self._max_history   = cfg.get("max_history_turns", 5)

        self._memory = MemorySaver()
        builder = StateGraph(MessagesState)
        builder.add_node("generate", self._generate_node)
        builder.add_edge(START, "generate")
        self._graph = builder.compile(checkpointer=self._memory)

        self._thread_id = str(uuid.uuid4())
        log.info("ConversationGraph ready (format=%s, max_history=%d)",
                 self._chat_format, self._max_history)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset_session(self) -> None:
        """Start a fresh conversation thread. Call at each wake-word activation."""
        self._thread_id = str(uuid.uuid4())
        log.info("Conversation memory reset (new session).")

    def stream_sentences(self, user_text: str) -> Iterator[str]:
        """Add user turn to history, generate response, yield sentences for TTS.

        Runs graph.invoke() in a background thread so we can yield from a queue
        while LangGraph executes synchronously to completion and checkpoints state.
        """
        sentence_q: queue.Queue[str | None] = queue.Queue()
        config = {"configurable": {
            "thread_id": self._thread_id,
            "sentence_q": sentence_q,
        }}

        def _run():
            try:
                self._graph.invoke(
                    {"messages": [HumanMessage(content=user_text)]},
                    config=config,
                )
            except Exception as exc:
                log.error("ConversationGraph invoke error: %s", exc, exc_info=True)
            finally:
                sentence_q.put(None)  # sentinel — always fires even on error

        threading.Thread(target=_run, daemon=True).start()

        while True:
            sentence = sentence_q.get()
            if sentence is None:
                break
            yield sentence

    def generate(self, user_text: str) -> str:
        """Non-streaming generation — collects all sentences into one string."""
        return " ".join(self.stream_sentences(user_text))

    # ------------------------------------------------------------------
    # LangGraph node
    # ------------------------------------------------------------------

    def _generate_node(self, state: MessagesState, config: RunnableConfig) -> dict:
        """LangGraph node: build multi-turn prompt, stream tokens, return AIMessage.

        Two-arg signature (state, config) lets LangGraph pass the full RunnableConfig.
        The sentence_q is retrieved from config["configurable"] and is never persisted.
        """
        sentence_q: queue.Queue = config["configurable"]["sentence_q"]
        prompt = self._build_prompt(state["messages"])
        full_response = self._stream_to_queue(prompt, sentence_q)
        return {"messages": [AIMessage(content=full_response)]}

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self, messages: list) -> str:
        """Build a flat prompt string from the full MessagesState history.

        messages[-1] is always the HumanMessage just added by graph.invoke().
        prior = messages[:-1] contains all previous turns.
        """
        prior   = messages[:-1]
        current = messages[-1].content

        # Sliding window: keep last max_history_turns human+AI pairs
        max_msgs = self._max_history * 2
        if len(prior) > max_msgs:
            log.debug("Trimming conversation history to %d messages.", max_msgs)
            prior = prior[-max_msgs:]
        # Always start the window on a HumanMessage so system prompt injection is correct
        while prior and not isinstance(prior[0], HumanMessage):
            prior = prior[1:]

        if self._chat_format == "gemma":
            return self._build_gemma(prior, current)
        return self._build_chatml(prior, current)

    def _build_chatml(self, prior: list, current_user: str) -> str:
        parts = [f"{_IM_START}system\n{self._system_prompt.strip()}{_IM_END}\n"]
        for msg in prior:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            parts.append(f"{_IM_START}{role}\n{msg.content}{_IM_END}\n")
        parts.append(f"{_IM_START}user\n{current_user}{_IM_END}\n")
        parts.append(f"{_IM_START}assistant\n")
        return "".join(parts)

    def _build_gemma(self, prior: list, current_user: str) -> str:
        """Gemma instruct format — system prompt folded into the first user turn only."""
        parts = []
        system_injected = False
        for msg in prior:
            if isinstance(msg, HumanMessage):
                if not system_injected:
                    parts.append(
                        f"<start_of_turn>user\n"
                        f"{self._system_prompt.strip()}\n\n"
                        f"{msg.content}<end_of_turn>\n"
                    )
                    system_injected = True
                else:
                    parts.append(f"<start_of_turn>user\n{msg.content}<end_of_turn>\n")
            elif isinstance(msg, AIMessage):
                parts.append(f"<start_of_turn>model\n{msg.content}<end_of_turn>\n")

        if not system_injected:
            # No prior history — fold system prompt into current user turn
            parts.append(
                f"<start_of_turn>user\n"
                f"{self._system_prompt.strip()}\n\n"
                f"{current_user}<end_of_turn>\n"
            )
        else:
            parts.append(f"<start_of_turn>user\n{current_user}<end_of_turn>\n")
        parts.append("<start_of_turn>model\n")
        return "".join(parts)

    # ------------------------------------------------------------------
    # Streaming helper
    # ------------------------------------------------------------------

    def _stop_tokens(self) -> list[str]:
        if self._chat_format == "gemma":
            return ["<end_of_turn>", "<start_of_turn>"]
        return [_IM_END, _IM_START]

    def _stream_to_queue(self, prompt: str, sentence_q: queue.Queue) -> str:
        """Call llama with stream=True, split into sentences, push each to queue.

        Returns the full response text for AIMessage construction.
        Does NOT push the sentinel — that is the caller's responsibility.
        """
        stream = self._llama(
            prompt,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            stop=self._stop_tokens(),
            echo=False,
            stream=True,
        )
        buffer    = ""
        collected: list[str] = []

        for chunk in stream:
            token  = chunk["choices"][0]["text"]
            buffer += token
            while True:
                m = _SENT_RE.search(buffer)
                if not m:
                    break
                sentence = buffer[:m.end()].strip()
                buffer   = buffer[m.end():]
                if sentence:
                    collected.append(sentence)
                    sentence_q.put(sentence)

        if buffer.strip():
            collected.append(buffer.strip())
            sentence_q.put(buffer.strip())

        return " ".join(collected)
