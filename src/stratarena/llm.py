"""Language-model backends behind one small interface.

- ``OllamaLLM``: a local model served by Ollama (the default; no data leaves the machine).
- ``AnthropicLLM``: Claude through the official ``anthropic`` SDK (optional extra
  ``stratarena[anthropic]``; credentials from the environment, e.g. ``ANTHROPIC_API_KEY``).
- ``ScriptedLLM``: canned answers, for tests and for running without any model.
- ``ExternalLLM``: answers written by someone else (a person, or a separate agent session):
  unanswered prompts go to a queue directory and the run stops; once the answers are
  in place, rerunning replays everything from the cache and continues.
- ``CachedLLM``: wraps any of them and stores every answer on disk, keyed by a hash of the
  request, so a run can be replayed exactly (LLM sampling is otherwise not reproducible).

No keys are read from or written to the repository.
"""

import hashlib
import json
import os
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx

from stratarena.config import LLMConfig


class LLMError(RuntimeError):
    """The model could not produce an answer (network, refusal, bad response)."""


class LLM(Protocol):
    name: str
    """Provider and model, e.g. ``"ollama/llama3.2"``."""

    def complete(self, system: str, prompt: str) -> str: ...


@dataclass(frozen=True)
class Exchange:
    """One request/response pair, kept for the run report."""

    llm: str
    system: str
    prompt: str
    response: str
    cached: bool

    @property
    def key(self) -> str:
        return request_key(self.llm, self.system, self.prompt)


def request_key(name: str, system: str, prompt: str) -> str:
    payload = json.dumps([name, system, prompt], ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


class OllamaLLM:
    """Chat completion against an Ollama server (``OLLAMA_HOST`` or localhost)."""

    def __init__(
        self,
        model: str = "llama3.2",
        host: str | None = None,
        temperature: float = 0.2,
        seed: int = 0,
        timeout_s: float = 300.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.name = f"ollama/{model}"
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        if not self.host.startswith("http"):
            self.host = f"http://{self.host}"
        self.options = {"temperature": temperature, "seed": seed}
        self.client = client or httpx.Client(timeout=timeout_s)

    def complete(self, system: str, prompt: str) -> str:
        body = {
            "model": self.model,
            "stream": False,
            "options": self.options,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        try:
            response = self.client.post(f"{self.host}/api/chat", json=body)
            response.raise_for_status()
            content = response.json()["message"]["content"]
        except (httpx.HTTPError, KeyError, ValueError) as e:
            raise LLMError(f"{self.name}: {e}") from e
        if not isinstance(content, str) or not content.strip():
            raise LLMError(f"{self.name}: empty response")
        return content


class AnthropicLLM:
    """Claude via the official SDK, with server-side refusal fallbacks enabled."""

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        effort: str = "medium",
        max_tokens: int = 16000,
        client: Any = None,
    ) -> None:
        self.model = model
        self.name = f"anthropic/{model}"
        self.effort = effort
        self.max_tokens = max_tokens
        if client is None:
            try:
                import anthropic
            except ImportError as e:  # pragma: no cover - depends on the optional extra
                raise LLMError(
                    "install the optional extra: pip install 'stratarena[anthropic]'"
                ) from e
            client = anthropic.Anthropic()
        self.client = client

    def complete(self, system: str, prompt: str) -> str:
        import anthropic

        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_config={"effort": self.effort},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.RateLimitError as e:
            raise LLMError(f"{self.name}: rate limited") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"{self.name}: HTTP {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise LLMError(f"{self.name}: connection error") from e
        except TypeError as e:  # raised by the SDK when no credentials can be resolved
            raise LLMError(f"{self.name}: no credentials ({e}); set ANTHROPIC_API_KEY") from e
        if response.stop_reason == "refusal":
            raise LLMError(f"{self.name}: request declined")
        text = "".join(block.text for block in response.content if block.type == "text")
        if not text.strip():
            raise LLMError(f"{self.name}: empty response")
        return text


class AnswerPending(Exception):  # noqa: N818 - a signal, not an error
    """Raised by ``ExternalLLM`` when a prompt has no answer yet. Stops the run."""

    def __init__(self, queue: Path) -> None:
        super().__init__(f"prompts are waiting for answers in {queue}")
        self.queue = queue


class ExternalLLM:
    """Answers come from files: ``<queue>/answers/<key>.txt`` for ``<queue>/<key>.json``.

    The key is the same request hash as the cache, so a prompt is written once and its
    answer is used once; afterwards ``CachedLLM`` replays it.
    """

    def __init__(self, model: str, queue_dir: Path) -> None:
        self.name = f"external/{model}"
        self.queue = queue_dir
        (self.queue / "answers").mkdir(parents=True, exist_ok=True)

    def complete(self, system: str, prompt: str) -> str:
        key = request_key(self.name, system, prompt)
        answer = self.queue / "answers" / f"{key}.txt"
        if answer.exists():
            return answer.read_text(encoding="utf-8")
        request = {"key": key, "llm": self.name, "system": system, "prompt": prompt}
        (self.queue / f"{key}.json").write_text(
            json.dumps(request, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        raise AnswerPending(self.queue)

    def pending(self) -> list[Path]:
        """Queued prompts that still have no answer."""
        return sorted(
            p
            for p in self.queue.glob("*.json")
            if not (self.queue / "answers" / f"{p.stem}.txt").exists()
        )


class ScriptedLLM:
    """Returns canned answers: a list consumed in order, or a function of the request."""

    def __init__(
        self,
        responses: Sequence[str] | Callable[[str, str], str],
        name: str = "scripted",
    ) -> None:
        self.name = name
        self._fn = responses if callable(responses) else None
        self._queue = [] if callable(responses) else list(responses)
        self._lock = threading.Lock()
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, prompt: str) -> str:
        with self._lock:
            self.calls.append((system, prompt))
            if self._fn is not None:
                return self._fn(system, prompt)
            if not self._queue:
                raise LLMError("scripted LLM has no answers left")
            return self._queue.pop(0)


@dataclass
class CachedLLM:
    """Disk cache in front of another model; records every exchange of the run."""

    inner: LLM
    directory: Path
    exchanges: list[Exchange] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.name = self.inner.name
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def complete(self, system: str, prompt: str) -> str:
        key = request_key(self.name, system, prompt)
        path = self.directory / f"{key}.json"
        if path.exists():
            response = json.loads(path.read_text(encoding="utf-8"))["response"]
            cached = True
        else:
            response = self.inner.complete(system, prompt)
            record = {"llm": self.name, "system": system, "prompt": prompt, "response": response}
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(path)
            cached = False
        with self._lock:
            self.exchanges.append(Exchange(self.name, system, prompt, response, cached))
        return str(response)


def make_llm(config: LLMConfig) -> CachedLLM | None:
    """Build the configured model (wrapped in the disk cache), or ``None`` for no LLM."""
    inner: LLM
    if config.provider == "none":
        return None
    if config.provider == "external":
        if not config.model:
            raise ValueError("the external provider needs a model name")
        inner = ExternalLLM(config.model, Path(config.queue_dir))
    elif config.provider == "ollama":
        inner = OllamaLLM(
            model=config.model or "llama3.2",
            host=config.host,
            temperature=config.temperature,
            seed=config.seed,
        )
    else:
        inner = AnthropicLLM(model=config.model or "claude-opus-5-5", effort=config.effort)
    return CachedLLM(inner, Path(config.cache_dir))
