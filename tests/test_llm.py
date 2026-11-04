import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from stratarena.config import LLMConfig
from stratarena.llm import (
    AnthropicLLM,
    CachedLLM,
    LLMError,
    OllamaLLM,
    ScriptedLLM,
    make_llm,
)


def test_ollama_sends_chat_request() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": "hi"}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    llm = OllamaLLM(model="m", host="localhost:1234", seed=7, client=client)
    assert llm.complete("sys", "prompt") == "hi"
    assert seen["url"] == "http://localhost:1234/api/chat"
    assert seen["body"]["model"] == "m"
    assert seen["body"]["options"]["seed"] == 7
    assert [m["role"] for m in seen["body"]["messages"]] == ["system", "user"]


@pytest.mark.parametrize(
    "response",
    [httpx.Response(500), httpx.Response(200, json={"message": {"content": "  "}})],
)
def test_ollama_errors(response: httpx.Response) -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda _: response))
    with pytest.raises(LLMError):
        OllamaLLM(client=client).complete("s", "p")


def test_ollama_host_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "http://box:11434/")
    assert OllamaLLM().host == "http://box:11434"


class FakeMessages:
    def __init__(self, stop_reason: str = "end_turn") -> None:
        self.kwargs: dict[str, Any] = {}
        self.stop_reason = stop_reason

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        blocks = [
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text="answer"),
        ]
        return SimpleNamespace(stop_reason=self.stop_reason, content=blocks)


def test_anthropic_request_shape() -> None:
    messages = FakeMessages()
    client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    llm = AnthropicLLM(client=client)
    assert llm.name == "anthropic/claude-opus-5-5"
    assert llm.complete("sys", "prompt") == "answer"
    kw = messages.kwargs
    assert kw["model"] == "claude-opus-5-5"
    assert kw["system"] == "sys"
    assert kw["fallbacks"] == "default"
    assert kw["output_config"] == {"effort": "medium"}
    assert "temperature" not in kw


def test_anthropic_refusal_is_an_error() -> None:
    client = SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages("refusal")))
    with pytest.raises(LLMError, match="declined"):
        AnthropicLLM(client=client).complete("s", "p")


def test_scripted_llm() -> None:
    llm = ScriptedLLM(["a", "b"])
    assert [llm.complete("s", "1"), llm.complete("s", "2")] == ["a", "b"]
    with pytest.raises(LLMError):
        llm.complete("s", "3")
    echo = ScriptedLLM(lambda s, p: p.upper())
    assert echo.complete("s", "x") == "X"
    assert echo.calls == [("s", "x")]


def test_cache_replays_without_calling_the_model(tmp_path: Path) -> None:
    inner = ScriptedLLM(["first"])
    cached = CachedLLM(inner, tmp_path)
    assert cached.complete("s", "p") == "first"
    assert CachedLLM(ScriptedLLM([]), tmp_path).complete("s", "p") == "first"
    assert [e.cached for e in cached.exchanges] == [False]
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_make_llm(tmp_path: Path) -> None:
    assert make_llm(LLMConfig()) is None
    llm = make_llm(LLMConfig(provider="ollama", model="x", cache_dir=str(tmp_path)))
    assert llm is not None and llm.name == "ollama/x"


def test_anthropic_missing_credentials_is_an_llm_error() -> None:
    class NoAuth:
        def create(self, **kwargs: Any) -> Any:
            raise TypeError("Could not resolve authentication method")

    client = SimpleNamespace(beta=SimpleNamespace(messages=NoAuth()))
    with pytest.raises(LLMError, match="no credentials"):
        AnthropicLLM(client=client).complete("s", "p")


def test_external_llm_queues_prompts_and_replays_answers(tmp_path: Path) -> None:
    from stratarena.llm import AnswerPending, ExternalLLM

    queue = tmp_path / "queue"
    llm = CachedLLM(ExternalLLM("someone", queue), tmp_path / "cache")
    with pytest.raises(AnswerPending):
        llm.complete("sys", "prompt")
    (pending,) = ExternalLLM("someone", queue).pending()
    request = json.loads(pending.read_text())
    assert request["prompt"] == "prompt" and request["llm"] == "external/someone"
    (queue / "answers" / f"{request['key']}.txt").write_text("the answer")
    assert ExternalLLM("someone", queue).pending() == []
    assert llm.complete("sys", "prompt") == "the answer"
    assert llm.exchanges[-1].cached is False
    replay = CachedLLM(ExternalLLM("someone", tmp_path / "other"), tmp_path / "cache")
    assert replay.complete("sys", "prompt") == "the answer"


def test_make_external_llm_needs_a_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        make_llm(LLMConfig(provider="external", queue_dir=str(tmp_path)))
