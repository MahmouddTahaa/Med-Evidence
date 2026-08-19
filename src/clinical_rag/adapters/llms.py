"""LLM client protocol + provider cascade for grounded generation.

Mirrors adapters/embedders.py: Protocol, dotenv keys, fail closed per provider,
no LangChain. Cascade skips missing key / import / runtime errors.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, runtime_checkable

from clinical_rag.config import GenerationSettings, get_settings
from clinical_rag.errors import QueryError


def _load_dotenv_keys() -> None:
    env_path = Path(".env")
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(env_path, override=False)


@runtime_checkable
class LlmClient(Protocol):
    name: str

    def complete(self, messages: list[dict], *, json_mode: bool = False) -> str: ...

    def stream(self, messages: list[dict]) -> Iterator[str]: ...


class CascadeClient:
    """Try providers in order; last resort raises QueryError if none succeed."""

    def __init__(self, clients: list[LlmClient]) -> None:
        self._clients = list(clients)
        self.last_provider: str = ""

    @property
    def name(self) -> str:
        return self.last_provider or "cascade"

    @property
    def providers(self) -> list[str]:
        return [c.name for c in self._clients]

    def complete(self, messages: list[dict], *, json_mode: bool = False) -> str:
        errors: list[str] = []
        for client in self._clients:
            try:
                text = client.complete(messages, json_mode=json_mode)
                self.last_provider = client.name
                return text
            except Exception as exc:  # noqa: BLE001 — cascade next
                errors.append(f"{client.name}: {exc}")
        raise QueryError(
            "All LLM providers failed. " + "; ".join(errors) if errors else "No LLM providers configured."
        )

    def stream(self, messages: list[dict]) -> Iterator[str]:
        errors: list[str] = []
        for client in self._clients:
            try:
                started = False
                for chunk in client.stream(messages):
                    if not started:
                        self.last_provider = client.name
                        started = True
                    yield chunk
                if started:
                    return
                # Empty stream still counts as success for that provider.
                self.last_provider = client.name
                return
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{client.name}: {exc}")
        raise QueryError(
            "All LLM providers failed. " + "; ".join(errors) if errors else "No LLM providers configured."
        )


class GeminiClient:
    name = "gemini"

    def __init__(self, model: str, api_key: str) -> None:
        try:
            from google import genai  # type: ignore
        except ImportError as exc:
            raise QueryError(
                "Gemini requires optional package. uv sync --extra generate"
            ) from exc
        self._model = model
        self._client = genai.Client(api_key=api_key)

    def complete(self, messages: list[dict], *, json_mode: bool = False) -> str:
        from google.genai import types  # type: ignore

        system = "\n\n".join(
            str(m.get("content", "")) for m in messages if m.get("role") == "system"
        )
        user_parts = [
            str(m.get("content", ""))
            for m in messages
            if m.get("role") in ("user", "assistant")
        ]
        prompt = "\n\n".join(user_parts)
        config_kwargs: dict = {}
        if system:
            config_kwargs["system_instruction"] = system
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"
        resp = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(**config_kwargs) if config_kwargs else None,
        )
        text = getattr(resp, "text", None) or ""
        if not text:
            raise QueryError("Gemini returned empty response")
        return text

    def stream(self, messages: list[dict]) -> Iterator[str]:
        from google.genai import types  # type: ignore

        system = "\n\n".join(
            str(m.get("content", "")) for m in messages if m.get("role") == "system"
        )
        user_parts = [
            str(m.get("content", ""))
            for m in messages
            if m.get("role") in ("user", "assistant")
        ]
        prompt = "\n\n".join(user_parts)
        config_kwargs: dict = {}
        if system:
            config_kwargs["system_instruction"] = system
        stream = self._client.models.generate_content_stream(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(**config_kwargs) if config_kwargs else None,
        )
        for chunk in stream:
            text = getattr(chunk, "text", None) or ""
            if text:
                yield text


class OpenAIChatClient:
    def __init__(self, *, name: str, model: str, api_key: str, base_url: str | None = None) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise QueryError(
                "OpenAI-compatible client requires optional package. uv sync --extra generate"
            ) from exc
        self.name = name
        self._model = model
        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)

    def complete(self, messages: list[dict], *, json_mode: bool = False) -> str:
        kwargs: dict = {"model": self._model, "messages": messages}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(**kwargs)
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            raise QueryError(f"{self.name} returned empty response")
        return text

    def stream(self, messages: list[dict]) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=True,
        )
        for event in stream:
            delta = event.choices[0].delta if event.choices else None
            if delta and delta.content:
                yield delta.content


class GroqClient:
    name = "groq"

    def __init__(self, model: str, api_key: str) -> None:
        try:
            from groq import Groq  # type: ignore
        except ImportError as exc:
            raise QueryError("Groq requires optional package. uv sync --extra generate") from exc
        self._model = model
        self._client = Groq(api_key=api_key)

    def complete(self, messages: list[dict], *, json_mode: bool = False) -> str:
        kwargs: dict = {"model": self._model, "messages": messages}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(**kwargs)
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            raise QueryError("Groq returned empty response")
        return text

    def stream(self, messages: list[dict]) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=True,
        )
        for event in stream:
            delta = event.choices[0].delta if event.choices else None
            if delta and delta.content:
                yield delta.content


class AnthropicClient:
    name = "anthropic"

    def __init__(self, model: str, api_key: str) -> None:
        try:
            import anthropic  # type: ignore
        except ImportError as exc:
            raise QueryError(
                "Anthropic requires optional package. uv sync --extra generate"
            ) from exc
        self._model = model
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(self, messages: list[dict], *, json_mode: bool = False) -> str:
        system = "\n\n".join(
            str(m.get("content", "")) for m in messages if m.get("role") == "system"
        )
        chat = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m.get("role") in ("user", "assistant")
        ]
        if json_mode and chat:
            chat = list(chat)
            chat[-1] = {
                **chat[-1],
                "content": str(chat[-1]["content"])
                + "\n\nRespond with valid JSON only.",
            }
        kwargs: dict = {
            "model": self._model,
            "max_tokens": 4096,
            "messages": chat,
        }
        if system:
            kwargs["system"] = system
        resp = self._client.messages.create(**kwargs)
        parts = []
        for block in resp.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        text = "".join(parts).strip()
        if not text:
            raise QueryError("Anthropic returned empty response")
        return text

    def stream(self, messages: list[dict]) -> Iterator[str]:
        system = "\n\n".join(
            str(m.get("content", "")) for m in messages if m.get("role") == "system"
        )
        chat = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m.get("role") in ("user", "assistant")
        ]
        kwargs: dict = {
            "model": self._model,
            "max_tokens": 4096,
            "messages": chat,
        }
        if system:
            kwargs["system"] = system
        with self._client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                if text:
                    yield text


def _try_local_reachable(base_url: str) -> bool:
    try:
        import urllib.request

        root = base_url.rstrip("/")
        # OpenAI-compatible /models; Ollama also serves this under /v1.
        url = f"{root}/models" if root.endswith("/v1") else f"{root}/v1/models"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=1.5) as resp:  # noqa: S310
            return 200 <= getattr(resp, "status", 200) < 300
    except Exception:
        return False


def build_cascade(
    settings: GenerationSettings | None = None,
    *,
    include_local: bool = True,
) -> CascadeClient:
    """Build provider cascade; skip missing keys / imports / unreachable local."""
    _load_dotenv_keys()
    cfg = settings or get_settings().generation
    clients: list[LlmClient] = []

    gemini_key = (
        os.environ.get("GEMINI_API_KEY", "").strip()
        or os.environ.get("GOOGLE_API_KEY", "").strip()
    )
    if gemini_key:
        try:
            clients.append(GeminiClient(cfg.gemini_model, gemini_key))
        except QueryError:
            pass

    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if openai_key:
        try:
            clients.append(
                OpenAIChatClient(
                    name="openai",
                    model=cfg.openai_model,
                    api_key=openai_key,
                    base_url=os.environ.get("OPENAI_BASE_URL", "").strip() or None,
                )
            )
        except QueryError:
            pass

    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    if groq_key:
        try:
            clients.append(GroqClient(cfg.groq_model, groq_key))
        except QueryError:
            pass

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if anthropic_key:
        try:
            clients.append(AnthropicClient(cfg.anthropic_model, anthropic_key))
        except QueryError:
            pass

    if include_local:
        base = (
            os.environ.get("LOCAL_LLM_BASE_URL", "").strip() or cfg.local_base_url
        )
        model = os.environ.get("LOCAL_LLM_MODEL", "").strip() or cfg.local_model
        if _try_local_reachable(base):
            try:
                clients.append(
                    OpenAIChatClient(
                        name="local",
                        model=model,
                        api_key=os.environ.get("LOCAL_LLM_API_KEY", "ollama").strip()
                        or "ollama",
                        base_url=base,
                    )
                )
            except QueryError:
                pass

    return CascadeClient(clients)


def parse_json_object(text: str) -> dict:
    """Best-effort JSON object extract from model output."""
    raw = text.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise QueryError("Model did not return a JSON object") from None
        data = json.loads(raw[start : end + 1])
    if not isinstance(data, dict):
        raise QueryError("Model JSON was not an object")
    return data
