"""
Universal Poison Armor - Interceptor Middleware SDK
===================================================
Provides transparent, client-side security middleware for LLM applications.
Ensures that all incoming documents, user inputs, tool outputs, and RAG context
are automatically sanitized before reaching the model context window, overcoming
the limitation of relying solely on voluntary agent tool-calling over stdio.

Supported Integrations:
1. `PoisonArmorClient`: Drop-in wrapper for OpenAI / LiteLLM Python SDK clients.
2. `wrap_openai(client)`: Function wrapper to patch an existing OpenAI client instance.
3. `sanitize_message_content(content)`: Standalone message content cleaner.
4. `filter_rag_chunks(chunks)`: Standalone batch RAG chunk cleaner with anomaly detection.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from .sanitizers import PoisonDefenseEngine

logger = logging.getLogger("UniversalPoisonArmor.Middleware")


class PoisonArmorMiddleware:
    """
    Transparent security middleware that enforces pre-model input sanitization,
    tracking pixel stripping, and cryptographic taint boundary framing.
    """

    def __init__(
        self,
        engine: Optional[PoisonDefenseEngine] = None,
        wrap_taint: bool = True,
        strict_anomaly_quarantine: bool = True,
    ) -> None:
        """
        Initialize middleware with a shared or custom PoisonDefenseEngine instance.

        Args:
            engine: Optional pre-configured PoisonDefenseEngine. Defaults to singleton.
            wrap_taint: If True, wraps sanitized external data in cryptographic delimiters.
            strict_anomaly_quarantine: If True, excludes critical dataset/RAG anomalies.
        """
        self.engine = engine or PoisonDefenseEngine()
        self.wrap_taint = wrap_taint
        self.strict_anomaly_quarantine = strict_anomaly_quarantine

    def sanitize_message_content(self, content: Union[str, List[Dict[str, Any]], Any]) -> Any:
        """
        Sanitizes standard or multimodal message content.

        Args:
            content: Raw message text string or list of content blocks.

        Returns:
            Sanitized message content with injections neutralized and pixels stripped.
        """
        if isinstance(content, str):
            # Strip XSS / tracking pixels first
            cleaned = self.engine.strip_markdown_xss(content)
            # Strip prompt injections and adversarial suffixes
            sanitized = self.engine.strip_injections(cleaned)
            if self.wrap_taint and ("[REDACTED_INJECTION_ATTEMPT]" in sanitized or "[ADVERSARIAL_SUFFIX" in sanitized):
                return self.engine.wrap_taint_boundary(sanitized, source="untrusted_input")
            return sanitized

        elif isinstance(content, list):
            sanitized_list = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_val = item.get("text", "")
                    cleaned_text = self.sanitize_message_content(text_val)
                    sanitized_list.append({**item, "text": cleaned_text})
                else:
                    sanitized_list.append(item)
            return sanitized_list

        return content

    def sanitize_messages(self, messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sanitize an array of chat messages in-place or return a sanitized copy.

        Args:
            messages: List of message dictionaries containing 'role' and 'content'.

        Returns:
            Sanitized list of messages safe for model consumption.
        """
        processed_messages = []
        for msg in messages:
            if not isinstance(msg, dict):
                processed_messages.append(msg)
                continue

            role = msg.get("role", "")
            content = msg.get("content", "")

            # User prompts, tool return values, and assistant inputs are sanitized
            if content:
                sanitized_content = self.sanitize_message_content(content)
                processed_messages.append({**msg, "content": sanitized_content})
            else:
                processed_messages.append(msg)

        return processed_messages

    def filter_rag_chunks(self, chunks: List[str]) -> List[str]:
        """
        Scans a batch of retrieved RAG chunks for poisoned clusters/anomalies,
        quarantines contaminated chunks if strict mode is enabled, and returns
        the sanitized clean chunks.

        Args:
            chunks: List of raw retrieved text chunks.

        Returns:
            List of verified, sanitized RAG chunks.
        """
        if not chunks:
            return []

        # Run semantic anomaly scan across chunks
        if len(chunks) >= 3:
            anomalies = self.engine.detect_semantic_anomalies(chunks)
            quarantined_indices = {
                a["index"]
                for a in anomalies
                if a.get("severity") in ("CRITICAL", "HIGH")
            } if self.strict_anomaly_quarantine else set()
        else:
            quarantined_indices = set()

        safe_chunks = []
        for idx, chunk in enumerate(chunks):
            if idx in quarantined_indices:
                logger.warning("Quarantined anomalous RAG chunk at index %d", idx)
                continue
            cleaned = self.engine.strip_markdown_xss(chunk)
            sanitized = self.engine.strip_injections(cleaned)
            if self.wrap_taint:
                sanitized = self.engine.wrap_taint_boundary(sanitized, source=f"rag_chunk_{idx}")
            safe_chunks.append(sanitized)

        return safe_chunks


class PoisonArmorClient:
    """
    Transparent proxy wrapper for OpenAI-compatible clients.
    Intercepts `client.chat.completions.create` to automatically sanitize
    messages before transmission.
    """

    def __init__(self, client: Any, middleware: Optional[PoisonArmorMiddleware] = None) -> None:
        self._client = client
        self.middleware = middleware or PoisonArmorMiddleware()

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._client, name)
        if name == "chat":
            return _ChatProxy(attr, self.middleware)
        return attr


class _ChatProxy:
    def __init__(self, chat_obj: Any, middleware: PoisonArmorMiddleware) -> None:
        self._chat_obj = chat_obj
        self.middleware = middleware

    @property
    def completions(self) -> Any:
        return _CompletionsProxy(self._chat_obj.completions, self.middleware)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat_obj, name)


class _CompletionsProxy:
    def __init__(self, completions_obj: Any, middleware: PoisonArmorMiddleware) -> None:
        self._completions_obj = completions_obj
        self.middleware = middleware

    def create(self, *args: Any, **kwargs: Any) -> Any:
        if "messages" in kwargs and isinstance(kwargs["messages"], list):
            kwargs["messages"] = self.middleware.sanitize_messages(kwargs["messages"])
        return self._completions_obj.create(*args, **kwargs)

    async def acreate(self, *args: Any, **kwargs: Any) -> Any:
        if "messages" in kwargs and isinstance(kwargs["messages"], list):
            kwargs["messages"] = self.middleware.sanitize_messages(kwargs["messages"])
        return await self._completions_obj.acreate(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._completions_obj, name)


def wrap_openai(client: Any, wrap_taint: bool = False) -> PoisonArmorClient:
    """
    Convenience function to wrap an existing OpenAI or LiteLLM client with PoisonArmorMiddleware.

    Example:
    ```python
    from openai import OpenAI
    from src.middleware import wrap_openai

    client = wrap_openai(OpenAI())
    # All messages sent through client.chat.completions.create are automatically sanitized!
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": untrusted_user_file}],
    )
    ```
    """
    middleware = PoisonArmorMiddleware(wrap_taint=wrap_taint)
    return PoisonArmorClient(client, middleware=middleware)


__all__ = [
    "PoisonArmorMiddleware",
    "PoisonArmorClient",
    "wrap_openai",
]
