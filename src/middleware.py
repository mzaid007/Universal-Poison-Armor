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
try:
    from src.config import get_config
    from src.sanitizers import PoisonDefenseEngine
except ImportError:
    try:
        from .config import get_config
        from .sanitizers import PoisonDefenseEngine
    except ImportError:
        from sanitizers import PoisonDefenseEngine  # type: ignore
        def get_config():
            return None

logger = logging.getLogger("UniversalPoisonArmor.Middleware")


class PoisonArmorMiddleware:
    """
    Transparent security middleware that enforces pre-model input sanitization,
    tracking pixel stripping, and cryptographic taint boundary framing.
    Supports dry_run audit mode for non-blocking telemetry and risk scoring.
    """

    def __init__(
        self,
        engine: Optional[PoisonDefenseEngine] = None,
        wrap_taint: bool = True,
        strict_anomaly_quarantine: bool = True,
        dry_run: Optional[bool] = None,
    ) -> None:
        """
        Initialize middleware with a shared or custom PoisonDefenseEngine instance.

        Args:
            engine: Optional pre-configured PoisonDefenseEngine. Defaults to singleton.
            wrap_taint: If True, wraps sanitized external data in cryptographic delimiters.
            strict_anomaly_quarantine: If True, excludes critical dataset/RAG anomalies.
            dry_run: If True, operates in score-only audit mode without mutating text.
                     Defaults to POISON_ARMOR_DRY_RUN config if omitted.
        """
        self.engine = engine or PoisonDefenseEngine()
        self.wrap_taint = wrap_taint
        self.strict_anomaly_quarantine = strict_anomaly_quarantine
        cfg = get_config() if callable(get_config) else None
        if dry_run is not None:
            self.dry_run = dry_run
        elif cfg and hasattr(cfg, "dry_run"):
            self.dry_run = cfg.dry_run
        else:
            self.dry_run = False
        self.last_assessments: List[Dict[str, Any]] = []

    def sanitize_message_content(self, content: Union[str, List[Dict[str, Any]], Any]) -> Any:
        """
        Sanitizes standard or multimodal message content.

        Args:
            content: Raw message text string or list of content blocks.

        Returns:
            Sanitized message content with injections neutralized and pixels stripped.
        """
        if self.dry_run:
            if isinstance(content, str):
                eval_res = self.engine.evaluate_document(content)
                self.last_assessments.append(eval_res)
                if not eval_res["is_safe"]:
                    logger.warning(
                        "[DRY RUN] Threat detected in message content (score=%.2f): %s",
                        eval_res["threat_score"],
                        eval_res["threats"],
                    )
                return content
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        self.sanitize_message_content(item.get("text", ""))
                return content
            return content

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

        if self.dry_run:
            if len(chunks) >= 3:
                anomalies = self.engine.detect_semantic_anomalies(chunks)
                for a in anomalies:
                    logger.warning("[DRY RUN] Anomaly detected in RAG chunk #%d: %s", a.get("index"), a)
            for idx, chunk in enumerate(chunks):
                eval_res = self.engine.evaluate_document(chunk)
                if not eval_res["is_safe"]:
                    logger.warning("[DRY RUN] Threat detected in RAG chunk #%d: %s", idx, eval_res["threats"])
            return list(chunks)

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

    def __init__(
        self,
        client: Any,
        middleware: Optional[PoisonArmorMiddleware] = None,
        dry_run: Optional[bool] = None,
    ) -> None:
        self._client = client
        if middleware is not None:
            self.middleware = middleware
            if dry_run is not None:
                self.middleware.dry_run = dry_run
        else:
            self.middleware = PoisonArmorMiddleware(dry_run=dry_run)

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


def wrap_openai(
    client: Any,
    wrap_taint: bool = False,
    dry_run: Optional[bool] = None,
) -> PoisonArmorClient:
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
    middleware = PoisonArmorMiddleware(wrap_taint=wrap_taint, dry_run=dry_run)
    return PoisonArmorClient(client, middleware=middleware)


# ==============================================================================
# ECOSYSTEM & FRAMEWORK PLUGINS
# ==============================================================================

# Graceful optional imports for LangChain and LlamaIndex base classes
try:
    from langchain_core.callbacks.base import BaseCallbackHandler  # type: ignore
except ImportError:
    try:
        from langchain.callbacks.base import BaseCallbackHandler  # type: ignore
    except ImportError:
        class BaseCallbackHandler:  # type: ignore
            """Fallback BaseCallbackHandler when LangChain is not installed."""
            pass

try:
    from llama_index.core.postprocessor.types import BaseNodePostprocessor  # type: ignore
except ImportError:
    try:
        from llama_index.postprocessor.types import BaseNodePostprocessor  # type: ignore
    except ImportError:
        class BaseNodePostprocessor:  # type: ignore
            """Fallback BaseNodePostprocessor when LlamaIndex is not installed."""
            pass


class LangChainPoisonArmorCallback(BaseCallbackHandler):
    """
    LangChain Callback Handler that intercepts prompts before they reach the LLM,
    sanitizes tool inputs, and filters egress credential leaks in model responses.
    Supports dry_run audit mode.
    """

    def __init__(
        self,
        engine: Optional[PoisonDefenseEngine] = None,
        wrap_taint: bool = True,
        filter_egress: bool = True,
        dry_run: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.engine = engine or PoisonDefenseEngine()
        self.wrap_taint = wrap_taint
        self.filter_egress = filter_egress
        cfg = get_config() if callable(get_config) else None
        if dry_run is not None:
            self.dry_run = dry_run
        elif cfg and hasattr(cfg, "dry_run"):
            self.dry_run = cfg.dry_run
        else:
            self.dry_run = False

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Sanitizes raw LLM prompts in-place before execution."""
        for i, prompt in enumerate(prompts):
            if isinstance(prompt, str):
                if self.dry_run:
                    eval_res = self.engine.evaluate_document(prompt)
                    if not eval_res["is_safe"]:
                        logger.warning("[DRY RUN] Threat in LangChain prompt #%d: %s", i, eval_res["threats"])
                    continue
                cleaned = self.engine.strip_markdown_xss(prompt)
                sanitized = self.engine.strip_injections(cleaned)
                if self.wrap_taint and ("[REDACTED_INJECTION_ATTEMPT]" in sanitized or "[ADVERSARIAL_SUFFIX" in sanitized):
                    sanitized = self.engine.wrap_taint_boundary(sanitized, source="langchain_prompt")
                prompts[i] = sanitized

    def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List[List[Any]], **kwargs: Any
    ) -> None:
        """Sanitizes chat model message histories in-place."""
        for message_list in messages:
            for msg in message_list:
                content = getattr(msg, "content", None)
                if isinstance(content, str) and content:
                    if self.dry_run:
                        eval_res = self.engine.evaluate_document(content)
                        if not eval_res["is_safe"]:
                            logger.warning("[DRY RUN] Threat in LangChain chat message: %s", eval_res["threats"])
                        continue
                    cleaned = self.engine.strip_markdown_xss(content)
                    sanitized = self.engine.strip_injections(cleaned)
                    if self.wrap_taint and ("[REDACTED_INJECTION_ATTEMPT]" in sanitized or "[ADVERSARIAL_SUFFIX" in sanitized):
                        sanitized = self.engine.wrap_taint_boundary(sanitized, source="langchain_chat")
                    msg.content = sanitized

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Sanitizes inputs passed into agent tools."""
        if isinstance(input_str, str):
            if self.dry_run:
                eval_res = self.engine.evaluate_document(input_str)
                if not eval_res["is_safe"]:
                    logger.warning("[DRY RUN] Threat in LangChain tool input: %s", eval_res["threats"])
                return input_str
            cleaned = self.engine.strip_markdown_xss(input_str)
            return self.engine.strip_injections(cleaned)
        return input_str

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Filters egress credential leaks and tracking pixels on LLM completions."""
        if not self.filter_egress or not hasattr(response, "generations"):
            return

        for gen_list in response.generations:
            for gen in gen_list:
                text = getattr(gen, "text", None)
                if isinstance(text, str) and text:
                    if self.dry_run:
                        _, leaks = self.engine.filter_egress_leaks(text)
                        if leaks:
                            logger.warning("[DRY RUN] Egress secret leak in LangChain LLM response: %s", leaks)
                        continue
                    sanitized, _ = self.engine.filter_egress_leaks(text)
                    gen.text = sanitized
                elif hasattr(gen, "message") and hasattr(gen.message, "content"):
                    msg_content = gen.message.content
                    if isinstance(msg_content, str) and msg_content:
                        if self.dry_run:
                            _, leaks = self.engine.filter_egress_leaks(msg_content)
                            if leaks:
                                logger.warning("[DRY RUN] Egress secret leak in LangChain LLM response: %s", leaks)
                            continue
                        sanitized, _ = self.engine.filter_egress_leaks(msg_content)
                        gen.message.content = sanitized


class LlamaIndexPoisonArmorPostprocessor(BaseNodePostprocessor):
    """
    LlamaIndex NodePostprocessor that inspects retrieved nodes, removes prompt injections
    and tracking pixels, runs semantic anomaly detection across the batch, and quarantines poisoned nodes.
    Supports dry_run audit mode.
    """

    def __init__(
        self,
        engine: Optional[PoisonDefenseEngine] = None,
        strict_quarantine: bool = True,
        wrap_taint: bool = True,
        dry_run: Optional[bool] = None,
    ) -> None:
        self.engine = engine or PoisonDefenseEngine()
        self.strict_quarantine = strict_quarantine
        self.wrap_taint = wrap_taint
        cfg = get_config() if callable(get_config) else None
        if dry_run is not None:
            self.dry_run = dry_run
        elif cfg and hasattr(cfg, "dry_run"):
            self.dry_run = cfg.dry_run
        else:
            self.dry_run = False

    def postprocess_nodes(
        self, nodes: List[Any], query_bundle: Optional[Any] = None
    ) -> List[Any]:
        """
        Processes retrieved RAG nodes, quarantining poisoned clusters and sanitizing safe nodes.
        """
        if not nodes:
            return []

        # Extract text from each node
        node_texts = []
        for n in nodes:
            if hasattr(n, "node") and hasattr(n.node, "get_content"):
                node_texts.append(n.node.get_content())
            elif hasattr(n, "get_content"):
                node_texts.append(n.get_content())
            elif hasattr(n, "text"):
                node_texts.append(n.text)
            else:
                node_texts.append(str(n))

        if self.dry_run:
            if len(node_texts) >= 3:
                anomalies = self.engine.detect_semantic_anomalies(node_texts)
                for a in anomalies:
                    logger.warning("[DRY RUN] Anomaly detected in LlamaIndex node #%d: %s", a.get("index"), a)
            for idx, text in enumerate(node_texts):
                eval_res = self.engine.evaluate_document(text)
                if not eval_res["is_safe"]:
                    logger.warning("[DRY RUN] Threat detected in LlamaIndex node #%d: %s", idx, eval_res["threats"])
            return list(nodes)

        # Perform semantic anomaly detection if enough nodes
        quarantined_indices = set()
        if len(node_texts) >= 3 and self.strict_quarantine:
            anomalies = self.engine.detect_semantic_anomalies(node_texts)
            quarantined_indices = {
                a["index"]
                for a in anomalies
                if a.get("severity") in ("CRITICAL", "HIGH")
            }

        safe_nodes = []
        for idx, node in enumerate(nodes):
            if idx in quarantined_indices:
                logger.warning("Quarantined poisoned LlamaIndex node at index %d", idx)
                continue

            raw_text = node_texts[idx]
            cleaned = self.engine.strip_markdown_xss(raw_text)
            sanitized = self.engine.strip_injections(cleaned)
            if self.wrap_taint:
                sanitized = self.engine.wrap_taint_boundary(sanitized, source=f"rag_node_{idx}")

            # Update content back into node
            if hasattr(node, "node") and hasattr(node.node, "text"):
                node.node.text = sanitized
            elif hasattr(node, "text"):
                node.text = sanitized
            elif hasattr(node, "set_content"):
                node.set_content(sanitized)

            safe_nodes.append(node)

        return safe_nodes


class CrewAIToolGuard:
    """
    Decorator and tool wrapper for CrewAI tools.
    Sanitizes both tool inputs (preventing SQL/command/tool injection)
    and tool execution outputs (neutralizing indirect injections before agent ingestion).
    Supports dry_run audit mode.
    """

    def __init__(
        self,
        tool_or_func: Optional[Any] = None,
        engine: Optional[PoisonDefenseEngine] = None,
        wrap_taint: bool = True,
        dry_run: Optional[bool] = None,
    ) -> None:
        self.engine = engine or PoisonDefenseEngine()
        self.wrap_taint = wrap_taint
        self.tool = tool_or_func
        cfg = get_config() if callable(get_config) else None
        if dry_run is not None:
            self.dry_run = dry_run
        elif cfg and hasattr(cfg, "dry_run"):
            self.dry_run = cfg.dry_run
        else:
            self.dry_run = False

        if tool_or_func is not None and not callable(tool_or_func):
            # Wrapping an existing CrewAI Tool object
            self._wrap_tool_instance(tool_or_func)

    def _wrap_tool_instance(self, tool_obj: Any) -> None:
        """Wraps tool._run or tool.run in-place."""
        for run_name in ("_run", "run"):
            if hasattr(tool_obj, run_name):
                original_run = getattr(tool_obj, run_name)

                @functools.wraps(original_run)
                def guarded_run(*args: Any, **kwargs: Any) -> Any:
                    sanitized_args = [self._sanitize_input_value(a) for a in args]
                    sanitized_kwargs = {k: self._sanitize_input_value(v) for k, v in kwargs.items()}
                    raw_result = original_run(*sanitized_args, **sanitized_kwargs)
                    return self._sanitize_output_value(raw_result)

                setattr(tool_obj, run_name, guarded_run)

    def _sanitize_input_value(self, val: Any) -> Any:
        if self.dry_run:
            if isinstance(val, str):
                eval_res = self.engine.evaluate_document(val)
                if not eval_res["is_safe"]:
                    logger.warning("[DRY RUN] Threat detected in CrewAI tool input: %s", eval_res["threats"])
            return val
        if isinstance(val, str):
            cleaned = self.engine.strip_markdown_xss(val)
            return self.engine.strip_injections(cleaned)
        return val

    def _sanitize_output_value(self, val: Any) -> Any:
        if self.dry_run:
            if isinstance(val, str):
                eval_res = self.engine.evaluate_document(val)
                _, leaks = self.engine.filter_egress_leaks(val)
                if not eval_res["is_safe"] or leaks:
                    logger.warning("[DRY RUN] Threat/leaks in CrewAI tool output: %s, %s", eval_res["threats"], leaks)
            return val
        if isinstance(val, str):
            cleaned = self.engine.strip_markdown_xss(val)
            sanitized = self.engine.strip_injections(cleaned)
            sanitized, _ = self.engine.filter_egress_leaks(sanitized)
            if self.wrap_taint and ("[REDACTED_INJECTION_ATTEMPT]" in sanitized or "[ADVERSARIAL_SUFFIX" in sanitized):
                return self.engine.wrap_taint_boundary(sanitized, source="crewai_tool_output")
            return sanitized
        return val

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Allows CrewAIToolGuard to be used as a decorator or callable."""
        if self.tool is not None and callable(self.tool):
            sanitized_args = [self._sanitize_input_value(a) for a in args]
            sanitized_kwargs = {k: self._sanitize_input_value(v) for k, v in kwargs.items()}
            raw_result = self.tool(*sanitized_args, **sanitized_kwargs)
            return self._sanitize_output_value(raw_result)

        # Decorator invocation
        func = args[0]
        @functools.wraps(func)
        def wrapper(*f_args: Any, **f_kwargs: Any) -> Any:
            sanitized_args = [self._sanitize_input_value(a) for a in f_args]
            sanitized_kwargs = {k: self._sanitize_input_value(v) for k, v in f_kwargs.items()}
            raw_result = func(*sanitized_args, **sanitized_kwargs)
            return self._sanitize_output_value(raw_result)

        return wrapper


__all__ = [
    "PoisonArmorMiddleware",
    "PoisonArmorClient",
    "wrap_openai",
    "LangChainPoisonArmorCallback",
    "LlamaIndexPoisonArmorPostprocessor",
    "CrewAIToolGuard",
]
