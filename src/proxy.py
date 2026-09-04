"""
Universal Poison Armor - Transparent Reverse Proxy Gateway
==========================================================
Provides an OpenAI-compatible HTTP/SSE reverse proxy server.
Sits transparently between LLM agents (regardless of programming language or framework)
and upstream model providers (OpenAI, Anthropic, Ollama, vLLM, Azure).

Every request to `/v1/chat/completions` is automatically sanitized:
1. Strips Markdown XSS and IP-exfiltrating tracking pixels.
2. Neutralizes invisible zero-width Unicode steganography.
3. Redacts heuristic prompt injection attempts.
4. Detects mathematical adversarial suffixes (GCG attacks).
5. Optionally wraps untrusted content in cryptographic taint delimiters.
6. Records all intercepted threats into `security_audit.json`.

Usage:
    python -m src.proxy --port 8000 --upstream https://api.openai.com/v1
Or set environment variables:
    UPSTREAM_API_BASE=https://api.openai.com/v1
    PROXY_PORT=8000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
import httpx
import uvicorn

# Support flexible imports
try:
    from .sanitizers import PoisonDefenseEngine
    from .server import log_security_audit, metrics
except ImportError:
    try:
        from src.sanitizers import PoisonDefenseEngine
        from src.server import log_security_audit, metrics
    except ImportError:
        from sanitizers import PoisonDefenseEngine  # type: ignore
        from server import log_security_audit, metrics  # type: ignore

try:
    from src.config import get_config
except ImportError:
    try:
        from config import get_config
    except ImportError:
        get_config = None

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
)
logger = logging.getLogger("UniversalPoisonArmor.Proxy")

app = FastAPI(
    title="Universal Poison Armor Reverse Proxy Gateway",
    description="Transparent OpenAI & Anthropic-compatible security firewall proxy for AI agents.",
    version="1.0.0",
)

# Global engine singleton
engine = PoisonDefenseEngine()
UPSTREAM_API_BASE = os.environ.get("UPSTREAM_API_BASE", "").rstrip("/")


def resolve_upstream(request: Request) -> str:
    """Resolves upstream API endpoint dynamically from headers or environment configuration."""
    custom_hdr = request.headers.get("x-upstream-api-base") or request.headers.get("x-upstream-base-url")
    if custom_hdr:
        return custom_hdr.rstrip("/")
    cfg = get_config() if get_config else None
    return UPSTREAM_API_BASE or (cfg.upstream_api_base.rstrip("/") if cfg else "")


def sanitize_payload(body: Dict[str, Any], wrap_taint: bool = False) -> Tuple[Dict[str, Any], List[str]]:
    """
    Inspects and sanitizes all messages inside an OpenAI-style chat payload.

    Args:
        body: Request JSON dictionary containing 'messages'.
        wrap_taint: If True, wraps modified or untrusted prompts with taint delimiters.

    Returns:
        Tuple of (sanitized_body, detected_threat_types).
    """
    messages = body.get("messages", [])
    if not isinstance(messages, list):
        return body, []

    detected_threats: List[str] = []
    sanitized_messages = []

    for msg in messages:
        if not isinstance(msg, dict):
            sanitized_messages.append(msg)
            continue

        role = msg.get("role", "")
        content = msg.get("content", "")

        if isinstance(content, str) and content:
            # 1. Strip Markdown XSS / tracking pixels
            xss_cleaned = engine.strip_markdown_xss(content)
            if xss_cleaned != content:
                detected_threats.append("MARKDOWN_XSS_TRACKING_PIXEL")

            # 2. Strip injections and zero-width steganography
            fully_sanitized = engine.strip_injections(xss_cleaned)
            if "[REDACTED_INJECTION_ATTEMPT]" in fully_sanitized:
                detected_threats.append("PROMPT_INJECTION_ATTEMPT")
            if "[ADVERSARIAL_SUFFIX_THREAT" in fully_sanitized:
                detected_threats.append("ADVERSARIAL_SUFFIX_THREAT")
            if engine.ZERO_WIDTH_PATTERN.search(content):
                detected_threats.append("ZERO_WIDTH_STEGANOGRAPHY")

            if wrap_taint and detected_threats:
                fully_sanitized = engine.wrap_taint_boundary(fully_sanitized, source=f"{role}_input")

            sanitized_messages.append({**msg, "content": fully_sanitized})
        else:
            sanitized_messages.append(msg)

    sanitized_body = {**body, "messages": sanitized_messages}
    return sanitized_body, detected_threats


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint for container orchestrators and load balancers."""
    return {
        "status": "healthy",
        "service": "Universal Poison Armor Reverse Proxy Gateway",
        "upstream_configured": bool(UPSTREAM_API_BASE),
    }


def sanitize_sse_line(line: str, engine: PoisonDefenseEngine) -> Tuple[str, List[str]]:
    """
    Inspects an SSE data line for model output delta content and redacts secret leaks.
    Returns (sanitized_line_str, detected_leaks).
    """
    if line.startswith("data: ") and line.strip() != "data: [DONE]":
        try:
            data_str = line[6:].strip()
            chunk_json = json.loads(data_str)
            choices = chunk_json.get("choices", [])
            all_leaks = []
            for choice in choices:
                delta = choice.get("delta", {})
                content = delta.get("content", "")
                if content and isinstance(content, str):
                    sanitized_content, detected_leaks = engine.filter_egress_leaks(content)
                    if sanitized_content != content:
                        delta["content"] = sanitized_content
                        all_leaks.extend(detected_leaks)
            if all_leaks:
                return f"data: {json.dumps(chunk_json)}\n\n", all_leaks
        except Exception:
            pass
    clean_line = line.rstrip("\r\n")
    return (f"{clean_line}\n\n" if clean_line else "\n"), []


async def stream_response_processor(
    upstream_response: Any,
    engine: PoisonDefenseEngine,
    request: Optional[Any] = None,
    metrics_tracker: Optional[Any] = None,
):
    """
    Asynchronous generator yielding sanitized SSE chunks while inspecting
    in-flight tokens for credential leaks and checking client disconnects.
    """
    async for line in upstream_response.aiter_lines():
        if request is not None and hasattr(request, "is_disconnected") and await request.is_disconnected():
            logger.info("Client disconnected from streaming request.")
            break
        if not line:
            yield b"\n"
            continue
        sanitized_line, leaks = sanitize_sse_line(line, engine)
        if leaks:
            for leak in leaks:
                if metrics_tracker is not None and hasattr(metrics_tracker, "increment_layer_hit"):
                    metrics_tracker.increment_layer_hit("STREAM_EGRESS")
                log_security_audit(
                    f"PROXY_STREAM_EGRESS_LEAK_{leak}",
                    line[:200],
                    detection_layer="STREAM_EGRESS",
                    severity="CRITICAL",
                )
        yield sanitized_line.encode("utf-8")


@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request) -> Response:
    """
    Transparent reverse proxy endpoint matching OpenAI /v1/chat/completions specification.
    Sanitizes all messages in the payload before forwarding to the upstream LLM API.
    Supports dry-run monitor mode via header 'X-Poison-Armor-Dry-Run: true'.
    """
    try:
        raw_body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {e}")

    # Determine dry-run mode
    cfg = get_config() if get_config else None
    is_dry_run = (
        request.headers.get("x-poison-armor-dry-run", "").strip().lower() in ("true", "1")
        or bool(cfg and cfg.dry_run)
    )

    # Sanitize incoming payload
    sanitized_body, detected_threats = sanitize_payload(raw_body)
    body_to_forward = raw_body if is_dry_run else sanitized_body

    # Log threats to central security audit file
    action_label = "FLAGGED_DRY_RUN" if is_dry_run else "REDACTED"
    for threat in dict.fromkeys(detected_threats):
        log_security_audit(
            f"PROXY_INTERCEPTED_{threat}",
            json.dumps(raw_body.get("messages", []))[:500],
            action=action_label,
            detection_layer="PROXY_GATEWAY",
        )

    is_streaming = bool(raw_body.get("stream", False))
    upstream = resolve_upstream(request)

    response_headers: Dict[str, str] = {}
    if detected_threats:
        response_headers["X-Poison-Armor-Threats-Detected"] = str(len(detected_threats))
        response_headers["X-Poison-Armor-Threat-Types"] = ",".join(dict.fromkeys(detected_threats))
    if is_dry_run:
        response_headers["X-Poison-Armor-Mode"] = "dry-run"
        response_headers["X-Poison-Armor-Dry-Run"] = "true"

    # If no upstream is configured, return sanitized payload inspection (Mock / Test mode)
    if not upstream:
        if is_streaming:
            async def mock_stream_generator():
                import time
                created_ts = int(time.time())
                model_id = sanitized_body.get("model", "universal-poison-armor-proxy")

                chunk_1 = {
                    "id": f"chatcmpl-mock-{created_ts}",
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk_1)}\n\n"

                threat_count = len(detected_threats)
                mock_text = f"[Universal Poison Armor Gateway: Mock Stream Verified - {threat_count} threats intercepted]"
                chunk_2 = {
                    "id": f"chatcmpl-mock-{created_ts}",
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {"content": mock_text}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk_2)}\n\n"

                chunk_3 = {
                    "id": f"chatcmpl-mock-{created_ts}",
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(chunk_3)}\n\n"
                yield "data: [DONE]\n\n"

            stream_hdrs = {"Cache-Control": "no-cache", "Connection": "keep-alive", **response_headers}
            return StreamingResponse(
                mock_stream_generator(),
                media_type="text/event-stream",
                headers=stream_hdrs,
            )

        return JSONResponse(
            content={
                "status": "sanitized",
                "proxy": "Universal Poison Armor Gateway",
                "threats_intercepted": detected_threats,
                "sanitized_request": sanitized_body,
                "message": "No UPSTREAM_API_BASE configured. Set UPSTREAM_API_BASE to forward to real LLM provider.",
            },
            headers=response_headers,
        )

    target_url = f"{upstream}/chat/completions" if not upstream.endswith("/chat/completions") else upstream
    forward_headers = dict(request.headers)
    forward_headers.pop("host", None)
    forward_headers.pop("content-length", None)

    if is_streaming:
        async def upstream_stream_generator():
            client = httpx.AsyncClient(timeout=120.0)
            try:
                async with client.stream(
                    "POST",
                    target_url,
                    json=body_to_forward,
                    headers=forward_headers,
                ) as upstream_response:
                    async for chunk in stream_response_processor(upstream_response, engine, request, metrics):
                        yield chunk
            except Exception as stream_err:
                logger.error("Streaming error from upstream %s: %s", target_url, stream_err)
                err_chunk = {"error": {"message": f"Upstream streaming error: {stream_err}", "type": "proxy_error"}}
                yield f"data: {json.dumps(err_chunk)}\n\n".encode("utf-8")
            finally:
                await client.aclose()

        stream_hdrs = {"Cache-Control": "no-cache", "Connection": "keep-alive", **response_headers}
        return StreamingResponse(
            upstream_stream_generator(),
            media_type="text/event-stream",
            headers=stream_hdrs,
        )

    # Non-streaming forward
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            upstream_resp = await client.post(
                target_url,
                json=body_to_forward,
                headers=forward_headers,
            )
            resp_content = upstream_resp.content
            resp_headers = dict(upstream_resp.headers)
            resp_headers.pop("content-length", None)
            resp_headers.pop("content-encoding", None)
            resp_headers.update(response_headers)

            # Egress leak filtering on upstream response
            try:
                resp_json = upstream_resp.json()
                choices = resp_json.get("choices", [])
                egress_modified = False
                for choice in choices:
                    msg = choice.get("message", {})
                    content = msg.get("content", "")
                    if isinstance(content, str) and content:
                        sanitized_content, detected_leaks = engine.filter_egress_leaks(content)
                        if sanitized_content != content:
                            msg["content"] = sanitized_content
                            egress_modified = True
                            for leak in detected_leaks:
                                log_security_audit(f"PROXY_INTERCEPTED_{leak}", content[:500])
                if egress_modified:
                    return JSONResponse(
                        content=resp_json,
                        status_code=upstream_resp.status_code,
                        headers=resp_headers,
                    )
            except Exception:
                pass

            return Response(
                content=resp_content,
                status_code=upstream_resp.status_code,
                headers=resp_headers,
            )
        except Exception as forward_err:
            logger.error("Failed to forward request to upstream %s: %s", target_url, forward_err)
            raise HTTPException(status_code=502, detail=f"Upstream provider connection error: {forward_err}")


@app.post("/v1/messages")
async def proxy_anthropic_messages(request: Request) -> Response:
    """
    Anthropic /v1/messages endpoint.
    Sanitizes messages before forwarding to upstream Anthropic or multi-provider API.
    """
    try:
        raw_body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {e}")

    sanitized_body, detected_threats = sanitize_payload(raw_body)
    for threat in dict.fromkeys(detected_threats):
        log_security_audit(f"PROXY_ANTHROPIC_{threat}", json.dumps(raw_body.get("messages", []))[:500])

    upstream = resolve_upstream(request)
    if not upstream:
        return JSONResponse({
            "status": "sanitized",
            "proxy": "Universal Poison Armor Gateway (Anthropic Mode)",
            "threats_intercepted": detected_threats,
            "sanitized_request": sanitized_body,
        })

    target_url = f"{upstream}/v1/messages" if not upstream.endswith("/v1") else f"{upstream}/messages"
    forward_headers = dict(request.headers)
    forward_headers.pop("host", None)
    forward_headers.pop("content-length", None)

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            upstream_resp = await client.post(target_url, json=sanitized_body, headers=forward_headers)
            return Response(
                content=upstream_resp.content,
                status_code=upstream_resp.status_code,
                headers=dict(upstream_resp.headers),
            )
        except Exception as forward_err:
            logger.error("Failed to forward request to Anthropic upstream %s: %s", target_url, forward_err)
            raise HTTPException(status_code=502, detail=f"Anthropic provider connection error: {forward_err}")


@app.get("/metrics")
async def get_proxy_prometheus_metrics() -> Response:
    """Returns Prometheus text-format metrics."""
    if metrics and hasattr(metrics, "get_prometheus_metrics"):
        body = metrics.get_prometheus_metrics()
        return PlainTextResponse(content=body, media_type="text/plain; version=0.0.4")
    return PlainTextResponse(content="# No metrics available\n")


@app.get("/v1/stats")
async def get_proxy_stats() -> Dict[str, Any]:
    """Returns real-time defense telemetry, throughput, and threat statistics."""
    stats = metrics.get_stats() if hasattr(metrics, "get_stats") else (metrics.get_metrics() if metrics else {})
    return {
        "status": "active",
        "service": "Universal Poison Armor Gateway",
        "total_requests": stats.get("total_scans", 0),
        "uptime_seconds": stats.get("uptime_seconds", 0),
        "layer_hits": stats.get("hits_by_layer", {}),
        "metrics": stats,
    }


def run_proxy(host: str = "0.0.0.0", port: int = 8000, upstream: Optional[str] = None) -> None:
    """Run the proxy server with Uvicorn."""
    global UPSTREAM_API_BASE
    if upstream:
        UPSTREAM_API_BASE = upstream.rstrip("/")
    logger.info("Starting Universal Poison Armor Proxy on http://%s:%d -> Upstream: %s", host, port, UPSTREAM_API_BASE or "None (Inspection Mode)")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal Poison Armor Reverse Proxy Gateway")
    parser.add_argument("--host", default=os.environ.get("PROXY_HOST", "0.0.0.0"), help="Host to bind to")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PROXY_PORT", 8000)), help="Port to bind to")
    parser.add_argument("--upstream", default=os.environ.get("UPSTREAM_API_BASE", ""), help="Upstream LLM API base URL (e.g. https://api.openai.com/v1)")
    args = parser.parse_args()

    run_proxy(host=args.host, port=args.port, upstream=args.upstream)
