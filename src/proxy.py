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
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
import uvicorn

# Support flexible imports
try:
    from .sanitizers import PoisonDefenseEngine
    from .server import log_security_audit
except ImportError:
    try:
        from src.sanitizers import PoisonDefenseEngine
        from src.server import log_security_audit
    except ImportError:
        from sanitizers import PoisonDefenseEngine  # type: ignore
        from server import log_security_audit  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
)
logger = logging.getLogger("UniversalPoisonArmor.Proxy")

app = FastAPI(
    title="Universal Poison Armor Reverse Proxy Gateway",
    description="Transparent OpenAI-compatible security firewall proxy for AI agents.",
    version="1.0.0",
)

# Global engine singleton
engine = PoisonDefenseEngine()
UPSTREAM_API_BASE = os.environ.get("UPSTREAM_API_BASE", "").rstrip("/")


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


@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request) -> Response:
    """
    Transparent reverse proxy endpoint matching OpenAI /v1/chat/completions specification.
    Sanitizes all messages in the payload before forwarding to the upstream LLM API.
    """
    try:
        raw_body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {e}")

    # Sanitize incoming payload
    sanitized_body, detected_threats = sanitize_payload(raw_body)

    # Log threats to central security audit file
    for threat in dict.fromkeys(detected_threats):
        log_security_audit(f"PROXY_INTERCEPTED_{threat}", json.dumps(raw_body.get("messages", []))[:500])

    upstream = UPSTREAM_API_BASE or os.environ.get("UPSTREAM_API_BASE", "").rstrip("/")

    # If no upstream is configured, return sanitized payload inspection (Mock / Test mode)
    if not upstream:
        return JSONResponse({
            "status": "sanitized",
            "proxy": "Universal Poison Armor Gateway",
            "threats_intercepted": detected_threats,
            "sanitized_request": sanitized_body,
            "message": "No UPSTREAM_API_BASE configured. Set UPSTREAM_API_BASE to forward to real LLM provider.",
        })

    # Forward to upstream LLM API
    target_url = f"{upstream}/chat/completions"
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            upstream_resp = await client.post(
                target_url,
                json=sanitized_body,
                headers=headers,
            )
            return Response(
                content=upstream_resp.content,
                status_code=upstream_resp.status_code,
                headers=dict(upstream_resp.headers),
            )
        except Exception as forward_err:
            logger.error("Failed to forward request to upstream %s: %s", target_url, forward_err)
            raise HTTPException(status_code=502, detail=f"Upstream provider connection error: {forward_err}")


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
