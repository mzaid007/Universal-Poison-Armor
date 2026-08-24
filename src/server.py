import os
import sys
from pathlib import Path

# Add repo root to sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from skills.ai_poison_defense.src.server import (  # type: ignore
    mcp,
    sanitize_document,
    scan_dataset_for_anomalies,
    verify_article_consensus,
)

if __name__ == "__main__":
    transport = os.environ.get("MCP_TRANSPORT", "sse").lower()
    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        port = int(os.environ.get("PORT", 8080))
        host = os.environ.get("HOST", "0.0.0.0")
        mcp.run(transport="sse", host=host, port=port)

