import os
import sys
from pathlib import Path

# Add repo root and skills source directory to sys.path
repo_root = Path(__file__).resolve().parent.parent
skill_src = repo_root / "skills" / "ai-poison-defense" / "src"

for p in [str(skill_src), str(repo_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from server import (  # type: ignore
        mcp,
        run_server,
        sanitize_document,
        scan_dataset_for_anomalies,
        verify_article_consensus,
    )
except ImportError:
    from skills.ai_poison_defense.src.server import (  # type: ignore
        mcp,
        run_server,
        sanitize_document,
        scan_dataset_for_anomalies,
        verify_article_consensus,
    )

__all__ = [
    "mcp",
    "run_server",
    "sanitize_document",
    "scan_dataset_for_anomalies",
    "verify_article_consensus",
]

if __name__ == "__main__":
    run_server()



