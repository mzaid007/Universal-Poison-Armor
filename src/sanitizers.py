"""
Sanitizers and Poison Defense Engine (Root wrapper / alias)
==========================================================
"""
import sys
from pathlib import Path

# Add repo root to sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from skills.ai_poison_defense.src.sanitizers import (  # type: ignore
        PoisonDefenseEngine,
        logger,
    )
except ImportError:
    from sanitizers import PoisonDefenseEngine, logger  # type: ignore

__all__ = ["PoisonDefenseEngine", "logger"]
