import sys
from pathlib import Path

# Add repo root and skills source directory to sys.path
repo_root = Path(__file__).resolve().parent.parent
skill_src = repo_root / "skills" / "ai-poison-defense" / "src"

for p in [str(skill_src), str(repo_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from sanitizers import PoisonDefenseEngine, logger  # type: ignore
except ImportError:
    from skills.ai_poison_defense.src.sanitizers import (  # type: ignore
        PoisonDefenseEngine,
        logger,
    )

__all__ = ["PoisonDefenseEngine", "logger"]

