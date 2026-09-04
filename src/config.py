"""
Universal Poison Armor - Configuration Layer
============================================
Provides centralized, zero-code configuration via environment variables
and optional JSON/YAML/ENV configuration files.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("UniversalPoisonArmor.Config")


def _bool_from_env(key: str, default: bool) -> bool:
    val = os.environ.get(key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on", "enabled")


@dataclass
class PoisonArmorConfig:
    """Centralized configuration for Universal Poison Armor defense layers."""

    # Mathematical & Entropy Detection
    entropy_threshold: float = field(
        default_factory=lambda: float(os.environ.get("POISON_ARMOR_ENTROPY_THRESHOLD", "4.5"))
    )

    # Neural & Semantic Detection
    neural_threshold: float = field(
        default_factory=lambda: float(os.environ.get("POISON_ARMOR_NEURAL_THRESHOLD", "0.45"))
    )
    check_neural: bool = field(
        default_factory=lambda: _bool_from_env("POISON_ARMOR_CHECK_NEURAL", True)
    )
    onnx_model_path: str = field(
        default_factory=lambda: os.environ.get(
            "POISON_ARMOR_ONNX_MODEL_PATH",
            "models/deberta-v3-prompt-injection/model.onnx"
        )
    )

    # Operational Modes
    dry_run: bool = field(
        default_factory=lambda: _bool_from_env("POISON_ARMOR_DRY_RUN", False)
    )
    wrap_taint: bool = field(
        default_factory=lambda: _bool_from_env("POISON_ARMOR_WRAP_TAINT", False)
    )

    # DoS & Resource Caps
    max_document_size: int = field(
        default_factory=lambda: int(
            os.environ.get("POISON_ARMOR_MAX_DOCUMENT_SIZE")
            or os.environ.get("POISON_ARMOR_MAX_DOC_SIZE")
            or str(5 * 1024 * 1024)
        )
    )
    max_batch_size: int = field(
        default_factory=lambda: int(os.environ.get("POISON_ARMOR_MAX_BATCH_SIZE", "500"))
    )

    # Audit Logging & Rotation
    audit_max_bytes: int = field(
        default_factory=lambda: int(os.environ.get("POISON_ARMOR_AUDIT_MAX_BYTES", str(10 * 1024 * 1024)))
    )
    audit_backup_count: int = field(
        default_factory=lambda: int(os.environ.get("POISON_ARMOR_AUDIT_BACKUP_COUNT", "5"))
    )

    # Gateway & Proxy Upstream
    upstream_api_base: str = field(
        default_factory=lambda: os.environ.get("UPSTREAM_API_BASE", os.environ.get("POISON_ARMOR_UPSTREAM_BASE", ""))
    )
    log_level: str = field(
        default_factory=lambda: os.environ.get("POISON_ARMOR_LOG_LEVEL", "INFO")
    )

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "PoisonArmorConfig":
        """
        Loads configuration from an explicit file path, discovery paths, or environment variables.
        Supported file formats: .json, .env
        """
        instance = cls()

        # Check explicit path or discoverable defaults
        candidates = []
        if config_path:
            candidates.append(Path(config_path))
        env_conf = os.environ.get("POISON_ARMOR_CONFIG")
        if env_conf:
            candidates.append(Path(env_conf))

        candidates.extend([
            Path("poison_armor.json"),
            Path("poison_armor_config.json"),
            Path(".poison_armor.json"),
        ])

        for path in candidates:
            if path.is_file():
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    for k, v in data.items():
                        if hasattr(instance, k):
                            setattr(instance, k, v)
                    logger.info("Loaded Universal Poison Armor config from %s", path)
                    break
                except Exception as err:
                    logger.warning("Failed loading config from %s: %s", path, err)

        return instance

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_GLOBAL_CONFIG: Optional[PoisonArmorConfig] = None


def get_config(reload: bool = False) -> PoisonArmorConfig:
    """Returns the global PoisonArmorConfig singleton."""
    global _GLOBAL_CONFIG
    if _GLOBAL_CONFIG is None or reload:
        _GLOBAL_CONFIG = PoisonArmorConfig.load()
    return _GLOBAL_CONFIG


def reset_config() -> None:
    """Resets the global config singleton so it reloads on next access."""
    global _GLOBAL_CONFIG
    _GLOBAL_CONFIG = None


__all__ = ["PoisonArmorConfig", "get_config", "reset_config"]
