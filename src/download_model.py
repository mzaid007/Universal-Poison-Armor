"""
Universal Poison Armor - Neural Model Downloader & ONNX Exporter
===============================================================
Automates the local acquisition and ONNX optimization of neural prompt injection classifiers.
Defaults to `protectai/deberta-v3-base-prompt-injection-v2` for 100% offline, low-latency,
zero-provider-dependency semantic prompt injection inference.

Usage:
    python -m src.download_model --model-id protectai/deberta-v3-base-prompt-injection-v2 --output-dir models/deberta-v3-prompt-injection
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
)
logger = logging.getLogger("UniversalPoisonArmor.ModelDownloader")

DEFAULT_MODEL_ID = "protectai/deberta-v3-base-prompt-injection-v2"
DEFAULT_OUTPUT_DIR = "models/deberta-v3-prompt-injection"


def download_and_export(
    model_id: str = DEFAULT_MODEL_ID,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    export_onnx: bool = True,
) -> Path:
    """
    Downloads a HuggingFace prompt injection classification model and exports it to ONNX format.

    Args:
        model_id: HuggingFace model repo identifier.
        output_dir: Local filesystem directory to save model assets and ONNX graph.
        export_onnx: If True, exports PyTorch weights to an optimized ONNX runtime graph.

    Returns:
        Path to the output directory containing the model.
    """
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    onnx_file_path = out_path / "model.onnx"

    logger.info("Initializing download for model '%s' -> %s", model_id, out_path)

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        logger.error("transformers package is required. Install via `pip install transformers`.")
        raise

    logger.info("Fetching tokenizer: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(out_path)
    logger.info("Tokenizer saved successfully.")

    if not export_onnx:
        logger.info("Fetching PyTorch weights without ONNX conversion...")
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        model.save_pretrained(out_path)
        logger.info("PyTorch model saved to %s", out_path)
        return out_path

    # Export to ONNX
    logger.info("Attempting ONNX export for '%s'...", model_id)

    # Strategy 1: Optimum ORTModel if available
    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification  # type: ignore
        logger.info("Using Optimum ONNX Runtime exporter...")
        ort_model = ORTModelForSequenceClassification.from_pretrained(model_id, export=True)
        ort_model.save_pretrained(out_path)
        logger.info("Optimum ONNX export succeeded: %s", onnx_file_path)
        return out_path
    except (ImportError, Exception) as opt_err:
        logger.debug("Optimum export skipped/failed (%s), falling back to PyTorch ONNX export...", opt_err)

    # Strategy 2: Native PyTorch ONNX export
    try:
        import torch
        logger.info("Loading PyTorch model for native torch.onnx.export...")
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        model.eval()

        dummy_text = "Hello, world! This is a test prompt injection payload."
        inputs = tokenizer(dummy_text, return_tensors="pt", padding=True, truncation=True, max_length=128)

        input_names = ["input_ids", "attention_mask"]
        output_names = ["logits"]
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        }

        # Handle token_type_ids if required by model
        dummy_args = (inputs["input_ids"], inputs["attention_mask"])
        if "token_type_ids" in inputs:
            input_names.append("token_type_ids")
            dynamic_axes["token_type_ids"] = {0: "batch_size", 1: "sequence_length"}
            dummy_args = (inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"])

        torch.onnx.export(
            model,
            dummy_args,
            str(onnx_file_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=14,
            do_constant_folding=True,
        )
        logger.info("PyTorch torch.onnx.export succeeded -> %s", onnx_file_path)

        # Save PyTorch config alongside ONNX model
        model.config.save_pretrained(out_path)

        # Strategy verification: Test ONNX Runtime session loading
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(str(onnx_file_path))
            logger.info("ONNX Runtime session verified successfully with %d inputs.", len(session.get_inputs()))
        except Exception as sess_err:
            logger.warning("ONNX session verification warning: %s", sess_err)

        return out_path

    except Exception as torch_err:
        logger.error("Failed native PyTorch ONNX export: %s", torch_err)
        # Save standard PyTorch weights as fallback
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        model.save_pretrained(out_path)
        logger.info("Saved PyTorch weights as fallback without ONNX graph to %s", out_path)
        return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Universal Poison Armor - Model Downloader & ONNX Exporter")
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"HuggingFace Model repo ID (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Target directory for saved ONNX model (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--no-onnx",
        action="store_true",
        help="Download PyTorch weights only without ONNX export",
    )
    args = parser.parse_args()

    download_and_export(
        model_id=args.model_id,
        output_dir=args.output_dir,
        export_onnx=not args.no_onnx,
    )


if __name__ == "__main__":
    main()
