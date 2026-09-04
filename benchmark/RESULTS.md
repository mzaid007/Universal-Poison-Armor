# Universal Poison Armor - Attack Suite Benchmark Results

> **Evaluation Run Date**: 2026-09-05
> **Corpus**: `benchmark/attack_suite.json`
> **Total Test Vectors Evaluated**: 87 (72 attacks, 15 benign controls)

## Executive Summary

- **Attack Neutralization Rate (Recall / TPR)**: **`100.0%`**
- **False Positive Rate (FPR)**: **`0.0%`** (0 false alarms on legitimate code, queries, and multi-lingual text)
- **Overall Accuracy**: **`100.0%`**
- **Precision**: **`100.0%`**
- **F1-Score**: **`1.0`**
- **Median Latency (P50)**: **`11.12 ms`**
- **95th Percentile Latency (P95)**: **`14.4 ms`**

## Classification Confusion Matrix

| Metric | Count | Description |
| :--- | :--- | :--- |
| **True Positives (TP)** | `72` | Attack vectors successfully neutralized |
| **True Negatives (TN)** | `15` | Benign inputs passed untouched without false alarm |
| **False Positives (FP)** | `0` | Benign inputs incorrectly flagged as attacks |
| **False Negatives (FN)** | `0` | Attacks that bypassed detection |

## Latency Profile

| Metric | Time (ms) | Description |
| :--- | :--- | :--- |
| **Mean Latency** | `11.48 ms` | Average end-to-end multi-layer evaluation latency |
| **P50 (Median)** | `11.12 ms` | 50% of requests evaluated under this threshold |
| **P95** | `14.4 ms` | 95% of requests evaluated under this threshold |
| **P99** | `18.99 ms` | 99% of requests evaluated under this threshold |

## Category Neutralization Breakdown

| Category | Total | Attacks | Detected | Attack Recall (TPR) | Benign Pass Rate | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `adversarial_suffixes_gcg` | 8 | 8 | 8 | **100.0%** | N/A | 10.16 ms |
| `benign_controls` | 15 | 0 | 0 | **N/A** | 100.0% | 10.56 ms |
| `direct_prompt_injection` | 12 | 12 | 12 | **100.0%** | N/A | 11.2 ms |
| `egress_secret_leaks` | 8 | 8 | 8 | **100.0%** | N/A | 13.84 ms |
| `indirect_prompt_injection` | 10 | 10 | 10 | **100.0%** | N/A | 11.2 ms |
| `jailbreak_dan` | 8 | 8 | 8 | **100.0%** | N/A | 11.15 ms |
| `markdown_xss_tracking_pixels` | 8 | 8 | 8 | **100.0%** | N/A | 11.3 ms |
| `multilingual_injections` | 10 | 10 | 10 | **100.0%** | N/A | 12.57 ms |
| `obfuscation_attacks` | 8 | 8 | 8 | **100.0%** | N/A | 12.13 ms |

## Multi-Layer Trigger Distribution

| Detection Layer | Trigger Count | Description |
| :--- | :---: | :--- |
| **`HEURISTIC_REGEX`** | `48` | Deterministic keyword & regex pattern matching (system overrides, DAN, jailbreaks) |
| **`EGRESS_FILTER`** | `16` | Outbound leak interception (OpenAI, Anthropic, AWS, GitHub, JWT tokens) |
| **`NEURAL_SEMANTIC`** | `10` | Local dense embeddings cosine similarity for semantic injection intent |
| **`SHANNON_ENTROPY`** | `10` | Mathematical entropy & character diversity detection for GCG adversarial suffixes |
| **`XSS_TRACKING_PIXEL`** | `8` | Markdown image, HTML img, and iframe tracking beacon stripping |
| **`DEOBFUSCATION`** | `5` | Recursive decoding of Base64, Hex, and URL percent-encoded payloads |
| **`UNICODE_STEGANOGRAPHY`** | `3` | Zero-width and invisible directional Unicode sequence stripping |

## Verification Reproducibility

To reproduce these benchmark results locally or on CI/CD pipelines:
```bash
# Run automated benchmark suite
python benchmark/run_benchmark.py
```