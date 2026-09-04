# Universal Poison Armor - Full Public Dataset Evaluation Report

> **Evaluation Run Date**: 2026-09-05
> **Corpora Evaluated**: Official Microsoft BIPIA & JailbreakBench datasets
> **Total External Samples Evaluated**: 53

## Executive Summary

- **Attack Neutralization Rate (Recall / TPR)**: **`81.82%`** (27/33)
- **False Positive Rate (FPR)**: **`20.0%`** (4/20)
- **Overall Accuracy**: **`81.13%`**
- **Precision**: **`87.1%`**
- **F1-Score**: **`0.8438`**
- **Median Latency (P50)**: **`41.66 ms`**
- **95th Percentile Latency (P95)**: **`136.49 ms`**

## Confusion Matrix

| Metric | Count | Description |
| :--- | :--- | :--- |
| **True Positives (TP)** | `27` | External attack vectors successfully neutralized |
| **True Negatives (TN)** | `16` | Benign samples passed without false alarm |
| **False Positives (FP)** | `4` | Benign samples incorrectly flagged |
| **False Negatives (FN)** | `6` | Attacks that bypassed detection |

## Multi-Layer Trigger Breakdown

| Layer Name | Detections | Primary Responsibility |
| :--- | :---: | :--- |
| **`NEURAL_SEMANTIC`** | `17` | Intercepted threats across official dataset vectors |
| **`HEURISTIC_REGEX`** | `14` | Intercepted threats across official dataset vectors |

## Reproduction
```bash
# Run full dataset evaluation against official corpora
python benchmark/eval_full_datasets.py --dataset all
```