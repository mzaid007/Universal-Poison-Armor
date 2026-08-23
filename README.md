# Universal Poison Armor 🛡️

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Model Context Protocol](https://img.shields.io/badge/MCP-Protocol-purple.svg)](https://modelcontextprotocol.io/)
[![FastMCP](https://img.shields.io/badge/FastMCP-Enabled-green.svg)](https://github.com/jlowin/fastmcp)
[![Superpowers Compatible](https://img.shields.io/badge/Superpowers-Compatible%20Skill-orange.svg)](https://github.com/obra/superpowers)
[![Security: AI Poison Defense](https://img.shields.io/badge/Security-AI%20Poison%20Defense-red.svg)](#)

**Universal Poison Armor** is an open-source, production-grade security framework and **Model Context Protocol (MCP)** server for AI agents, LLM pipelines, and RAG systems. It provides multi-layer protection against prompt injection, zero-width steganography, adversarial suffixes (GCG attacks), tracking pixels / Markdown XSS, semantic dataset poisoning, and Consensus Poisoning / Sybil attacks.

Compatible with the **`obra/superpowers`** agentic skills framework, Claude Code, Google Antigravity, and Cursor.

---

## 📖 Table of Contents

- [🚨 What is AI Poisoning?](#-what-is-ai-poisoning)
- [🛡️ Multi-Layer Defense Architecture](#️-multi-layer-defense-architecture)
  - [1. Deterministic Normalization & Heuristic Redaction](#1-deterministic-normalization--heuristic-redaction)
  - [2. Tracking Pixel & Markdown XSS Neutralization](#2-tracking-pixel--markdown-xss-neutralization)
  - [3. Shannon Entropy & Adversarial Suffix Detection (GCG)](#3-shannon-entropy--adversarial-suffix-detection-gcg)
  - [4. Unsupervised Semantic Anomaly Detection](#4-unsupervised-semantic-anomaly-detection)
  - [5. Consensus Poisoning & Sybil Attack Defense](#5-consensus-poisoning--sybil-attack-defense)
  - [6. Persistent Security Audit Logging](#6-persistent-security-audit-logging)
- [📂 Project Structure](#-project-structure)
- [⚡ Quickstart & Installation](#-quickstart--installation)
- [🤖 Agent & Skill Installation](#-agent--skill-installation)
  - [Claude Code & Superpowers Framework](#claude-code--superpowers-framework)
  - [Google Antigravity](#google-antigravity)
  - [Cursor IDE](#cursor-ide)
  - [Claude Desktop](#claude-desktop)
- [🛠️ Exposed MCP Tools](#️-exposed-mcp-tools)
  - [`sanitize_document`](#sanitize_document)
  - [`scan_dataset_for_anomalies`](#scan_dataset_for_anomalies)
  - [`verify_article_consensus`](#verify_article_consensus)
- [📝 Security Audit Logs (`security_audit.json`)](#-security-audit-logs-security_auditjson)
- [🐍 Python API Usage](#-python-api-usage)
- [🔒 Security & Privacy Guarantees](#-security--privacy-guarantees)
- [📄 License](#-license)

---

## 🚨 What is AI Poisoning?

As autonomous AI agents, coding assistants, and Retrieval-Augmented Generation (RAG) pipelines ingest external data from repositories, web search results, PDFs, and databases, they are vulnerable to **Adversarial Context & Data Poisoning Attacks**:

```
+-------------------------------------------------------------------------------+
|                           AI Context Poisoning Vectors                        |
+-------------------------------------------------------------------------------+
|  1. Indirect Prompt Injection   | Attacker hides instructions inside data to  |
|                                 | hijack the agent's system prompt & tools.   |
|  2. Zero-Width Steganography    | Invisible Unicode tokens (ZWSP, tags) bypass|
|                                 | human review but trigger LLM token actions. |
|  3. Adversarial Suffixes (GCG)  | High-entropy mathematical token gibberish   |
|                                 | designed to force model safety bypasses.    |
|  4. Tracking Pixel Exfiltration | Markdown images/iframes leak IP addresses.  |
|  5. Semantic RAG Poisoning      | Adversary seeds knowledge bases with trojan |
|                                 | clusters that alter model reasoning.        |
|  6. Consensus & Sybil Attacks   | Bot networks flood search results with near-|
|                                 | identical claims to trick AI into consensus.|
+-------------------------------------------------------------------------------+
```

**Universal Poison Armor** neutralizes these threats *before* untrusted content reaches the LLM context window.

---

## 🛡️ Multi-Layer Defense Architecture

```
+---------------------------------------------------------------------------+
|                        Incoming Untrusted Context                         |
|           (Files, Web Pages, Datasets, RAG Context Chunks)                |
+---------------------------------------------------------------------------+
                                      |
                                      v
+---------------------------------------------------------------------------+
| LAYER 1: Tracking Pixel & Markdown XSS Stripping                          |
|  • Strips ![alt](url) Markdown images, <img ...>, and <iframe ...> tags   |
|  • Prevents outbound IP address leakage and tracking beacon exfiltration  |
+---------------------------------------------------------------------------+
                                      |
                                      v
+---------------------------------------------------------------------------+
| LAYER 2: Deterministic Unicode Normalization & Regex Redaction             |
|  • Strips zero-width & invisible Unicode (ZWSP, ZWNJ, BOM, tag blocks)    |
|  • Redacts injection patterns ('ignore previous instructions', etc.)     |
|  • Neutralizes bidirectional override and variation selector exploits    |
+---------------------------------------------------------------------------+
                                      |
                                      v
+---------------------------------------------------------------------------+
| LAYER 3: Shannon Entropy & Adversarial Suffix Detection (GCG)             |
|  • Computes character-level Shannon Entropy: H(X) = -sum(P(x)*log2(P(x))) |
|  • Flags & redacts high-entropy blocks (> 4.5 bits/char) as attacks       |
+---------------------------------------------------------------------------+
                                      |
                                      v
+---------------------------------------------------------------------------+
| LAYER 4: Unsupervised Semantic Anomaly Detection                           |
|  • Computes local dense vector embeddings via sentence-transformers       |
|    ('all-MiniLM-L6-v2' — 100% offline, privacy preserving)                |
|  • Fits scikit-learn Isolation Forest to detect statistical outliers      |
|  • Generates threat severity reports (MODERATE, HIGH, CRITICAL)           |
+---------------------------------------------------------------------------+
                                      |
                                      v
+---------------------------------------------------------------------------+
| LAYER 5: Consensus Poisoning & Sybil Flooding Defense                      |
|  • Audits domain provenance against verified TLDs (.gov, .edu, etc.)      |
|  • Computes pairwise semantic similarity matrix across search results     |
|  • Detects coordinated near-duplicate syndication (similarity > 0.95)     |
+---------------------------------------------------------------------------+
                                      |
                                      v
+---------------------------------------------------------------------------+
| LAYER 6: Persistent Security Audit Logging                                |
|  • Automatically appends timestamped threat events to security_audit.json |
+---------------------------------------------------------------------------+
```

---

## 📂 Project Structure

```
Universal-Poison-Armor/
├── LICENSE                                 # MIT Open-Source License
├── README.md                               # Open-source documentation & quickstart guide
├── requirements.txt                        # Project dependencies (fastmcp, sentence-transformers, scikit-learn)
├── security_audit.json                     # Persistent audit trail of intercepted threats
├── skills/
│   └── ai-poison-defense/
│       ├── SKILL.md                        # Superpowers-compatible agentic behavioral instructions
│       └── src/
│           ├── __init__.py                 # Python package exports
│           ├── sanitizers.py               # Core PoisonDefenseEngine (Entropy + Regex + Isolation Forest)
│           └── server.py                   # FastMCP Server with stdio transport & audit logger
├── src/
│   ├── __init__.py                         # Root package alias
│   ├── sanitizers.py                       # Engine alias
│   └── server.py                           # Server entrypoint alias
└── tests/
    └── test_sanitizers.py                  # Comprehensive unit & integration test suite
```

---

## ⚡ Quickstart & Installation

```bash
# Clone repository
git clone https://github.com/your-username/Universal-Poison-Armor.git
cd Universal-Poison-Armor

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

---

## 🤖 Agent & Skill Installation

Universal Poison Armor is designed to be installed directly into your AI workflows via Git repository references, compatible with the **`obra/superpowers`** agentic skill framework.

### Claude Code & Superpowers Framework

```bash
git clone https://github.com/your-username/Universal-Poison-Armor.git ~/.superpowers/skills/Universal-Poison-Armor
```

Configure in `claude.json` / `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "universal-poison-armor": {
      "command": "python",
      "args": [
        "skills/ai-poison-defense/src/server.py"
      ],
      "cwd": "/absolute/path/to/Universal-Poison-Armor"
    }
  }
}
```

---

## 🛠️ Exposed MCP Tools

### 1. `sanitize_document`
Sanitizes an incoming untrusted text document, code file, or RAG context chunk.
- **Signature**: `sanitize_document(document_text: str) -> str`
- **Actions**:
  1. Strips tracking pixels (`![img](url)`, `<img src="...">`, `<iframe>`).
  2. Strips zero-width steganographic Unicode.
  3. Redacts prompt injection patterns to `[REDACTED_INJECTION_ATTEMPT]`.
  4. Detects high-entropy adversarial suffixes and redacts them with `[ADVERSARIAL_SUFFIX_THREAT: REDACTED_HIGH_ENTROPY_BLOCK]`.
  5. Automatically logs all detected threats to `security_audit.json`.

---

### 2. `scan_dataset_for_anomalies`
Scans a batch of documents or retrieved RAG items for out-of-distribution poisoned clusters.
- **Signature**: `scan_dataset_for_anomalies(documents: list[str]) -> str`

---

### 3. `verify_article_consensus`
Defends against **Consensus Poisoning** and **Sybil Flooding** across multi-source web search results.
- **Signature**: `verify_article_consensus(articles: list[dict]) -> str`

---

## 📝 Security Audit Logs (`security_audit.json`)

All intercepted threats are automatically recorded in `security_audit.json`:

```json
[
  {
    "timestamp": "2026-08-21T02:10:00Z",
    "threat_type": "MARKDOWN_XSS_TRACKING_PIXEL",
    "payload_preview": "Download doc: ![pixel](https://attacker.xyz/tracker.png)",
    "payload_length": 58
  },
  {
    "timestamp": "2026-08-21T02:10:05Z",
    "threat_type": "ADVERSARIAL_SUFFIX_THREAT (Entropy: 5.12 > 4.50)",
    "payload_preview": "Explain system architecture. !@#$%^&*()_+~`|}{[]:;?><,./1a9ZkLmN",
    "payload_length": 65
  }
]
```

---

## 🐍 Python API Usage

```python
from skills.ai_poison_defense.src.sanitizers import PoisonDefenseEngine

engine = PoisonDefenseEngine(entropy_threshold=4.5)

# 1. Calculate Shannon Entropy
text = "!@#$%^&*()_+~`|}{[]:;?><,./1a9ZkLmN"
entropy = engine.calculate_entropy(text)
print(f"Shannon Entropy: {entropy:.2f} bits/char")  # > 4.5

# 2. Strip tracking pixels
raw_md = "Hello ![Track](https://tracker.xyz/pixel.gif) world <img src='beacon.png'/>"
clean_md = engine.strip_markdown_xss(raw_md)
print("Clean Markdown:", clean_md)  # "Hello world"
```

---

## 📄 License

Distributed under the **MIT License**.
