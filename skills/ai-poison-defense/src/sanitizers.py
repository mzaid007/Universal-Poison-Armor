"""
Sanitizers and Poison Defense Engine
====================================
This module provides the core defense algorithms for Universal Poison Armor.
It combines deterministic Unicode normalization & regular expression redaction,
Markdown XSS & tracking pixel neutralization, Shannon Entropy adversarial suffix detection,
unsupervised semantic anomaly detection using sentence embeddings and Isolation Forests,
and Consensus Poisoning / Sybil Attack defense across multi-source web corpora.

Key Capabilities:
1. Zero-width Unicode stripping (neutralizes invisible steganography & token-smuggling).
2. Heuristic prompt injection redaction (neutralizes override phrases and jailbreak templates).
3. Markdown XSS & Tracking Pixel stripping (removes ![img](url), <img>, and <iframe>).
4. Shannon Entropy calculation & Adversarial Suffix detection (e.g., GCG attacks > 4.5 entropy).
5. Dense vector semantic embeddings via SentenceTransformers ('all-MiniLM-L6-v2').
6. Hybrid unsupervised anomaly / outlier detection via IsolationForest + Centroid Distance Metrics.
7. Consensus Poisoning & Sybil Attack defense (TLD trust analysis + near-duplicate semantic flood detection).
"""

from __future__ import annotations

from collections import Counter
import logging
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import IsolationForest

# Configure structured logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class PoisonDefenseEngine:
    """
    Production-grade defense engine for neutralizing AI poisoning, prompt injections,
    adversarial suffixes (GCG), tracking pixels / Markdown XSS, semantic dataset contamination,
    and Consensus Poisoning / Sybil attacks in LLM workflows.

    Attributes:
        model_name (str): SentenceTransformers model identifier used for semantic embeddings.
        model (SentenceTransformer): Loaded local embedding model.
        contamination (Union[float, str]): Outlier contamination factor or 'auto'.
        random_state (int): Seed for deterministic anomaly detection.
        entropy_threshold (float): Shannon entropy threshold above which text is flagged as adversarial (default: 4.5).
    """

    # Comprehensive regular expression targeting invisible, zero-width, and directional Unicode characters
    # frequently utilized in steganographic prompt injection and LLM token-evasion attacks.
    ZERO_WIDTH_PATTERN = re.compile(
        r"["
        r"\u200B-\u200D"          # Zero-Width Space (ZWSP), Zero-Width Non-Joiner (ZWNJ), Zero-Width Joiner (ZWJ)
        r"\uFEFF"                  # Zero-Width No-Break Space / Byte Order Mark (BOM)
        r"\u200E\u200F"            # Left-to-Right Mark (LRM) and Right-to-Left Mark (RLM)
        r"\u202A-\u202E"           # Directional formatting overrides (LRE, RLE, PDF, LRO, RLO)
        r"\u2060-\u2064"           # Word Joiner, Invisible Times, Invisible Separator
        r"\u2066-\u2069"           # Directional isolate controls (LRI, RLI, FSI, PDI)
        r"\u00AD"                  # Soft Hyphen (used to break keyword tokenizers)
        r"\u180E"                  # Mongolian Vowel Separator (invisible glyph)
        r"\uFE00-\uFE0F"           # Variation Selectors (can alter token parsing invisibly)
        r"\U000E0000-\U000E007F"  # Unicode Tags Block (steganographic character hiding)
        r"]+",
        re.UNICODE,
    )

    # Standardized redaction marker for detected prompt injection attempts
    REDACTION_MARKER = "[REDACTED_INJECTION_ATTEMPT]"

    # Standardized marker for detected adversarial suffix threats (e.g. GCG gibberish)
    ADVERSARIAL_SUFFIX_MARKER = "[ADVERSARIAL_SUFFIX_THREAT: REDACTED_HIGH_ENTROPY_BLOCK]"

    # Heuristic pattern signatures targeting common prompt injection, role-play overrides, and jailbreak vectors
    INJECTION_PATTERNS = [
        # Instruction override directives
        re.compile(
            r"(?i)\b(?:ignore|disregard|forget|override|bypass|cancel|drop|negate)\s+"
            r"(?:all\s+|any\s+|the\s+|prior\s+|earlier\s+|above\s+)*"
            r"(?:previous\s+|prior\s+|earlier\s+|above\s+)*"
            r"(?:instructions|directions|prompts|rules|commands|constraints|directives|guidelines)\b"
        ),
        # System prompt leakage and role overrides
        re.compile(
            r"(?i)\b(?:system\s+prompt|system\s+instructions|system\s+directive|developer\s+instructions|core\s+prompt)\b"
        ),
        # Role-play & jailbreak persona switches (e.g. DAN, GodMode, EvilBot)
        re.compile(
            r"(?i)\b(?:you\s+are\s+now|act\s+as|pretend\s+to\s+be|roleplay\s+as)\s+"
            r"(?:an?\s+)?(?:unrestricted|jailbroken|unfiltered|DAN|evil|godmode|developer\s+mode|unaligned)\b"
        ),
        # Unconstrained execution triggers
        re.compile(
            r"(?i)\b(?:do\s+anything\s+now|DAN\s+mode|jailbreak\s+mode|unlimited\s+mode|sudo\s+mode)\b"
        ),
        # Secret extraction and configuration exfiltration
        re.compile(
            r"(?i)\b(?:reveal|print|display|output|show|leak|echo|dump)\s+"
            r"(?:the\s+|your\s+|all\s+)?(?:secret|initial|hidden|system|developer|master)\s+"
            r"(?:prompt|instructions|context|tokens|keys|passwords)\b"
        ),
        # Artificial high-priority rules injected inside untrusted data
        re.compile(
            r"(?i)\b(?:new\s+rule|special\s+directive|priority\s+override|mandatory\s+instruction)\s*:\s*(?:you\s+must|always|never|ignore)\b"
        ),
        # Pseudo-XML / HTML system tag injection attacks
        re.compile(
            r"(?i)<\s*(?:script|system_override|admin_command|prompt_injection|system|developer)\s*>.*?</\s*(?:script|system_override|admin_command|prompt_injection|system|developer)\s*>"
        ),
    ]

    # Markdown XSS, tracking pixel, and embedded iframe patterns
    MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\([^)]+\)|!\[[^\]]*\]\[[^\]]*\]")
    HTML_IMG_PATTERN = re.compile(r"(?i)<img\b[^>]*\/?>", re.DOTALL)
    HTML_IFRAME_PATTERN = re.compile(r"(?i)<iframe\b[^>]*>.*?</iframe>|<iframe\b[^>]*\/?>", re.DOTALL)

    # Baseline set of verified, authoritative Top-Level Domains (TLDs) and official government/academic suffixes
    TRUSTED_TLD_SUFFIXES: Set[str] = {
        ".gov",
        ".edu",
        ".mil",
        ".int",
        ".gov.uk",
        ".ac.uk",
        ".gov.au",
        ".edu.au",
        ".gov.ca",
        ".gc.ca",
        ".europa.eu",
        ".who.int",
        ".un.org",
    }

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        contamination: Union[float, str] = 0.1,
        random_state: int = 42,
        entropy_threshold: float = 4.5,
    ) -> None:
        """
        Initialize the Poison Defense Engine with local embedding model, Isolation Forest,
        and Shannon Entropy threshold settings.

        Args:
            model_name: HuggingFace sentence-transformers model identifier or local path.
                        Defaults to 'all-MiniLM-L6-v2' (lightweight, fast, 384-dimensional).
            contamination: Expected proportion of outliers (default: 0.1).
            random_state: Random state seed for deterministic, reproducible results.
            entropy_threshold: Shannon entropy cutoff above which text blocks are flagged as adversarial suffixes (default: 4.5).
        """
        self.model_name = model_name
        self.contamination = contamination
        self.random_state = random_state
        self.entropy_threshold = entropy_threshold

        logger.info(
            "Initializing PoisonDefenseEngine with embedding model: '%s' and entropy_threshold: %0.2f...",
            self.model_name,
            self.entropy_threshold,
        )

        # Load local embedding model (cached locally; runs offline without third-party API dependencies)
        self.model = SentenceTransformer(self.model_name)

        # Initialize base Isolation Forest
        self.detector = IsolationForest(
            contamination=self.contamination if isinstance(self.contamination, (int, float)) else "auto",
            random_state=self.random_state,
            n_estimators=100,
        )
        logger.info("PoisonDefenseEngine initialized successfully.")

    def calculate_entropy(self, text: str) -> float:
        """
        Calculates the Shannon entropy of the character distribution in a string.

        Formula: H(X) = -sum(P(x) * log2(P(x)))

        Natural language typically exhibits entropy between 3.0 and 4.2.
        Unnaturally high entropy (> 4.5) indicates high-variance gibberish,
        pseudorandom token sequences, or adversarial suffixes (e.g. GCG attacks).

        Args:
            text: The input string to evaluate.

        Returns:
            Shannon entropy as a float value in bits per character.
        """
        if not text:
            return 0.0

        length = len(text)
        if length == 0:
            return 0.0

        counts = Counter(text)
        entropy = -sum(
            (count / length) * math.log2(count / length)
            for count in counts.values()
            if count > 0
        )
        return float(round(entropy, 4))

    def strip_markdown_xss(self, text: str) -> str:
        """
        Strips Markdown image tags, HTML <img> tags, and <iframe> tags to prevent
        tracking pixel IP leaks, SSRF, and cross-site scripting attacks in agent renderers.

        Targets:
        1. Markdown image syntax: `![alt](url)` and `![alt][ref]`
        2. HTML Image tags: `<img src="..." />`
        3. HTML iframe tags: `<iframe src="..."></iframe>`

        Args:
            text: Input raw text or Markdown string.

        Returns:
            Sanitized text with all image tags and iframes completely removed.
        """
        if not text:
            return ""

        # Remove markdown images
        sanitized = self.MARKDOWN_IMAGE_PATTERN.sub("", text)
        # Remove HTML <img> tags
        sanitized = self.HTML_IMG_PATTERN.sub("", sanitized)
        # Remove HTML <iframe> tags
        sanitized = self.HTML_IFRAME_PATTERN.sub("", sanitized)

        return sanitized.strip()

    def strip_injections(self, text: str) -> str:
        """
        Sanitizes a document or prompt by stripping invisible/zero-width Unicode
        characters, redacting known prompt injection attack phrases, and detecting
        adversarial suffix attacks via Shannon Entropy analysis.

        Workflow:
        1. Strips all zero-width and invisible steganographic Unicode sequences.
        2. Applies compiled regex patterns targeting known prompt injection phrases.
        3. Analyzes Shannon Entropy of distinct blocks and tokens. If an adversarial
           suffix or high-entropy block (> 4.5) is detected, it is redacted with
           `[ADVERSARIAL_SUFFIX_THREAT: REDACTED_HIGH_ENTROPY_BLOCK]`.
        4. Cleans and normalizes redundant whitespace and repeated markers.

        Args:
            text: The raw input string to sanitize.

        Returns:
            Sanitized and safe string with harmful characters stripped,
            injection phrases replaced with `[REDACTED_INJECTION_ATTEMPT]`,
            and high-entropy adversarial blocks replaced with `[ADVERSARIAL_SUFFIX_THREAT: REDACTED_HIGH_ENTROPY_BLOCK]`.
        """
        if not text:
            return ""

        # Step 1: Strip zero-width, invisible, and malicious directional Unicode characters
        sanitized = self.ZERO_WIDTH_PATTERN.sub("", text)

        # Step 2: Redact common prompt injection patterns
        for pattern in self.INJECTION_PATTERNS:
            sanitized = pattern.sub(self.REDACTION_MARKER, sanitized)

        # Step 3: Adversarial Suffix & High-Entropy Detection
        # Check individual lines, blocks, and tokens to isolate adversarial suffix payloads
        lines = sanitized.splitlines()
        processed_lines = []

        for line in lines:
            stripped_line = line.strip()
            if not stripped_line:
                processed_lines.append(line)
                continue

            # If the entire line is a high-entropy string (> 4.5) with no natural spaces/words
            if len(stripped_line) >= 20 and " " not in stripped_line and self.calculate_entropy(stripped_line) > self.entropy_threshold:
                logger.warning(
                    "Adversarial high-entropy line detected (Entropy: %0.2f > %0.2f): %s",
                    self.calculate_entropy(stripped_line),
                    self.entropy_threshold,
                    stripped_line[:60],
                )
                processed_lines.append(self.ADVERSARIAL_SUFFIX_MARKER)
                continue

            # Check individual space-separated tokens in the line for embedded high-entropy suffixes
            tokens = stripped_line.split(" ")
            sanitized_tokens = []
            has_high_entropy_token = False

            for token in tokens:
                # If a single token has length >= 18 and entropy > 4.5 (e.g. GCG suffix string)
                if len(token) >= 18 and self.calculate_entropy(token) > self.entropy_threshold:
                    sanitized_tokens.append(self.ADVERSARIAL_SUFFIX_MARKER)
                    has_high_entropy_token = True
                else:
                    sanitized_tokens.append(token)

            if has_high_entropy_token:
                processed_lines.append(" ".join(sanitized_tokens))
            else:
                # Check multi-token trailing suffix (e.g. last 4 tokens combined)
                if len(tokens) >= 3:
                    trailing_suffix = " ".join(tokens[-3:])
                    if len(trailing_suffix) >= 20 and self.calculate_entropy(trailing_suffix) > self.entropy_threshold:
                        prefix = " ".join(tokens[:-3])
                        processed_lines.append(f"{prefix} {self.ADVERSARIAL_SUFFIX_MARKER}".strip())
                        continue
                processed_lines.append(line)

        sanitized = "\n".join(processed_lines)

        # Step 4: Collapse repeated redaction markers to keep text clean and readable
        repeated_marker_pattern = re.compile(
            rf"(?:{re.escape(self.REDACTION_MARKER)}\s*){{2,}}"
        )
        sanitized = repeated_marker_pattern.sub(f"{self.REDACTION_MARKER} ", sanitized)

        repeated_suffix_pattern = re.compile(
            rf"(?:{re.escape(self.ADVERSARIAL_SUFFIX_MARKER)}\s*){{2,}}"
        )
        sanitized = repeated_suffix_pattern.sub(f"{self.ADVERSARIAL_SUFFIX_MARKER} ", sanitized)

        # Normalize remaining trailing and leading whitespace
        return sanitized.strip()

    def detect_semantic_anomalies(
        self, documents: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Identifies semantic anomalies and poisoned clusters across a collection of documents.

        Uses sentence embeddings to project documents into a dense semantic vector space,
        and applies an Isolation Forest + Centroid Outlier analysis to flag documents that
        significantly deviate from the normal corpus distribution.

        Args:
            documents: Array of strings representing documents, RAG context chunks,
                       or dataset samples.

        Returns:
            A list of dictionary records containing anomaly details for flagged outliers:
            [
                {
                    "index": int,
                    "document": str,
                    "anomaly_score": float,  # Lower/negative means more anomalous
                    "is_anomaly": bool,
                    "severity": str          # 'CRITICAL', 'HIGH', or 'MODERATE'
                },
                ...
            ]
        """
        if not documents:
            logger.warning("Empty document list provided for anomaly detection.")
            return []

        # Filter out empty or whitespace-only documents while tracking original indices
        valid_docs_with_indices = [
            (idx, doc) for idx, doc in enumerate(documents) if doc and doc.strip()
        ]

        if not valid_docs_with_indices:
            return []

        indices, valid_docs = zip(*valid_docs_with_indices)
        n_samples = len(valid_docs)

        # Isolation Forest requires at least 3 distinct samples for meaningful statistical clustering
        if n_samples < 3:
            logger.info(
                "Sample size (%d) is too small for statistical anomaly detection.",
                n_samples,
            )
            return []

        # Generate dense semantic embeddings for all valid documents
        logger.info("Computing dense embeddings for %d documents...", n_samples)
        embeddings = self.model.encode(
            list(valid_docs),
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        # Dynamic contamination tuning
        if isinstance(self.contamination, (int, float)):
            effective_contamination = min(max(float(self.contamination), 0.01), 0.5)
        else:
            effective_contamination = "auto"

        # Train a fresh Isolation Forest for the current batch
        batch_detector = IsolationForest(
            contamination=effective_contamination,
            random_state=self.random_state,
            n_estimators=100,
        )

        predictions = batch_detector.fit_predict(embeddings)
        scores = batch_detector.decision_function(embeddings)

        # Calculate centroid and cosine distance statistics
        centroid = np.mean(embeddings, axis=0, keepdims=True)
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-9)
        cosine_similarities = np.dot(embeddings, centroid_norm.T).flatten()
        cosine_distances = 1.0 - cosine_similarities

        mean_dist = float(np.mean(cosine_distances))
        std_dist = float(np.std(cosine_distances))

        anomalies: List[Dict[str, Any]] = []

        for original_idx, doc, pred, score, dist, sim in zip(
            indices, valid_docs, predictions, scores, cosine_distances, cosine_similarities
        ):
            # Check standard deviations from mean distance if variation exists
            z_dist = ((dist - mean_dist) / std_dist) if std_dist > 0.03 else 0.0

            # A document is an anomaly if:
            # 1. IsolationForest flags it (pred == -1) AND score is distinctly negative (< -0.04) AND similarity to centroid is not high (< 0.70)
            # OR
            # 2. Distance z-score >= 1.5 AND absolute similarity to centroid < 0.60
            is_anomaly = False
            if pred == -1 and score < -0.04 and sim < 0.70:
                is_anomaly = True
            elif z_dist >= 1.5 and sim < 0.60:
                is_anomaly = True

            if is_anomaly:
                # Severity categorization based on distance from the decision boundary & z_dist
                combined_score = float(round(score - (0.1 * max(z_dist, 0.0)), 4))

                if combined_score < -0.12 or z_dist > 2.0 or sim < 0.40:
                    severity = "CRITICAL"
                elif combined_score < -0.04 or z_dist > 1.3:
                    severity = "HIGH"
                else:
                    severity = "MODERATE"

                anomalies.append(
                    {
                        "index": int(original_idx),
                        "document": doc,
                        "anomaly_score": combined_score,
                        "similarity_to_centroid": float(round(sim, 4)),
                        "is_anomaly": True,
                        "severity": severity,
                    }
                )

        # Sort anomalies by anomaly_score ascending (most anomalous/dangerous first)
        anomalies.sort(key=lambda x: x["anomaly_score"])

        logger.info(
            "Detected %d semantic anomalies out of %d documents.",
            len(anomalies),
            n_samples,
        )
        return anomalies

    def _extract_domain(self, url: str) -> Tuple[str, bool]:
        """
        Extract the hostname / domain from a URL and evaluate whether it belongs to a trusted TLD.

        Args:
            url: The URL string to evaluate.

        Returns:
            Tuple of (domain_name: str, is_trusted_tld: bool).
        """
        if not url:
            return ("unknown", False)

        raw_url = url.strip()
        if not raw_url.startswith(("http://", "https://")):
            raw_url = "https://" + raw_url

        try:
            parsed = urlparse(raw_url)
            hostname = (parsed.netloc or parsed.path).lower().split(":")[0]
        except Exception:
            hostname = raw_url.lower()

        # Check if the hostname ends with any trusted TLD suffix
        is_trusted = any(
            hostname == tld.lstrip(".") or hostname.endswith(tld)
            for tld in self.TRUSTED_TLD_SUFFIXES
        )

        return (hostname, is_trusted)

    def analyze_consensus_threat(
        self,
        articles: List[Dict[str, Any]],
        similarity_threshold: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Analyzes a batch of web articles or search results to defend against Consensus Poisoning
        and Sybil attacks (coordinated disinformation flooding).

        Logic:
        1. Analyzes each article's URL against a baseline of trusted Top-Level Domains (.gov, .edu, etc.)
           and flags unfamiliar or suspicious domain origins.
        2. Generates dense vector semantic embeddings for each article's text using sentence-transformers.
        3. Computes pairwise cosine similarities across all articles.
        4. Identifies if multiple articles originating from untrusted/suspicious domains share a near-identical
           semantic similarity score (above 0.95), indicating a synthetic, manufactured Sybil flood.

        Args:
            articles: List of dictionaries, each containing at least:
                      - 'url' (str): The origin URL of the article.
                      - 'text' (str): The article body or summary text.
                      - (optional) 'title' (str): Article headline.
            similarity_threshold: Cosine similarity cutoff for near-identical text (default: 0.95).

        Returns:
            A structured assessment dictionary:
            {
                "is_sybil_attack": bool,
                "threat_level": str,  # 'CRITICAL', 'HIGH', 'MODERATE', 'LOW', 'NONE'
                "threat_type": str,   # 'Coordinated Flooding / Sybil Attack' or 'Clean Consensus'
                "total_articles": int,
                "untrusted_count": int,
                "trusted_count": int,
                "flagged_clusters": list[dict],
                "domain_evaluations": list[dict],
                "summary": str
            }
        """
        if not articles:
            logger.warning("Empty articles list provided for consensus threat analysis.")
            return {
                "is_sybil_attack": False,
                "threat_level": "NONE",
                "threat_type": "No Data",
                "total_articles": 0,
                "untrusted_count": 0,
                "trusted_count": 0,
                "flagged_clusters": [],
                "domain_evaluations": [],
                "summary": "No articles provided for consensus analysis.",
            }

        # Step 1: Evaluate domain origins and filter valid text entries
        domain_evaluations: List[Dict[str, Any]] = []
        valid_entries: List[Tuple[int, Dict[str, Any], str, str, bool]] = []

        untrusted_count = 0
        trusted_count = 0

        for idx, art in enumerate(articles):
            if not isinstance(art, dict):
                continue

            url = str(art.get("url", "")).strip()
            text = str(art.get("text", "")).strip()
            title = str(art.get("title", "")).strip()

            domain, is_trusted = self._extract_domain(url)

            if is_trusted:
                trusted_count += 1
                trust_status = "TRUSTED_TLD"
            else:
                untrusted_count += 1
                trust_status = "UNTRUSTED_DOMAIN"

            eval_record = {
                "index": idx,
                "url": url,
                "domain": domain,
                "is_trusted_tld": is_trusted,
                "trust_status": trust_status,
                "title": title,
            }
            domain_evaluations.append(eval_record)

            if text:
                valid_entries.append((idx, art, domain, text, is_trusted))

        total_valid = len(valid_entries)

        # If fewer than 2 articles have text, pairwise consensus comparison is not applicable
        if total_valid < 2:
            return {
                "is_sybil_attack": False,
                "threat_level": "LOW" if untrusted_count > 0 else "NONE",
                "threat_type": "Insufficient Corpus",
                "total_articles": len(articles),
                "untrusted_count": untrusted_count,
                "trusted_count": trusted_count,
                "flagged_clusters": [],
                "domain_evaluations": domain_evaluations,
                "summary": f"Scanned {len(articles)} article(s). Insufficient data for multi-source consensus comparison.",
            }

        # Step 2: Compute dense semantic embeddings for all valid articles
        texts = [entry[3] for entry in valid_entries]
        logger.info("Computing dense embeddings for %d articles in consensus analysis...", total_valid)
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        # Step 3: Compute pairwise cosine similarity matrix
        similarity_matrix = np.dot(embeddings, embeddings.T)

        # Step 4: Detect Sybil / Coordinated Flooding clusters
        flagged_pairs: List[Dict[str, Any]] = []
        cluster_map: Dict[int, Set[int]] = {}

        for i in range(total_valid):
            for j in range(i + 1, total_valid):
                sim = float(similarity_matrix[i][j])
                idx_i, art_i, dom_i, text_i, trusted_i = valid_entries[i]
                idx_j, art_j, dom_j, text_j, trusted_j = valid_entries[j]

                # Check if similarity is above threshold
                if sim >= similarity_threshold:
                    is_suspicious_pair = False

                    if not trusted_i or not trusted_j:
                        is_suspicious_pair = True
                    elif dom_i != dom_j:
                        is_suspicious_pair = True

                    if is_suspicious_pair:
                        flagged_pairs.append({
                            "article_a_index": idx_i,
                            "article_b_index": idx_j,
                            "domain_a": dom_i,
                            "domain_b": dom_j,
                            "url_a": art_i.get("url", ""),
                            "url_b": art_j.get("url", ""),
                            "similarity": float(round(sim, 4)),
                            "is_untrusted_source": (not trusted_i) or (not trusted_j),
                        })

                        cluster_map.setdefault(idx_i, set()).add(idx_j)
                        cluster_map.setdefault(idx_j, set()).add(idx_i)

        # Construct distinct cluster groups from connected components
        visited: Set[int] = set()
        flagged_clusters: List[Dict[str, Any]] = []
        cluster_id = 1

        for idx in cluster_map:
            if idx not in visited:
                component: Set[int] = set()
                queue = [idx]
                visited.add(idx)

                while queue:
                    curr = queue.pop()
                    component.add(curr)
                    for neighbor in cluster_map.get(curr, set()):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)

                cluster_indices = sorted(list(component))
                cluster_articles = [articles[i] for i in cluster_indices if i < len(articles)]
                cluster_domains = [domain_evaluations[i]["domain"] for i in cluster_indices if i < len(domain_evaluations)]
                untrusted_in_cluster = [d for d, i in zip(cluster_domains, cluster_indices) if not domain_evaluations[i]["is_trusted_tld"]]

                max_cluster_sim = max(
                    [p["similarity"] for p in flagged_pairs if p["article_a_index"] in component and p["article_b_index"] in component]
                    or [similarity_threshold]
                )

                flagged_clusters.append({
                    "cluster_id": cluster_id,
                    "article_indices": cluster_indices,
                    "domains": cluster_domains,
                    "untrusted_domains": untrusted_in_cluster,
                    "max_similarity": float(round(max_cluster_sim, 4)),
                    "cluster_size": len(cluster_indices),
                    "urls": [art.get("url", "") for art in cluster_articles],
                    "excerpt": (cluster_articles[0].get("text", "")[:200] + "...") if cluster_articles else "",
                })
                cluster_id += 1

        # Step 5: Formulate risk verdict
        is_sybil_attack = len(flagged_clusters) > 0 and any(
            len(c["untrusted_domains"]) >= 1 for c in flagged_clusters
        )

        if is_sybil_attack:
            max_cluster_size = max(c["cluster_size"] for c in flagged_clusters)
            if max_cluster_size >= 3 or untrusted_count >= 3:
                threat_level = "CRITICAL"
            else:
                threat_level = "HIGH"
            threat_type = "Coordinated Flooding / Sybil Attack"
            summary = (
                f"🚨 Detected {len(flagged_clusters)} coordinated Sybil / flooding cluster(s) "
                f"across {total_valid} articles. Multiple untrusted domains share near-identical "
                f"(similarity >= {similarity_threshold}) synthetic consensus."
            )
        elif untrusted_count > 0 and trusted_count == 0:
            threat_level = "MODERATE"
            threat_type = "Unverified Origins (No Authoritative TLDs)"
            summary = (
                f"⚠️ Scanned {total_valid} article(s). All sources originate from unverified / non-authoritative domains, "
                f"though no near-duplicate Sybil flood was detected."
            )
        else:
            threat_level = "LOW" if untrusted_count > 0 else "NONE"
            threat_type = "Clean Consensus"
            summary = (
                f"✅ Scanned {total_valid} article(s). No coordinated Sybil flooding or consensus poisoning detected."
            )

        logger.info(
            "Consensus Threat Analysis complete: is_sybil_attack=%s, threat_level=%s",
            is_sybil_attack,
            threat_level,
        )

        return {
            "is_sybil_attack": is_sybil_attack,
            "threat_level": threat_level,
            "threat_type": threat_type,
            "total_articles": len(articles),
            "untrusted_count": untrusted_count,
            "trusted_count": trusted_count,
            "flagged_clusters": flagged_clusters,
            "domain_evaluations": domain_evaluations,
            "summary": summary,
        }
