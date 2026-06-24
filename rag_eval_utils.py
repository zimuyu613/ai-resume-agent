import re
from typing import Any


def normalize_text(text: str) -> str:
    """Normalize case and whitespace for lightweight evidence matching."""
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def extract_chunk_text(chunk: dict[str, Any]) -> str:
    """Read chunk text from the common formats used by retrieval libraries."""
    if not isinstance(chunk, dict):
        return ""
    for key in ("text", "content", "document"):
        value = chunk.get(key)
        if value is not None:
            return str(value)
    return ""


def get_chunk_section(chunk: dict[str, Any]) -> str:
    """Read section from a chunk or its nested metadata."""
    if not isinstance(chunk, dict):
        return "unknown"
    section = chunk.get("section")
    if section:
        return str(section)
    metadata = chunk.get("metadata")
    if isinstance(metadata, dict) and metadata.get("section"):
        return str(metadata["section"])
    return "unknown"


def hit_expected_sections(
    chunks: list[dict[str, Any]],
    expected_sections: list[str],
) -> dict[str, Any]:
    expected = list(dict.fromkeys(str(section) for section in (expected_sections or [])))
    retrieved = {get_chunk_section(chunk) for chunk in (chunks or [])}
    hit_sections = [section for section in expected if section in retrieved]
    missing_sections = [section for section in expected if section not in retrieved]
    return {
        "hit_sections": hit_sections,
        "missing_sections": missing_sections,
        "section_hit_rate": round(len(hit_sections) / len(expected), 4) if expected else 0.0,
    }


def hit_expected_keywords(
    chunks: list[dict[str, Any]],
    expected_keywords: list[str],
) -> dict[str, Any]:
    expected = list(dict.fromkeys(str(keyword) for keyword in (expected_keywords or [])))
    combined_text = normalize_text(" ".join(extract_chunk_text(chunk) for chunk in (chunks or [])))
    hit_keywords = [keyword for keyword in expected if normalize_text(keyword) in combined_text]
    missing_keywords = [keyword for keyword in expected if keyword not in hit_keywords]
    return {
        "hit_keywords": hit_keywords,
        "missing_keywords": missing_keywords,
        "keyword_hit_rate": round(len(hit_keywords) / len(expected), 4) if expected else 0.0,
    }


def _chunk_matches_evidence(chunk: dict[str, Any], evidence: dict[str, Any]) -> tuple[bool, list[str]]:
    expected_section = str(evidence.get("section", ""))
    if expected_section and get_chunk_section(chunk) != expected_section:
        return False, []

    keywords = [str(keyword) for keyword in evidence.get("keywords", [])]
    if not keywords:
        return bool(expected_section), []

    text = normalize_text(extract_chunk_text(chunk))
    hits = [keyword for keyword in keywords if normalize_text(keyword) in text]
    return bool(hits), hits


def evaluate_gold_evidence(
    chunks: list[dict[str, Any]],
    gold_evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    chunks = chunks or []
    evidence_items = [item for item in (gold_evidence or []) if isinstance(item, dict)]
    details = []
    hit_count = 0

    for evidence in evidence_items:
        matched_ranks = []
        matched_keywords = []
        for rank, chunk in enumerate(chunks, start=1):
            matched, keyword_hits = _chunk_matches_evidence(chunk, evidence)
            if matched:
                matched_ranks.append(rank)
                matched_keywords.extend(keyword_hits)

        hit = bool(matched_ranks)
        hit_count += int(hit)
        details.append(
            {
                "section": evidence.get("section", ""),
                "keywords": evidence.get("keywords", []),
                "hit": hit,
                "matched_chunk_ranks": matched_ranks,
                "hit_keywords": list(dict.fromkeys(matched_keywords)),
            }
        )

    return {
        "gold_total": len(evidence_items),
        "gold_hit": hit_count,
        "gold_recall": round(hit_count / len(evidence_items), 4) if evidence_items else 0.0,
        "gold_details": details,
    }


def calculate_recall_at_k(
    chunks: list[dict[str, Any]],
    gold_evidence: list[dict[str, Any]],
    k: int,
) -> float:
    if k <= 0:
        return 0.0
    return evaluate_gold_evidence((chunks or [])[:k], gold_evidence).get("gold_recall", 0.0)


def calculate_mrr(
    chunks: list[dict[str, Any]],
    gold_evidence: list[dict[str, Any]],
) -> float:
    evidence_items = [item for item in (gold_evidence or []) if isinstance(item, dict)]
    if not evidence_items:
        return 0.0
    for rank, chunk in enumerate(chunks or [], start=1):
        if any(_chunk_matches_evidence(chunk, evidence)[0] for evidence in evidence_items):
            return round(1.0 / rank, 4)
    return 0.0


def evaluate_retrieval_result(
    chunks: list[dict[str, Any]],
    expected: dict[str, Any],
    k_values: list[int] | None = None,
) -> dict[str, Any]:
    chunks = chunks or []
    expected = expected or {}
    k_values = k_values or [1, 3, 5]
    gold_evidence = expected.get("gold_evidence", [])
    return {
        "retrieved_count": len(chunks),
        "section_metrics": hit_expected_sections(chunks, expected.get("expected_sections", [])),
        "keyword_metrics": hit_expected_keywords(chunks, expected.get("expected_keywords", [])),
        "gold_metrics": evaluate_gold_evidence(chunks, gold_evidence),
        "recall_at_k": {
            str(k): calculate_recall_at_k(chunks, gold_evidence, k)
            for k in k_values
        },
        "mrr": calculate_mrr(chunks, gold_evidence),
    }
