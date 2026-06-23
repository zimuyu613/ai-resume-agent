import re
from typing import Any


TECH_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Python": ("python",),
    "RAG": ("rag", "retrieval augmented generation"),
    "LLM": ("llm", "large language model"),
    "大模型": ("大模型", "大型语言模型"),
    "Agent": ("agent", "智能体"),
    "AI": ("ai", "人工智能"),
    "Streamlit": ("streamlit",),
    "FastAPI": ("fastapi",),
    "ChromaDB": ("chromadb", "chroma"),
    "embedding": ("embedding", "嵌入"),
    "向量数据库": ("向量数据库", "向量库"),
    "Prompt": ("prompt", "提示词"),
    "API": ("api", "接口"),
    "NLP": ("nlp", "自然语言处理"),
    "数据分析": ("数据分析",),
    "项目经验": ("项目经验", "项目经历"),
    "Tool Calling": ("tool calling", "工具调用"),
    "Trace": ("trace", "链路追踪"),
    "Eval": ("eval", "评测"),
}

ENGLISH_STOPWORDS = {
    "and", "are", "for", "from", "have", "into", "job", "our", "the",
    "this", "using", "with", "work", "岗位", "要求",
}

DEFAULT_SECTION_BONUSES = {
    "skills": 1.5,
    "project_experience": 1.5,
    "internship_experience": 1.0,
}


def extract_keywords(text: str) -> list[str]:
    """Extract deterministic Chinese tech terms and useful English tokens."""
    normalized = (text or "").lower()
    keywords: list[str] = []

    for canonical, aliases in TECH_KEYWORDS.items():
        if any(alias.lower() in normalized for alias in aliases):
            keywords.append(canonical)

    english_tokens = re.findall(r"[a-z][a-z0-9_+.-]{2,}", normalized)
    known_aliases = {
        alias.lower()
        for aliases in TECH_KEYWORDS.values()
        for alias in aliases
        if re.fullmatch(r"[a-z][a-z0-9_+.-]*", alias.lower())
    }
    for token in english_tokens:
        if token in ENGLISH_STOPWORDS or token in known_aliases:
            continue
        if token not in {keyword.lower() for keyword in keywords}:
            keywords.append(token)

    return keywords


def score_chunk(
    chunk: dict[str, Any],
    jd_keywords: list[str],
    preferred_sections: dict[str, float] | list[str] | None = None,
) -> dict[str, Any]:
    """Return explainable rule-based scores for one retrieved chunk."""
    text = str(chunk.get("text", chunk.get("content", "")))
    normalized_text = text.lower()
    keyword_hits = [keyword for keyword in jd_keywords if keyword.lower() in normalized_text]
    keyword_overlap_score = len(keyword_hits)

    if preferred_sections is None:
        section_bonuses = DEFAULT_SECTION_BONUSES
    elif isinstance(preferred_sections, dict):
        section_bonuses = preferred_sections
    else:
        section_bonuses = {section: 1.0 for section in preferred_sections}
    section_bonus = float(section_bonuses.get(str(chunk.get("section", "unknown")), 0.0))

    distance = chunk.get("distance")
    if distance is None:
        distance_score = 0.0
    else:
        distance_score = 1.0 / (1.0 + max(float(distance), 0.0))

    final_score = keyword_overlap_score * 2.0 + section_bonus + distance_score
    return {
        "keyword_overlap_score": keyword_overlap_score,
        "keyword_hits": keyword_hits,
        "section_bonus": round(section_bonus, 4),
        "distance_score": round(distance_score, 4),
        "rerank_score": round(final_score, 4),
    }


def rerank_chunks(
    chunks: list[dict[str, Any]],
    job_description: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Score, sort and return top_k chunks while preserving source fields."""
    if not chunks:
        return []

    jd_keywords = extract_keywords(job_description)
    scored_chunks = []
    for original_rank, chunk in enumerate(chunks, start=1):
        scored_chunks.append(
            {
                **chunk,
                **score_chunk(chunk, jd_keywords),
                "original_rank": original_rank,
            }
        )

    scored_chunks.sort(
        key=lambda item: (
            item["rerank_score"],
            item["keyword_overlap_score"],
            -item["original_rank"],
        ),
        reverse=True,
    )
    return scored_chunks[:max(top_k, 0)]
