import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class TraceStep:
    """One observable tool call (or an explicitly skipped tool call)."""

    step_name: str
    tool_name: str
    success: bool
    message: str
    input_summary: dict[str, Any] = field(default_factory=dict)
    output_summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    start_time: str = ""
    end_time: str = ""
    duration_ms: float = 0.0


@dataclass
class WorkflowTrace:
    """Lightweight trace for one fixed Agent Workflow run."""

    run_id: str
    mode: str
    start_time: str
    end_time: str | None
    duration_ms: float | None
    resume_length: int
    job_description_length: int
    top_k: int
    embedding_provider: str | None
    used_rag: bool
    used_fallback: bool | None
    used_rerank: bool = False
    rerank_method: str | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    use_mock_llm: bool = False
    fallback_to_mock: bool = True
    fallback_used: bool | None = None
    original_provider: str | None = None
    provider_error: str | None = None
    review_passed: bool | None = None
    query_refinement_used: bool = False
    retrieval_attempts: int = 1
    steps: list[TraceStep] = field(default_factory=list)
    final_status: str = "running"
    error: str | None = None


def create_run_id() -> str:
    """Create a compact unique ID suitable for filenames and UI display."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{timestamp}_{uuid.uuid4().hex[:8]}"


def now_iso() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def summarize_text(text: str, max_len: int = 120) -> str:
    """Collapse whitespace and truncate text so traces do not store full inputs."""
    normalized = " ".join((text or "").split())
    if len(normalized) <= max_len:
        return normalized
    return normalized[:max_len] + "..."


def summarize_chunks(chunks: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Summarize retrieval results without copying resume chunk text into traces."""
    chunks = chunks or []
    sections = sorted({str(chunk.get("section", "unknown")) for chunk in chunks})
    distances = [
        float(chunk["distance"])
        for chunk in chunks
        if chunk.get("distance") is not None
    ]
    return {
        "retrieved_chunk_count": len(chunks),
        "sections": sections,
        "top_distance_min": min(distances) if distances else None,
        "top_distance_max": max(distances) if distances else None,
    }


def trace_to_dict(trace: WorkflowTrace) -> dict[str, Any]:
    """Convert nested dataclasses into a JSON-serializable dictionary."""
    return asdict(trace)


def save_trace_json(
    trace: WorkflowTrace,
    output_dir: str | Path = "outputs/traces",
) -> str:
    """Save one trace as UTF-8 JSON and return its absolute path."""
    directory = Path(output_dir)
    if not directory.is_absolute():
        directory = Path(__file__).parent / directory
    directory.mkdir(parents=True, exist_ok=True)

    output_path = directory / f"trace_{trace.run_id}.json"
    output_path.write_text(
        json.dumps(trace_to_dict(trace), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(output_path.resolve())
