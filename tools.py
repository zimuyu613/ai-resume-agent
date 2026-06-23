from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

from docx import Document
from pypdf import PdfReader

from agent import call_llm, extract_section, run_agent_workflow
from prompts import COMPREHENSIVE_ANALYSIS_PROMPT, RAG_ANALYSIS_PROMPT
from rag import retrieve_relevant_chunks_with_sources


@dataclass
class ToolResult:
    """Small, interview-friendly return shape for every tool call."""

    success: bool
    tool_name: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _success(tool_name: str, message: str, data: dict[str, Any] | None = None) -> ToolResult:
    return ToolResult(
        success=True,
        tool_name=tool_name,
        message=message,
        data=data or {},
        error=None,
    )


def _failure(tool_name: str, message: str, error: Exception | str) -> ToolResult:
    return ToolResult(
        success=False,
        tool_name=tool_name,
        message=message,
        data={},
        error=str(error),
    )


def _read_txt_bytes(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("gbk")


def _read_pdf_bytes(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(page for page in pages if page.strip()).strip()


def _read_docx_bytes(file_bytes: bytes) -> str:
    document = Document(BytesIO(file_bytes))
    paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs]
    return "\n".join(paragraph for paragraph in paragraphs if paragraph).strip()


def parse_resume_tool(
    file_path: str | Path | None = None,
    resume_text: str | None = None,
) -> ToolResult:
    """Parse resume text from direct input or a local txt/pdf/docx file."""
    tool_name = "parse_resume_tool"

    try:
        if resume_text and resume_text.strip():
            text = resume_text.strip()
            return _success(
                tool_name,
                "Resume text received from direct input.",
                {
                    "resume_text": text,
                    "source_type": "text",
                    "source_name": "direct_input",
                    "char_count": len(text),
                },
            )

        if not file_path:
            return _failure(tool_name, "No resume input provided.", "file_path or resume_text is required")

        path = Path(file_path)
        file_bytes = path.read_bytes()
        suffix = path.suffix.lower()

        if suffix == ".txt":
            text = _read_txt_bytes(file_bytes)
        elif suffix == ".pdf":
            text = _read_pdf_bytes(file_bytes)
        elif suffix == ".docx":
            text = _read_docx_bytes(file_bytes)
        else:
            return _failure(tool_name, "Unsupported resume file type.", f"Unsupported suffix: {suffix}")

        text = text.strip()
        if not text:
            return _failure(tool_name, "Resume parsing returned empty text.", "empty resume text")

        return _success(
            tool_name,
            "Resume parsed successfully.",
            {
                "resume_text": text,
                "source_type": suffix.lstrip("."),
                "source_name": path.name,
                "char_count": len(text),
            },
        )

    except Exception as exc:
        return _failure(tool_name, "Resume parsing failed.", exc)


def rag_retrieve_tool(
    resume_text: str,
    job_description: str,
    top_k: int = 5,
    source_name: str = "resume_text",
    section_filter: str | None = None,
) -> ToolResult:
    """Retrieve resume chunks with section-aware metadata for RAG analysis."""
    tool_name = "rag_retrieve_tool"

    try:
        retrieval_result = retrieve_relevant_chunks_with_sources(
            job_description=job_description,
            resume_text=resume_text,
            source_name=source_name,
            top_k=top_k,
            section_filter=section_filter,
        )
        chunks = retrieval_result.get("sources", [])
        return _success(
            tool_name,
            f"Retrieved {len(chunks)} chunk(s).",
            {
                "retrieved_context": retrieval_result.get("context", ""),
                "chunks": chunks,
                "retrieved_chunk_count": len(chunks),
                "embedding_provider": retrieval_result.get("embedding_provider"),
                "rag_top_k": top_k,
                "rag_total_chunks": retrieval_result.get("total_chunks"),
                "rag_section_filter": retrieval_result.get("section_filter"),
                "rag_available_filtered_chunks": retrieval_result.get("available_filtered_chunks"),
            },
        )
    except Exception as exc:
        return _failure(tool_name, "RAG retrieval failed.", exc)


def _split_report(full_report: str) -> dict[str, str]:
    return {
        "job_analysis": extract_section(full_report, "岗位要求分析", "个人能力分析"),
        "resume_analysis": extract_section(full_report, "个人能力分析", "匹配度分析"),
        "match_analysis": extract_section(full_report, "匹配度分析", "简历优化建议"),
        "suggestions": extract_section(full_report, "简历优化建议", None),
        "full_report": full_report,
    }


def llm_match_analysis_tool(
    resume_text: str,
    job_description: str,
    retrieved_chunks: list[dict[str, Any]] | None = None,
    use_rag: bool = True,
    llm_callable: Callable[[str], str] | None = None,
) -> ToolResult:
    """Run normal LLM analysis or RAG-grounded LLM analysis."""
    tool_name = "llm_match_analysis_tool"

    try:
        if use_rag and retrieved_chunks:
            context_parts = []
            for index, chunk in enumerate(retrieved_chunks, start=1):
                text = (chunk.get("text") or "").strip()
                if not text:
                    continue
                context_parts.append(
                    f"[RAG chunk {index} | section={chunk.get('section', 'unknown')} "
                    f"| chunk_index={chunk.get('chunk_index', index)}]\n{text}"
                )

            prompt = RAG_ANALYSIS_PROMPT.format(
                job_description=job_description[:4000],
                retrieved_context="\n\n".join(context_parts),
            )
            full_report = (llm_callable or call_llm)(prompt)
            analysis = _split_report(full_report)
        elif llm_callable:
            prompt = COMPREHENSIVE_ANALYSIS_PROMPT.format(
                job_description=job_description[:4000],
                resume_text=resume_text[:4000],
            )
            analysis = _split_report(llm_callable(prompt))
        else:
            analysis = run_agent_workflow(job_description, resume_text)

        return _success(
            tool_name,
            "LLM match analysis completed.",
            {
                **analysis,
                "use_rag": use_rag,
                "retrieved_chunk_count": len(retrieved_chunks or []),
            },
        )
    except Exception as exc:
        return _failure(tool_name, "LLM match analysis failed.", exc)


def export_markdown_tool(
    analysis_result: dict[str, Any] | str,
    metadata: dict[str, Any] | None = None,
    output_path: str | Path | None = None,
) -> ToolResult:
    """Build or save a Markdown report from analysis output."""
    tool_name = "export_markdown_tool"

    try:
        metadata = metadata or {}
        if isinstance(analysis_result, str):
            body = analysis_result
        else:
            body = "\n\n".join(
                str(analysis_result.get(key, "")).strip()
                for key in ["job_analysis", "resume_analysis", "match_analysis", "suggestions"]
                if analysis_result.get(key)
            )

        lines = ["# AI Resume Match Report", ""]
        if metadata:
            lines.extend(["## Metadata", ""])
            for key, value in metadata.items():
                lines.append(f"- {key}: {value}")
            lines.append("")

        lines.extend(["## Analysis", "", body.strip()])
        markdown = "\n".join(lines).strip() + "\n"

        saved_path = None
        if output_path:
            path = Path(output_path)
            path.write_text(markdown, encoding="utf-8")
            saved_path = str(path)

        return _success(
            tool_name,
            "Markdown report generated.",
            {
                "markdown": markdown,
                "saved_path": saved_path,
                "char_count": len(markdown),
            },
        )
    except Exception as exc:
        return _failure(tool_name, "Markdown export failed.", exc)
