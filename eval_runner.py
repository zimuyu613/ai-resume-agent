import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Eval must be runnable without model downloads or external API access.
os.environ["EMBEDDING_PROVIDER"] = "local"

from agent_workflow import run_resume_agent_workflow
from tools import rag_retrieve_tool


BASE_DIR = Path(__file__).parent
DEFAULT_CASES_DIR = BASE_DIR / "eval_cases"
DEFAULT_RESULTS_DIR = BASE_DIR / "eval_results"
REQUIRED_CASE_FILES = ("resume.txt", "jd.txt", "expected.json")


def discover_eval_cases(cases_dir: str | Path = DEFAULT_CASES_DIR) -> list[Path]:
    """Return sorted case directories that contain all required input files."""
    root = Path(cases_dir)
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and all((path / name).is_file() for name in REQUIRED_CASE_FILES)
    )


def load_eval_case(case_dir: str | Path) -> dict[str, Any]:
    """Load one resume, JD and expected rule set."""
    path = Path(case_dir)
    return {
        "case_name": path.name,
        "resume_text": (path / "resume.txt").read_text(encoding="utf-8").strip(),
        "job_description": (path / "jd.txt").read_text(encoding="utf-8").strip(),
        "expected": json.loads((path / "expected.json").read_text(encoding="utf-8")),
    }


def mock_llm_for_eval(_prompt: str) -> str:
    """Deterministic output for validating workflow structure without Gemini."""
    return """## 岗位要求分析
岗位关注 Python、RAG、LLM 应用开发和 Agent 工程实践。

## 个人能力分析
### 能力匹配
候选人的项目体现了 Python、RAG 和大模型应用基础。

### 差距
生产环境部署、规模化评测和复杂 Agent planning 仍需补充。

## 匹配度分析
当前经历与岗位存在基础能力匹配，但应继续用指标验证检索质量。

## 简历优化建议
补充召回率、响应耗时、测试覆盖和实际问题修复案例。
"""


def _contains_hits(text: str, expected_values: list[str]) -> list[str]:
    normalized = (text or "").lower()
    return [value for value in expected_values if value.lower() in normalized]


def _trace_is_valid(workflow_result: dict[str, Any]) -> bool:
    trace = workflow_result.get("trace") or {}
    return bool(trace.get("run_id") and trace.get("steps"))


def evaluate_case(case: dict[str, Any], top_k: int = 5) -> dict[str, Any]:
    """Run deterministic RAG, Agent, trace and non-RAG workflow checks."""
    resume_text = case["resume_text"]
    job_description = case["job_description"]
    expected = case["expected"]
    errors: list[str] = []

    rag_result = rag_retrieve_tool(
        resume_text=resume_text,
        job_description=job_description,
        top_k=top_k,
        source_name=f"{case['case_name']}/resume.txt",
    )
    retrieved_chunks = rag_result.data.get("chunks", []) if rag_result.success else []
    retrieved_sections = sorted({chunk.get("section", "unknown") for chunk in retrieved_chunks})
    expected_sections_hit = sorted(
        set(expected.get("expected_sections", [])) & set(retrieved_sections)
    )
    retrieved_text = "\n".join(chunk.get("text", "") for chunk in retrieved_chunks)
    expected_keywords_hit = _contains_hits(
        retrieved_text,
        expected.get("expected_keywords", []),
    )
    rag_retrieve_passed = bool(
        rag_result.success
        and retrieved_chunks
        and expected_sections_hit
        and expected_keywords_hit
    )
    if not rag_result.success:
        errors.append(f"RAG retrieval error: {rag_result.error}")
    elif not expected_sections_hit:
        errors.append("RAG retrieval did not hit any expected section.")
    elif not expected_keywords_hit:
        errors.append("RAG retrieval did not hit any expected keyword.")

    agent_result = run_resume_agent_workflow(
        resume_text=resume_text,
        job_description=job_description,
        top_k=top_k,
        use_rag=True,
        source_name=f"{case['case_name']}/resume.txt",
        llm_callable=mock_llm_for_eval,
    )
    analysis = agent_result.get("analysis", "")
    agent_workflow_passed = bool(
        agent_result.get("success")
        and analysis
        and agent_result.get("workflow_steps")
    )
    trace_passed = _trace_is_valid(agent_result)
    if not agent_workflow_passed:
        errors.append(f"Agent workflow error: {agent_result.get('error') or 'missing output'}")
    if not trace_passed:
        errors.append("Agent workflow trace is incomplete.")

    required_headings = expected.get("required_output_headings", [])
    required_headings_hit = _contains_hits(analysis, required_headings)
    required_headings_passed = bool(required_headings) and len(required_headings_hit) == len(required_headings)
    if required_headings_passed:
        required_headings_status = "PASS"
    elif required_headings_hit:
        required_headings_status = "PARTIAL"
    else:
        required_headings_status = "FAIL"
        errors.append("Analysis did not contain any required heading or keyword.")

    non_rag_result = run_resume_agent_workflow(
        resume_text=resume_text,
        job_description=job_description,
        top_k=top_k,
        use_rag=False,
        llm_callable=mock_llm_for_eval,
    )
    non_rag_trace = non_rag_result.get("trace") or {}
    non_rag_steps = non_rag_trace.get("steps") or []
    non_rag_workflow_passed = bool(
        non_rag_result.get("success")
        and non_rag_result.get("analysis")
        and non_rag_trace.get("used_rag") is False
        and non_rag_steps
        and non_rag_steps[0].get("output_summary", {}).get("skipped") is True
    )
    if not non_rag_workflow_passed:
        errors.append("Non-RAG workflow did not complete or record the skipped RAG step.")

    overall_passed = bool(
        rag_retrieve_passed
        and agent_workflow_passed
        and trace_passed
        and non_rag_workflow_passed
        and required_headings_hit
    )
    return {
        "case_name": case["case_name"],
        "llm_mode": "mock",
        "rag_retrieve_passed": rag_retrieve_passed,
        "agent_workflow_passed": agent_workflow_passed,
        "non_rag_workflow_passed": non_rag_workflow_passed,
        "trace_passed": trace_passed,
        "required_headings_passed": required_headings_passed,
        "required_headings_status": required_headings_status,
        "required_headings_hit": required_headings_hit,
        "expected_sections_hit": expected_sections_hit,
        "expected_keywords_hit": expected_keywords_hit,
        "retrieved_chunk_count": len(retrieved_chunks),
        "errors": errors,
        "overall_passed": overall_passed,
    }


def save_eval_results(
    results: dict[str, Any],
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> str:
    """Save one aggregate eval run and return the absolute JSON path."""
    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"eval_result_{timestamp}.json"
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(output_path.resolve())


def run_evaluations(
    cases_dir: str | Path = DEFAULT_CASES_DIR,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    top_k: int = 5,
) -> tuple[dict[str, Any], str]:
    """Evaluate all discovered cases and persist an aggregate result."""
    case_paths = discover_eval_cases(cases_dir)
    case_results = [evaluate_case(load_eval_case(path), top_k=top_k) for path in case_paths]
    passed_cases = sum(1 for result in case_results if result["overall_passed"])
    total_cases = len(case_results)
    failed_cases = total_cases - passed_cases
    results = {
        "run_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "llm_mode": "mock",
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": failed_cases,
        "pass_rate": round((passed_cases / total_cases * 100), 1) if total_cases else 0.0,
        "cases": case_results,
    }
    return results, save_eval_results(results, results_dir)


def _status(value: bool) -> str:
    return "PASS" if value else "FAIL"


def main() -> int:
    print("Eval Runner started")
    case_paths = discover_eval_cases()
    print(f"Found {len(case_paths)} eval cases")

    results, output_path = run_evaluations()
    for case_result in results["cases"]:
        print(f"\n[{case_result['case_name']}]")
        print(f"* RAG retrieve: {_status(case_result['rag_retrieve_passed'])}")
        print(f"* Agent workflow: {_status(case_result['agent_workflow_passed'])}")
        print(f"* Non-RAG workflow: {_status(case_result['non_rag_workflow_passed'])}")
        print(f"* Trace generated: {_status(case_result['trace_passed'])}")
        print(f"* Required headings: {case_result['required_headings_status']}")
        print(f"* Overall: {_status(case_result['overall_passed'])}")
        for error in case_result["errors"]:
            print(f"  Error: {error}")

    print("\nSummary:")
    print(f"Total cases: {results['total_cases']}")
    print(f"Passed: {results['passed_cases']}")
    print(f"Failed: {results['failed_cases']}")
    print(f"Pass rate: {results['pass_rate']:.1f}%")
    print(f"LLM mode: {results['llm_mode']}")
    print(f"Result saved: {output_path}")
    return 0 if results["failed_cases"] == 0 and results["total_cases"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
