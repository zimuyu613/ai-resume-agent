import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EVAL_EMBEDDING_PROVIDERS = ["local"]
# Keep the default eval deterministic and independent from model downloads/APIs.
os.environ["EMBEDDING_PROVIDER"] = EVAL_EMBEDDING_PROVIDERS[0]

from agent_workflow import run_resume_agent_workflow
from rag_eval_utils import evaluate_retrieval_result
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


def _compare_rerank(rag_metrics: dict[str, Any], rerank_metrics: dict[str, Any]) -> dict[str, Any]:
    rag_recall_3 = float(rag_metrics.get("recall_at_k", {}).get("3", 0.0))
    rerank_recall_3 = float(rerank_metrics.get("recall_at_k", {}).get("3", 0.0))
    rag_mrr = float(rag_metrics.get("mrr", 0.0))
    rerank_mrr = float(rerank_metrics.get("mrr", 0.0))
    recall_delta = round(rerank_recall_3 - rag_recall_3, 4)
    mrr_delta = round(rerank_mrr - rag_mrr, 4)

    if recall_delta > 0 or mrr_delta > 0:
        status = "improved"
    elif recall_delta == 0 and mrr_delta == 0:
        status = "same"
    else:
        status = "worse"

    return {
        "status": status,
        "rerank_improved": status == "improved",
        "recall_at_3_delta": recall_delta,
        "mrr_delta": mrr_delta,
        "improvement_summary": (
            f"Rerank is {status}: Recall@3 delta={recall_delta:+.4f}, "
            f"MRR delta={mrr_delta:+.4f}."
        ),
    }


def evaluate_case(case: dict[str, Any], top_k: int = 5) -> dict[str, Any]:
    """Evaluate retrieval quality plus the existing Agent engineering checks."""
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
    rag_metrics = evaluate_retrieval_result(retrieved_chunks, expected)
    rag_retrieve_passed = bool(
        rag_result.success
        and retrieved_chunks
        and rag_metrics["section_metrics"]["section_hit_rate"] > 0
        and rag_metrics["keyword_metrics"]["keyword_hit_rate"] > 0
        and rag_metrics["gold_metrics"]["gold_recall"] > 0
    )
    if not rag_retrieve_passed:
        errors.append(f"RAG retrieval quality check failed: {rag_result.error or 'no expected evidence hit'}")

    rerank_result = rag_retrieve_tool(
        resume_text=resume_text,
        job_description=job_description,
        top_k=top_k,
        source_name=f"{case['case_name']}/resume.txt",
        use_rerank=True,
    )
    reranked_chunks = rerank_result.data.get("chunks", []) if rerank_result.success else []
    rerank_metrics = evaluate_retrieval_result(reranked_chunks, expected)
    rerank_keyword_hits = sorted(
        {
            keyword
            for chunk in reranked_chunks
            for keyword in chunk.get("keyword_hits", [])
        }
    )
    rerank_passed = bool(
        rerank_result.success
        and reranked_chunks
        and rerank_result.data.get("used_rerank") is True
        and all("rerank_score" in chunk for chunk in reranked_chunks)
        and rerank_metrics["gold_metrics"]["gold_recall"] > 0
    )
    if not rerank_passed:
        errors.append(f"Rerank quality check failed: {rerank_result.error or 'missing score/evidence'}")

    top_rerank_score = max(
        (float(chunk["rerank_score"]) for chunk in reranked_chunks if "rerank_score" in chunk),
        default=None,
    )
    rerank_comparison = _compare_rerank(rag_metrics, rerank_metrics)

    agent_result = run_resume_agent_workflow(
        resume_text=resume_text,
        job_description=job_description,
        top_k=top_k,
        use_rag=True,
        use_rerank=True,
        source_name=f"{case['case_name']}/resume.txt",
        llm_callable=mock_llm_for_eval,
    )
    analysis = agent_result.get("analysis", "")
    agent_workflow_passed = bool(
        agent_result.get("success") and analysis and agent_result.get("workflow_steps")
    )
    trace_passed = bool(
        _trace_is_valid(agent_result)
        and agent_result.get("trace", {}).get("used_rerank") is True
    )
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
        errors.append("Non-RAG workflow did not record the skipped RAG step.")

    overall_passed = bool(
        rag_retrieve_passed
        and rerank_passed
        and agent_workflow_passed
        and trace_passed
        and non_rag_workflow_passed
        and required_headings_hit
    )
    return {
        "case_name": case["case_name"],
        "llm_mode": "mock",
        "embedding_provider": EVAL_EMBEDDING_PROVIDERS[0],
        "rag_retrieve_passed": rag_retrieve_passed,
        "rag_metrics": rag_metrics,
        "rerank_passed": rerank_passed,
        "rerank_metrics": rerank_metrics,
        "rerank_comparison": rerank_comparison,
        "rerank_improved": rerank_comparison["rerank_improved"],
        "improvement_summary": rerank_comparison["improvement_summary"],
        "rerank_keyword_hits": rerank_keyword_hits,
        "rerank_score": top_rerank_score,
        "used_rerank": rerank_result.data.get("used_rerank", False),
        "agent_workflow_passed": agent_workflow_passed,
        "non_rag_workflow_passed": non_rag_workflow_passed,
        "trace_passed": trace_passed,
        "required_headings_passed": required_headings_passed,
        "required_headings_status": required_headings_status,
        "required_headings_hit": required_headings_hit,
        # Keep the original compact fields for backward-readable eval JSON.
        "expected_sections_hit": rag_metrics["section_metrics"]["hit_sections"],
        "expected_keywords_hit": rag_metrics["keyword_metrics"]["hit_keywords"],
        "retrieved_chunk_count": rag_metrics["retrieved_count"],
        "errors": errors,
        "overall_passed": overall_passed,
    }


def _average(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _aggregate_retrieval(case_results: list[dict[str, Any]], metrics_key: str) -> dict[str, Any]:
    metrics = [result[metrics_key] for result in case_results]
    return {
        "average_recall_at_k": {
            k: _average([float(item["recall_at_k"].get(k, 0.0)) for item in metrics])
            for k in ("1", "3", "5")
        },
        "average_mrr": _average([float(item.get("mrr", 0.0)) for item in metrics]),
        "average_gold_recall": _average(
            [float(item["gold_metrics"].get("gold_recall", 0.0)) for item in metrics]
        ),
        "average_section_hit_rate": _average(
            [float(item["section_metrics"].get("section_hit_rate", 0.0)) for item in metrics]
        ),
        "average_keyword_hit_rate": _average(
            [float(item["keyword_metrics"].get("keyword_hit_rate", 0.0)) for item in metrics]
        ),
    }


def build_aggregate_metrics(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = [result["rerank_comparison"]["status"] for result in case_results]
    return {
        "rag": _aggregate_retrieval(case_results, "rag_metrics"),
        "rerank": _aggregate_retrieval(case_results, "rerank_metrics"),
        "rerank_comparison": {
            "improved_cases": statuses.count("improved"),
            "same_cases": statuses.count("same"),
            "worse_cases": statuses.count("worse"),
        },
    }


def save_eval_results(
    results: dict[str, Any],
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    timestamp: str | None = None,
) -> str:
    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"eval_result_{timestamp}.json"
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(output_path.resolve())


def save_eval_summary(
    results: dict[str, Any],
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    timestamp: str | None = None,
) -> str:
    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"eval_summary_{timestamp}.md"
    aggregate = results["aggregate_metrics"]
    rag = aggregate["rag"]
    rerank = aggregate["rerank"]

    lines = [
        "# RAG Evaluation Summary",
        "",
        f"- 评测时间：{results['run_time']}",
        f"- Embedding provider：{', '.join(results['embedding_providers'])}",
        f"- 总 case 数：{results['total_cases']}",
        f"- 通过：{results['passed_cases']}",
        f"- 失败：{results['failed_cases']}",
        f"- 通过率：{results['pass_rate']:.1f}%",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | RAG | RAG + Rerank |",
        "| --- | ---: | ---: |",
    ]
    for k in ("1", "3", "5"):
        lines.append(
            f"| Recall@{k} | {rag['average_recall_at_k'][k]:.4f} | "
            f"{rerank['average_recall_at_k'][k]:.4f} |"
        )
    lines.extend(
        [
            f"| MRR | {rag['average_mrr']:.4f} | {rerank['average_mrr']:.4f} |",
            f"| Gold Recall | {rag['average_gold_recall']:.4f} | {rerank['average_gold_recall']:.4f} |",
            "",
            "## RAG vs RAG + Rerank",
            "",
            f"- Improved cases：{aggregate['rerank_comparison']['improved_cases']}",
            f"- Same cases：{aggregate['rerank_comparison']['same_cases']}",
            f"- Worse cases：{aggregate['rerank_comparison']['worse_cases']}",
            "",
            "## Case Results",
            "",
            "| Case | RAG R@3 | RAG MRR | Rerank R@3 | Rerank MRR | Comparison | Overall |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for case in results["cases"]:
        lines.append(
            f"| {case['case_name']} | {case['rag_metrics']['recall_at_k']['3']:.4f} | "
            f"{case['rag_metrics']['mrr']:.4f} | "
            f"{case['rerank_metrics']['recall_at_k']['3']:.4f} | "
            f"{case['rerank_metrics']['mrr']:.4f} | "
            f"{case['rerank_comparison']['status']} | "
            f"{'PASS' if case['overall_passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## 评测边界",
            "",
            "当前评测使用少量脱敏样例，并以 section + keywords 作为简化 gold evidence。",
            "它主要衡量检索排序和工程链路，不是逐 chunk 人工标注的工业评测，",
            "也不代表 mock LLM 或真实 Gemini 的最终回答质量。",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return str(output_path.resolve())


def run_evaluations(
    cases_dir: str | Path = DEFAULT_CASES_DIR,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    top_k: int = 5,
) -> tuple[dict[str, Any], str]:
    case_paths = discover_eval_cases(cases_dir)
    case_results = [evaluate_case(load_eval_case(path), top_k=top_k) for path in case_paths]
    passed_cases = sum(1 for result in case_results if result["overall_passed"])
    total_cases = len(case_results)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results = {
        "run_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "llm_mode": "mock",
        "embedding_providers": EVAL_EMBEDDING_PROVIDERS,
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": total_cases - passed_cases,
        "pass_rate": round((passed_cases / total_cases * 100), 1) if total_cases else 0.0,
        "cases": case_results,
        "aggregate_metrics": build_aggregate_metrics(case_results),
    }
    summary_path = save_eval_summary(results, results_dir, timestamp)
    results["artifacts"] = {"markdown_summary": summary_path}
    json_path = save_eval_results(results, results_dir, timestamp)
    return results, json_path


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
        print(f"* RAG Recall@3: {case_result['rag_metrics']['recall_at_k']['3']:.2f}")
        print(f"* RAG MRR: {case_result['rag_metrics']['mrr']:.2f}")
        print(f"* RAG+Rerank Recall@3: {case_result['rerank_metrics']['recall_at_k']['3']:.2f}")
        print(f"* RAG+Rerank MRR: {case_result['rerank_metrics']['mrr']:.2f}")
        print(f"* Rerank comparison: {case_result['rerank_comparison']['status']}")
        print(f"* Agent workflow: {_status(case_result['agent_workflow_passed'])}")
        print(f"* Trace generated: {_status(case_result['trace_passed'])}")
        print(f"* Overall: {_status(case_result['overall_passed'])}")
        for error in case_result["errors"]:
            print(f"  Error: {error}")

    print("\nSummary:")
    print(f"Total cases: {results['total_cases']}")
    print(f"Passed: {results['passed_cases']}")
    print(f"Failed: {results['failed_cases']}")
    print(f"Pass rate: {results['pass_rate']:.1f}%")
    print(f"Embedding providers: {', '.join(results['embedding_providers'])}")
    if results.get("aggregate_metrics"):
        rag = results["aggregate_metrics"]["rag"]
        rerank = results["aggregate_metrics"]["rerank"]
        print(f"Average RAG Recall@3: {rag['average_recall_at_k']['3']:.2f}")
        print(f"Average RAG MRR: {rag['average_mrr']:.2f}")
        print(f"Average RAG+Rerank Recall@3: {rerank['average_recall_at_k']['3']:.2f}")
        print(f"Average RAG+Rerank MRR: {rerank['average_mrr']:.2f}")
    print(f"JSON result saved: {output_path}")
    print(f"Markdown summary saved: {results['artifacts']['markdown_summary']}")
    return 0 if results["failed_cases"] == 0 and results["total_cases"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
