import importlib
import json
import subprocess
import sys
import textwrap
from importlib.metadata import entry_points
from pathlib import Path

import pytest

from claimscope.service import ClaimScopeService


REPO_ROOT = Path(__file__).resolve().parents[1]


class _InjectedPlanner:
    def extract_core_claim(self, direction: str) -> str:
        return f"Injected claim for {direction}"


def test_service_uses_injected_heuristic_planner():
    service = ClaimScopeService(heuristic_planner=_InjectedPlanner())

    payload = service.extract_core_claim("a research direction", mode="heuristic")

    assert payload["selected_claim"] == "Injected claim for a research direction"


def test_service_strict_agent_mode_requires_a_provider(monkeypatch):
    from claimscope.service import AgentProviderRequired

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    with pytest.raises(AgentProviderRequired):
        ClaimScopeService().extract_core_claim(
            "a research direction", mode="llm_strict"
        )


def test_mcp_module_import_has_no_server_side_effect():
    module = importlib.import_module("claimscope.mcp_server")

    assert module.mcp.name == "ClaimScope"


def test_extract_core_claim_tool_returns_json_compatible_trace():
    module = importlib.import_module("claimscope.mcp_server")

    payload = module.extract_core_claim_payload(
        "Can retrieval make medical QA safer?", mode="heuristic"
    )

    json.dumps(payload)
    assert payload["selected_claim"]
    assert payload["events"]
    assert payload["mode"] == "heuristic"


def test_analyze_research_direction_payload_is_keyless_offline_by_default():
    module = importlib.import_module("claimscope.mcp_server")

    payload = module.analyze_research_direction_payload(
        "RAG can reduce hallucination in LLM-generated answers",
        online=False,
        limit=6,
    )

    assert payload["claim"]
    assert payload["papers"]
    assert all(item["is_fixture"] for item in payload["papers"])
    assert json.loads(json.dumps(payload)) == payload


def test_build_evidence_queries_payload_exposes_assumption_queries_without_retrieval():
    module = importlib.import_module("claimscope.mcp_server")

    payload = module.build_evidence_queries_payload(
        "Diffusion models improve MRI tumor segmentation with limited labels"
    )

    assert payload["claim"]
    assert payload["assumptions"]
    assert payload["queries"]
    assert {item["kind"] for item in payload["queries"]} >= {
        "support",
        "contradict",
        "limitation",
        "null_result",
    }
    assert json.loads(json.dumps(payload)) == payload


@pytest.mark.parametrize(
    ("function_name", "args", "message"),
    [
        ("extract_core_claim_payload", ("   ",), "direction must be non-empty"),
        ("extract_core_claim_payload", ("direction", "invalid"), "mode must be"),
        (
            "analyze_research_direction_payload",
            ("direction", False, 0),
            "limit must be an integer between 1 and 50",
        ),
    ],
)
def test_mcp_payload_boundaries_reject_invalid_requests(function_name, args, message):
    module = importlib.import_module("claimscope.mcp_server")

    with pytest.raises(ValueError, match=message):
        getattr(module, function_name)(*args)


def test_benchmark_and_demo_payloads_are_json_compatible():
    module = importlib.import_module("claimscope.mcp_server")

    benchmark_payload = module.evaluate_claim_benchmark_payload()
    demo_payload = module.get_demo_report_payload()

    assert benchmark_payload["summary"]["case_count"] >= 24
    assert benchmark_payload["case_results"]
    assert demo_payload["markdown"].startswith("# ClaimScope Report")
    assert demo_payload["report"]["papers"]
    json.dumps(benchmark_payload)
    json.dumps(demo_payload)


def test_package_console_scripts_are_declared():
    scripts = {
        item.name: item.value
        for item in entry_points(group="console_scripts")
        if item.name in {"claimscope", "claimscope-mcp"}
    }

    assert scripts["claimscope"] == "claimscope.cli:main"
    assert scripts["claimscope-mcp"] == "claimscope.mcp_server:main"


def test_mcp_stdio_initialize_and_list_tools_smoke():
    script = textwrap.dedent(
        """
        import asyncio
        import json
        import sys
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        async def main():
            params = StdioServerParameters(
                command=sys.executable,
                args=["-m", "claimscope.mcp_server"],
                cwd=r"%s",
            )
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    print(json.dumps(sorted(tool.name for tool in tools.tools)))

        asyncio.run(main())
        """
        % str(REPO_ROOT)
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == [
        "analyze_research_direction",
        "build_evidence_queries",
        "evaluate_claim_benchmark",
        "extract_core_claim",
        "get_demo_report",
    ]


def test_mcp_stdio_calls_all_tools_and_reports_validation_errors():
    script = textwrap.dedent(
        """
        import asyncio
        import json
        import sys
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        async def main():
            params = StdioServerParameters(
                command=sys.executable,
                args=["-m", "claimscope.mcp_server"],
                cwd=r"%s",
            )
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    requests = {
                        "extract_core_claim": {
                            "direction": "RAG reduces unsupported answers versus closed-book QA",
                            "mode": "heuristic",
                        },
                        "analyze_research_direction": {
                            "direction": "RAG reduces unsupported answers",
                            "online": False,
                            "limit": 3,
                        },
                        "build_evidence_queries": {
                            "direction": "RAG reduces unsupported answers",
                            "mode": "heuristic",
                        },
                        "evaluate_claim_benchmark": {},
                        "get_demo_report": {},
                    }
                    outputs = {}
                    for name, arguments in requests.items():
                        result = await session.call_tool(name, arguments)
                        text = next(
                            item.text for item in result.content if hasattr(item, "text")
                        )
                        outputs[name] = {
                            "error": bool(result.isError),
                            "payload": json.loads(text),
                        }
                    failure = await session.call_tool(
                        "extract_core_claim",
                        {"direction": " ", "mode": "heuristic"},
                    )
                    listed = await session.list_tools()
                    schemas = {tool.name: tool.inputSchema for tool in listed.tools}
                    print(json.dumps({
                        "errors": {name: item["error"] for name, item in outputs.items()},
                        "extract_claim": outputs["extract_core_claim"]["payload"]["selected_claim"],
                        "extract_mode": outputs["extract_core_claim"]["payload"]["mode"],
                        "paper_count": len(outputs["analyze_research_direction"]["payload"]["papers"]),
                        "query_count": len(outputs["build_evidence_queries"]["payload"]["queries"]),
                        "metric_name": outputs["evaluate_claim_benchmark"]["payload"]["summary"]["metric_name"],
                        "demo_markdown": outputs["get_demo_report"]["payload"]["markdown"][:19],
                        "failure_error": bool(failure.isError),
                        "direction_min": schemas["extract_core_claim"]["properties"]["direction"].get("minLength"),
                        "direction_max": schemas["extract_core_claim"]["properties"]["direction"].get("maxLength"),
                        "mode_enum": schemas["extract_core_claim"]["properties"]["mode"].get("enum"),
                        "limit_min": schemas["analyze_research_direction"]["properties"]["limit"].get("minimum"),
                        "limit_max": schemas["analyze_research_direction"]["properties"]["limit"].get("maximum"),
                    }))

        asyncio.run(main())
        """
        % str(REPO_ROOT)
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=40,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert not any(payload["errors"].values())
    assert payload["extract_claim"]
    assert payload["extract_mode"] == "heuristic"
    assert payload["paper_count"] == 3
    assert payload["query_count"] >= 4
    assert payload["metric_name"] == "structural_quality_score"
    assert payload["demo_markdown"] == "# ClaimScope Report"
    assert payload["failure_error"]
    assert payload["direction_min"] == 1
    assert payload["direction_max"] == 8000
    assert payload["mode_enum"] == ["auto", "heuristic", "llm", "llm_strict"]
    assert payload["limit_min"] == 1
    assert payload["limit_max"] == 50
