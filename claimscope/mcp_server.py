from __future__ import annotations

import argparse

from mcp.server.fastmcp import FastMCP

from .service import ClaimScopeService


mcp = FastMCP("ClaimScope")


def build_service() -> ClaimScopeService:
    return ClaimScopeService.from_env()


def extract_core_claim_payload(direction: str, mode: str = "auto") -> dict:
    return build_service().extract_core_claim(direction, mode=mode)


def analyze_research_direction_payload(
    direction: str, online: bool = False, limit: int = 12
) -> dict:
    return build_service().analyze_research_direction(
        direction,
        online=online,
        limit=limit,
    )


def build_evidence_queries_payload(direction: str, mode: str = "auto") -> dict:
    return build_service().build_evidence_queries(direction, mode=mode)


def evaluate_claim_benchmark_payload() -> dict:
    return build_service().evaluate_claim_benchmark()


def get_demo_report_payload() -> dict:
    return build_service().get_demo_report()


@mcp.tool()
def extract_core_claim(direction: str, mode: str = "auto") -> dict:
    """Turn a fuzzy research direction into an auditable falsifiable claim."""
    return extract_core_claim_payload(direction, mode=mode)


@mcp.tool()
def analyze_research_direction(
    direction: str, online: bool = False, limit: int = 12
) -> dict:
    """Build the full ClaimScope assumption and evidence report."""
    return analyze_research_direction_payload(
        direction,
        online=online,
        limit=limit,
    )


@mcp.tool()
def build_evidence_queries(direction: str, mode: str = "auto") -> dict:
    """Generate adversarial evidence queries without retrieving papers."""
    return build_evidence_queries_payload(direction, mode=mode)


@mcp.tool()
def evaluate_claim_benchmark() -> dict:
    """Run the deterministic ClaimBench benchmark for the core-claim engine."""
    return evaluate_claim_benchmark_payload()


@mcp.tool()
def get_demo_report() -> dict:
    """Return a deterministic keyless demo report using bundled fixture papers."""
    return get_demo_report_payload()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ClaimScope MCP server.")
    parser.parse_args()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
