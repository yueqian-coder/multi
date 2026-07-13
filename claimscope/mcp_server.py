from __future__ import annotations

import argparse
from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from .service import ClaimScopeService


mcp = FastMCP("ClaimScope")
VALID_MODES = {"auto", "heuristic", "llm", "llm_strict"}
MAX_DIRECTION_LENGTH = 8000
Direction = Annotated[
    str,
    Field(min_length=1, max_length=MAX_DIRECTION_LENGTH),
]
Mode = Literal["auto", "heuristic", "llm", "llm_strict"]
Limit = Annotated[int, Field(ge=1, le=50)]


def build_service() -> ClaimScopeService:
    return ClaimScopeService.from_env()


def extract_core_claim_payload(direction: str, mode: str = "auto") -> dict:
    return build_service().extract_core_claim(
        _validated_direction(direction), mode=_validated_mode(mode)
    )


def analyze_research_direction_payload(
    direction: str, online: bool = False, limit: int = 12
) -> dict:
    validated_limit = _validated_limit(limit)
    return build_service().analyze_research_direction(
        _validated_direction(direction),
        online=online,
        limit=validated_limit,
    )


def build_evidence_queries_payload(direction: str, mode: str = "auto") -> dict:
    return build_service().build_evidence_queries(
        _validated_direction(direction), mode=_validated_mode(mode)
    )


def evaluate_claim_benchmark_payload() -> dict:
    return build_service().evaluate_claim_benchmark()


def get_demo_report_payload() -> dict:
    return build_service().get_demo_report()


def _validated_direction(direction: str) -> str:
    if not isinstance(direction, str) or not direction.strip():
        raise ValueError("direction must be non-empty text")
    normalized = direction.strip()
    if len(normalized) > MAX_DIRECTION_LENGTH:
        raise ValueError(
            f"direction must not exceed {MAX_DIRECTION_LENGTH} characters"
        )
    return normalized


def _validated_mode(mode: str) -> str:
    normalized = mode.strip().lower() if isinstance(mode, str) else ""
    if normalized not in VALID_MODES:
        raise ValueError("mode must be one of: auto, heuristic, llm, llm_strict")
    return normalized


def _validated_limit(limit: int) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 50:
        raise ValueError("limit must be an integer between 1 and 50")
    return limit


@mcp.tool()
def extract_core_claim(direction: Direction, mode: Mode = "auto") -> dict:
    """Turn a fuzzy research direction into an auditable falsifiable claim."""
    return extract_core_claim_payload(direction, mode=mode)


@mcp.tool()
def analyze_research_direction(
    direction: Direction, online: bool = False, limit: Limit = 12
) -> dict:
    """Build the full ClaimScope assumption and evidence report."""
    return analyze_research_direction_payload(
        direction,
        online=online,
        limit=limit,
    )


@mcp.tool()
def build_evidence_queries(direction: Direction, mode: Mode = "auto") -> dict:
    """Generate adversarial evidence queries without retrieving papers."""
    return build_evidence_queries_payload(direction, mode=mode)


@mcp.tool()
def evaluate_claim_benchmark() -> dict:
    """Measure deterministic claim-structure quality, not scientific truth accuracy."""
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
