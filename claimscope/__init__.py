"""ClaimScope: pre-ideation claim, assumption, and negative-evidence mapping."""

from .models import (
    AnalysisReport,
    Assumption,
    ClaimVariant,
    EvidenceItem,
    IdeaOpportunity,
    NegativeEvidence,
    Paper,
)
from .pipeline import ClaimScopePipeline
from .retrievers import StaticPaperRetriever

__all__ = [
    "AnalysisReport",
    "Assumption",
    "ClaimScopePipeline",
    "ClaimVariant",
    "EvidenceItem",
    "IdeaOpportunity",
    "NegativeEvidence",
    "Paper",
    "StaticPaperRetriever",
]
