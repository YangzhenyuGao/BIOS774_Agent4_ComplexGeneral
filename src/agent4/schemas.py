"""Typed records passed between the stages of the agent (README 6.1, 6.2, 6.3)."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

Severity = Literal["low", "medium", "high"]
DIFFICULTY_TAGS = ("measurement_noise", "technical_confounding", "class_imbalance", "small_n_large_p", "missingness",
                   "hierarchy", "strong_baseline", "label_provenance", "multiple_comparisons", "statistical_power")
TASK_FAMILIES = ("cross_modal_prediction", "incremental_value_classification", "single_modality_classification",
                 "cross_modal_retrieval")


class Difficulty(BaseModel):
    id: str
    tag: str
    severity: Severity
    evidence: str                       # one sentence with the measured numbers
    numbers: dict = Field(default_factory=dict)
    implication: str                    # what an analysis must do because of it


class DataProfile(BaseModel):
    task_id: str
    adapter: str
    unit: str
    summary: dict                       # sizes, sparsity, missingness, pairing
    dataset_card: dict                  # what the data provider documents (provenance, known caveats)
    difficulties: list[Difficulty]


class AcceptanceCriterion(BaseModel):
    id: str
    statement: str
    check: str                          # key of a deterministic check the audit knows how to run
    rule: str = ""                      # how any threshold is derived from data (never a hard-coded number)


class Objective(BaseModel):
    question: str
    task_family: str
    analysis_unit: str
    estimand: str
    target: str
    allowed_inputs: list[str]
    forbidden_inputs: list[str]
    forbidden_reasons: dict = Field(default_factory=dict)
    required_comparators: list[str]
    primary_metric: str
    secondary_metrics: list[str] = Field(default_factory=list)
    acceptance_criteria: list[AcceptanceCriterion]
    conclusion_options: list[str]
    conclusion_scope: str
    notes: str = ""


class Budget(BaseModel):
    needs_gpu: bool = False
    pilot_minutes: float = 10
    full_minutes: float = 60
    cpus: int = 16


class Plan(BaseModel):
    plan_id: str
    name: str
    summary: str                        # plain-language description, no results
    complexity: Literal["simple_reference", "ambitious"]
    analysis_unit: str
    inputs: dict
    preprocessing: list[str] = Field(default_factory=list)
    model: dict                         # {"component": registry key, "params": {...}}
    fusion: Optional[str] = None        # how the baseline modality joins the added one, where the catalogue defines it
    training: dict = Field(default_factory=dict)
    validation: dict                    # the plan's own validation design
    metrics: list[str] = Field(default_factory=list)
    allowed_conclusions: str = ""
    budget: Budget = Field(default_factory=Budget)
    diversity: dict                     # {"data": ..., "model": ..., "method": ...}
    rationale: str = ""
    difficulties_addressed: list[str] = Field(default_factory=list)
    new_components: list[dict] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    id: str
    kind: Literal["self_score", "ruler", "convergence", "validation_design", "cost", "compliance", "note"]
    text: str
    value: Optional[Any] = None


class EvidenceCard(BaseModel):
    plan_id: str
    scale: str
    items: list[EvidenceItem]
    raw_files: list[str] = Field(default_factory=list)


class Verdict(BaseModel):
    winner: Literal["first", "second", "tie"]
    cited_evidence: list[str]
    reason: str
    validity_concerns: list[str] = Field(default_factory=list)
