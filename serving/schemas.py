"""Request/response contracts for the online API."""
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src import config as C

_BAL = dict(default=None, ge=0, le=1e11, allow_inf_nan=False)


class TransactionRequest(BaseModel):
    """One transaction at authorisation time. The model uses only pre-settlement
    fields (type, amount, step, oldbalanceOrg, oldbalanceDest). The two "new"
    balances are accepted for compatibility with PaySim-style records but are
    ignored. Missing balances are imputed with the training median and the
    response carries a warning."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    transaction_id: str = Field(min_length=1, max_length=64)
    step: int = Field(ge=1, le=100_000, description="hour index since start of the log")
    type: str
    amount: float = Field(ge=0, le=1e10, allow_inf_nan=False)
    oldbalanceOrg: Optional[float] = Field(**_BAL)
    newbalanceOrig: Optional[float] = Field(**_BAL)
    oldbalanceDest: Optional[float] = Field(**_BAL)
    newbalanceDest: Optional[float] = Field(**_BAL)

    @field_validator("type")
    @classmethod
    def _known_type(cls, v: str) -> str:
        v = v.upper()
        if v not in C.KNOWN_TYPES:
            raise ValueError(f"type must be one of {list(C.KNOWN_TYPES)}")
        return v

    def missing_fields(self) -> List[str]:
        return [f for f in C.REQUIRED_FOR_MODEL if getattr(self, f) is None]


class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    decision_threshold: float
    mode: Literal["model", "fallback_rule", "out_of_scope_rule"]
    model_version: str
    needs_review: bool = False   # true when the score cannot be trusted (e.g. sender balance missing)
    warnings: List[str] = []


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: List[TransactionRequest] = Field(min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    count: int
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str
    decision_threshold: Optional[float]
    uptime_seconds: float
