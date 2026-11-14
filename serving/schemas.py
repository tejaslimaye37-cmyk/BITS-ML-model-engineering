from typing import Annotated, List, Optional

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Cat = Optional[Annotated[str, StringConstraints(strip_whitespace=True, max_length=64)]]


class CustomerRecord(BaseModel):
    """One subscriber. tenure, MonthlyCharges and Contract are required; other fields may be omitted."""
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "customerID": "7590-VHVEG", "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service", "InternetService": "DSL",
        "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85}})

    customerID: Optional[str] = Field(None, max_length=64)
    gender: Cat = None
    SeniorCitizen: Optional[int] = Field(None, ge=0, le=1)
    Partner: Cat = None
    Dependents: Cat = None
    tenure: int = Field(..., ge=0)
    PhoneService: Cat = None
    MultipleLines: Cat = None
    InternetService: Cat = None
    OnlineSecurity: Cat = None
    OnlineBackup: Cat = None
    DeviceProtection: Cat = None
    TechSupport: Cat = None
    StreamingTV: Cat = None
    StreamingMovies: Cat = None
    Contract: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]
    PaperlessBilling: Cat = None
    PaymentMethod: Cat = None
    MonthlyCharges: float = Field(..., ge=0, allow_inf_nan=False)
    TotalCharges: Optional[float] = Field(None, ge=0, allow_inf_nan=False)


class PredictionResponse(BaseModel):
    customer_id: Optional[str] = None
    churn_probability: float
    churn_flag: bool
    threshold: float
    scoring_source: str
    model_version: str
    warnings: List[str] = []
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    fallback_available: bool
