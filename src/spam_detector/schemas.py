"""Request and response models (validated by pydantic)."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

MAX_TEXT_LENGTH = 1000


class PredictRequest(BaseModel):
    text: str = Field(
        ..., min_length=1, max_length=MAX_TEXT_LENGTH, examples=["Free entry! Text WIN to 80082"]
    )

    @field_validator("text")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class PredictResponse(BaseModel):
    label: Literal["spam", "ham"]
    spam_probability: float = Field(..., ge=0.0, le=1.0)
    threshold: float
    model_version: str


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    model_loaded: bool
    model_version: str | None = None
