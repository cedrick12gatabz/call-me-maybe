"""Pydantic data models used by the project."""
from typing import Any, Literal

from pydantic import BaseModel, Field


class ParamDef(BaseModel):
    """Schema of one function parameter."""

    type: Literal["string", "number", "integer", "boolean", "null"]


class FunctionDef(BaseModel):
    """Definition of one callable function."""

    name: str = Field(pattern=r'^[^"\\\x00-\x1f]+$')
    description: str = ""
    parameters: dict[str, ParamDef] = Field(default_factory=dict)
    returns: dict[str, Any] = Field(default_factory=dict)


class PromptItem(BaseModel):
    """One natural-language request."""

    prompt: str


class FunctionCall(BaseModel):
    """One line of the output file."""

    prompt: str
    name: str
    parameters: dict[str, Any]