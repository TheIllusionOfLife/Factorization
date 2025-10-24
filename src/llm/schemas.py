"""Pydantic schemas for LLM structured output with field validation"""
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional


def _validate_residues_in_modulus_range(residues: list[int], modulus: int) -> list[int]:
    """Validate that all residues are in range [0, modulus)"""
    if any(r >= modulus or r < 0 for r in residues):
        raise ValueError(f"All residues must be in range [0, {modulus})")
    return residues


class PowerMutation(BaseModel):
    new_power: int = Field(..., ge=2, le=5, description="New power value (2-5)")


class AddFilterMutation(BaseModel):
    modulus: int = Field(..., ge=2, le=37, description="Prime modulus")
    residues: list[int] = Field(..., min_length=1, max_length=10, description="Allowed residues")

    @field_validator('residues')
    @classmethod
    def validate_residues(cls, v, info):
        modulus = info.data.get('modulus')
        if modulus:
            return _validate_residues_in_modulus_range(v, modulus)
        return v


class ModifyFilterMutation(BaseModel):
    index: int = Field(..., ge=0, description="Filter index to modify")
    modulus: int = Field(..., ge=2, le=37)
    residues: list[int] = Field(..., min_length=1, max_length=10)

    @field_validator('residues')
    @classmethod
    def validate_residues(cls, v, info):
        modulus = info.data.get('modulus')
        if modulus:
            return _validate_residues_in_modulus_range(v, modulus)
        return v


class RemoveFilterMutation(BaseModel):
    index: int = Field(..., ge=0, description="Filter index to remove")


class AdjustSmoothnessMutation(BaseModel):
    bound_delta: int = Field(0, ge=-2, le=2, description="Change to smoothness bound")
    hits_delta: int = Field(0, ge=-1, le=1, description="Change to min hits")


class MutationResponse(BaseModel):
    reasoning: str = Field(..., description="Brief explanation of mutation strategy")
    mutation_type: Literal["power", "add_filter", "modify_filter", "remove_filter", "adjust_smoothness"]

    # Only one of these will be populated based on mutation_type
    power_params: Optional[PowerMutation] = None
    add_filter_params: Optional[AddFilterMutation] = None
    modify_filter_params: Optional[ModifyFilterMutation] = None
    remove_filter_params: Optional[RemoveFilterMutation] = None
    adjust_smoothness_params: Optional[AdjustSmoothnessMutation] = None
