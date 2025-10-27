"""Tests for Pydantic schemas with validators"""
import pytest
from pydantic import ValidationError


def test_power_mutation_valid():
    """Test PowerMutation with valid values"""
    from src.llm.schemas import PowerMutation

    mutation = PowerMutation(new_power=3)
    assert mutation.new_power == 3


def test_power_mutation_invalid_range():
    """Test PowerMutation rejects out-of-range values"""
    from src.llm.schemas import PowerMutation

    with pytest.raises(ValidationError):
        PowerMutation(new_power=1)  # Too low

    with pytest.raises(ValidationError):
        PowerMutation(new_power=6)  # Too high


def test_add_filter_mutation_valid():
    """Test AddFilterMutation with valid residues"""
    from src.llm.schemas import AddFilterMutation

    mutation = AddFilterMutation(modulus=5, residues=[0, 1, 2])
    assert mutation.modulus == 5
    assert mutation.residues == [0, 1, 2]


def test_add_filter_mutation_invalid_residues():
    """Test AddFilterMutation rejects residues >= modulus"""
    from src.llm.schemas import AddFilterMutation

    with pytest.raises(ValidationError, match="must be in range"):
        AddFilterMutation(modulus=5, residues=[0, 1, 5])  # 5 >= modulus

    with pytest.raises(ValidationError, match="must be in range"):
        AddFilterMutation(modulus=3, residues=[0, 1, 3])  # 3 >= modulus

    with pytest.raises(ValidationError, match="must be in range"):
        AddFilterMutation(modulus=7, residues=[-1, 0, 1])  # -1 < 0


def test_modify_filter_mutation_valid():
    """Test ModifyFilterMutation with valid values"""
    from src.llm.schemas import ModifyFilterMutation

    mutation = ModifyFilterMutation(index=0, modulus=7, residues=[0, 2, 4])
    assert mutation.index == 0
    assert mutation.modulus == 7
    assert mutation.residues == [0, 2, 4]


def test_modify_filter_mutation_invalid_residues():
    """Test ModifyFilterMutation rejects invalid residues"""
    from src.llm.schemas import ModifyFilterMutation

    with pytest.raises(ValidationError, match="must be in range"):
        ModifyFilterMutation(index=0, modulus=5, residues=[0, 5])


def test_remove_filter_mutation_valid():
    """Test RemoveFilterMutation"""
    from src.llm.schemas import RemoveFilterMutation

    mutation = RemoveFilterMutation(index=1)
    assert mutation.index == 1


def test_adjust_smoothness_mutation_valid():
    """Test AdjustSmoothnessMutation with valid deltas"""
    from src.llm.schemas import AdjustSmoothnessMutation

    mutation = AdjustSmoothnessMutation(bound_delta=2, hits_delta=1)
    assert mutation.bound_delta == 2
    assert mutation.hits_delta == 1


def test_adjust_smoothness_mutation_defaults():
    """Test AdjustSmoothnessMutation defaults to 0"""
    from src.llm.schemas import AdjustSmoothnessMutation

    mutation = AdjustSmoothnessMutation()
    assert mutation.bound_delta == 0
    assert mutation.hits_delta == 0


def test_adjust_smoothness_mutation_invalid_range():
    """Test AdjustSmoothnessMutation rejects out-of-range values"""
    from src.llm.schemas import AdjustSmoothnessMutation

    with pytest.raises(ValidationError):
        AdjustSmoothnessMutation(bound_delta=3)  # Max is 2

    with pytest.raises(ValidationError):
        AdjustSmoothnessMutation(hits_delta=2)  # Max is 1


def test_mutation_response_power():
    """Test MutationResponse with power mutation"""
    from src.llm.schemas import MutationResponse, PowerMutation

    response = MutationResponse(
        reasoning="Testing power change",
        mutation_type="power",
        power_params=PowerMutation(new_power=4)
    )
    assert response.mutation_type == "power"
    assert response.power_params.new_power == 4
    assert response.add_filter_params is None


def test_mutation_response_add_filter():
    """Test MutationResponse with add_filter mutation"""
    from src.llm.schemas import MutationResponse, AddFilterMutation

    response = MutationResponse(
        reasoning="Adding filter",
        mutation_type="add_filter",
        add_filter_params=AddFilterMutation(modulus=7, residues=[0, 1])
    )
    assert response.mutation_type == "add_filter"
    assert response.add_filter_params.modulus == 7


def test_mutation_response_invalid_type():
    """Test MutationResponse rejects invalid mutation type"""
    from src.llm.schemas import MutationResponse

    with pytest.raises(ValidationError):
        MutationResponse(
            reasoning="Test",
            mutation_type="invalid_type"
        )
