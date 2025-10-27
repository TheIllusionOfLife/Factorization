"""Tests for LLM provider classes"""

import os

import pytest


def test_config_loading():
    """Test configuration loads correctly"""
    from src.config import Config

    config = Config(api_key="test-key-123", max_llm_calls=50)
    assert config.api_key == "test-key-123"
    assert config.max_llm_calls == 50
    assert config.temperature_base == 0.8
    assert config.temperature_max == 1.2


def test_llm_response_dataclass():
    """Test LLMResponse dataclass"""
    from src.llm.base import LLMResponse

    response = LLMResponse(
        success=True,
        mutation_params={"mutation_type": "power", "parameters": {"new_power": 3}},
        reasoning="Test reasoning",
        cost=0.001,
        input_tokens=100,
        output_tokens=50,
    )
    assert response.success is True
    assert response.mutation_params["mutation_type"] == "power"
    assert response.cost == 0.001
    assert response.input_tokens == 100


def test_llm_response_with_error():
    """Test LLMResponse with error"""
    from src.llm.base import LLMResponse

    response = LLMResponse(success=False, mutation_params={}, error="API call failed")
    assert response.success is False
    assert response.error == "API call failed"


def test_gemini_provider_init():
    """Test GeminiProvider initialization"""
    from src.config import Config
    from src.llm.gemini import GeminiProvider

    config = Config(api_key="test-key")
    provider = GeminiProvider("test-key", config)

    assert provider.call_count == 0
    assert provider.total_cost == 0.0


def test_temperature_scaling():
    """Test temperature decreases with generation (exploration → exploitation)"""
    from src.config import Config
    from src.llm.gemini import GeminiProvider

    config = Config(api_key="test", temperature_base=0.8, temperature_max=1.2)
    provider = GeminiProvider("test", config)

    temp_gen_0 = provider._calculate_temperature(0)
    temp_gen_5 = provider._calculate_temperature(5)
    temp_gen_10 = provider._calculate_temperature(10)

    assert temp_gen_0 == 1.2  # Start high for exploration
    assert 0.8 < temp_gen_5 < 1.2  # Mid-range
    assert temp_gen_10 == 0.8  # End low for exploitation


def test_cost_calculation():
    """Test cost calculation"""
    from src.config import Config
    from src.llm.gemini import GeminiProvider

    config = Config(api_key="test")
    provider = GeminiProvider("test", config)

    # Test cost calculation (input: $0.10/M, output: $0.40/M)
    cost = provider._calculate_cost(1000, 500)
    expected = (1000 / 1_000_000) * 0.10 + (500 / 1_000_000) * 0.40
    assert abs(cost - expected) < 0.000001


def test_convert_response_power():
    """Test response conversion for power mutation"""
    from src.config import Config
    from src.llm.gemini import GeminiProvider

    config = Config(api_key="test")
    provider = GeminiProvider("test", config)

    mutation_data = {
        "mutation_type": "power",
        "power_params": {"new_power": 3},
        "reasoning": "Test",
    }

    result = provider._convert_response(mutation_data)
    assert result["mutation_type"] == "power"
    assert result["parameters"]["new_power"] == 3


def test_convert_response_add_filter():
    """Test response conversion for add_filter mutation"""
    from src.config import Config
    from src.llm.gemini import GeminiProvider

    config = Config(api_key="test")
    provider = GeminiProvider("test", config)

    mutation_data = {
        "mutation_type": "add_filter",
        "add_filter_params": {"modulus": 5, "residues": [0, 1]},
        "reasoning": "Test",
    }

    result = provider._convert_response(mutation_data)
    assert result["mutation_type"] == "add_filter"
    assert result["parameters"]["modulus"] == 5
    assert result["parameters"]["residues"] == [0, 1]


def test_convert_response_missing_params():
    """Test response conversion handles missing params gracefully"""
    from src.config import Config
    from src.llm.gemini import GeminiProvider

    config = Config(api_key="test")
    provider = GeminiProvider("test", config)

    mutation_data = {
        "mutation_type": "power",
        "reasoning": "Test",
        # No power_params provided
    }

    result = provider._convert_response(mutation_data)
    assert result["mutation_type"] == "power"
    assert result["parameters"] == {}


def test_api_call_limit():
    """Test API call limit enforcement"""
    from src.config import Config
    from src.llm.gemini import GeminiProvider

    config = Config(api_key="test", max_llm_calls=0)
    provider = GeminiProvider("test", config)

    parent = {
        "power": 2,
        "modulus_filters": [],
        "smoothness_bound": 13,
        "min_small_prime_hits": 2,
    }
    response = provider.propose_mutation(parent, 50, 0, [])

    assert response.success is False
    assert "limit reached" in response.error


def test_api_call_count_increments_on_failure():
    """Test that API call count increments even when API call fails"""
    from unittest.mock import patch

    from src.config import Config
    from src.llm.gemini import GeminiProvider

    config = Config(api_key="test", max_llm_calls=10)
    provider = GeminiProvider("test", config)

    assert provider.call_count == 0

    parent = {
        "power": 2,
        "modulus_filters": [(3, [0, 1])],
        "smoothness_bound": 13,
        "min_small_prime_hits": 2,
    }

    # Mock the API call to raise an exception
    with patch.object(
        provider.client.models, "generate_content", side_effect=Exception("API Error")
    ):
        response = provider.propose_mutation(
            parent, fitness=50, generation=0, fitness_history=[]
        )

        # Should return error response
        assert response.success is False
        assert "API error" in response.error

        # Call count should still increment (critical for preventing infinite retries)
        assert provider.call_count == 1


@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"), reason="No GEMINI_API_KEY environment variable"
)
def test_real_gemini_call():
    """Integration test with real Gemini API (requires GEMINI_API_KEY)"""
    from src.config import load_config
    from src.llm.gemini import GeminiProvider

    config = load_config()
    provider = GeminiProvider(config.api_key, config)

    parent = {
        "power": 2,
        "modulus_filters": [(3, [0, 1]), (5, [0])],
        "smoothness_bound": 13,
        "min_small_prime_hits": 2,
    }

    response = provider.propose_mutation(parent, 45, 3, [30, 35, 40, 42, 45])

    assert response.success, f"API call failed: {response.error}"
    assert response.mutation_params is not None
    assert "mutation_type" in response.mutation_params
    assert response.reasoning is not None
    assert len(response.reasoning) > 10  # Non-trivial reasoning

    print("\n[Real API Test Results]")
    print(f"  Mutation type: {response.mutation_params['mutation_type']}")
    print(f"  Reasoning: {response.reasoning}")
    print(f"  Cost: ${response.cost:.6f}")
    print(f"  Tokens: {response.input_tokens} in, {response.output_tokens} out")
