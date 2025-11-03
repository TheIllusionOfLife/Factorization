"""Gemini API provider with structured output"""

import json
import logging
from typing import List

from google import genai
from google.genai import types

from src.strategy import Strategy

from .base import LLMProvider, LLMResponse
from .schemas import MutationResponse

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    """Gemini 2.5 Flash Lite provider with structured JSON output"""

    def __init__(self, api_key: str, config):
        self.client = genai.Client(api_key=api_key)
        self.config = config
        self._total_cost = 0.0
        self._call_count = 0
        self._input_tokens = 0
        self._output_tokens = 0

    def propose_mutation(
        self,
        parent_strategy: dict,
        fitness: int,
        generation: int,
        fitness_history: list,
    ) -> LLMResponse:
        """Propose a mutation using Gemini API with structured output"""
        if self._call_count >= self.config.max_llm_calls:
            return LLMResponse(
                success=False, mutation_params={}, error="API call limit reached"
            )

        prompt = self._build_prompt(
            parent_strategy, fitness, generation, fitness_history
        )
        temperature = self._calculate_temperature(generation)

        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=self.config.max_tokens,
                    response_mime_type="application/json",
                    response_schema=MutationResponse,
                ),
            )

            # Parse structured JSON response
            mutation_data = json.loads(response.text)

            # Track token usage with defensive access
            try:
                usage = getattr(response, "usage_metadata", None)
                if usage:
                    input_tokens = getattr(usage, "prompt_token_count", 0)
                    output_tokens = getattr(usage, "candidates_token_count", 0)
                    self._input_tokens += input_tokens
                    self._output_tokens += output_tokens
                    cost = self._calculate_cost(input_tokens, output_tokens)
                    self._total_cost += cost
                else:
                    input_tokens = output_tokens = 0
                    cost = 0.0
            except (AttributeError, TypeError) as e:
                # Expected if API response structure changes
                logger.debug(f"Token tracking failed: {e}")
                input_tokens = output_tokens = 0
                cost = 0.0

            # Convert to standard mutation format
            mutation_params = self._convert_response(mutation_data)

            return LLMResponse(
                success=True,
                mutation_params=mutation_params,
                reasoning=mutation_data.get("reasoning", ""),
                cost=cost,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            return LLMResponse(
                success=False,
                mutation_params={},
                error=f"Invalid JSON response: {str(e)}",
            )
        except (ValueError, KeyError) as e:
            logger.error(f"Invalid LLM response structure: {e}")
            return LLMResponse(
                success=False,
                mutation_params={},
                error=f"Invalid response structure: {str(e)}",
            )
        except Exception as e:
            # Catch unexpected errors with full logging
            logger.exception(f"Unexpected error in LLM API call: {e}")
            return LLMResponse(
                success=False, mutation_params={}, error=f"API error: {str(e)}"
            )
        finally:
            # Increment call count regardless of success/failure to prevent infinite retries
            self._call_count += 1

    def _build_prompt(
        self,
        parent_strategy: dict,
        fitness: int,
        generation: int,
        fitness_history: list,
    ) -> str:
        """Build prompt for mutation proposal"""
        recent_history = fitness_history[-5:] if len(fitness_history) > 0 else []

        return f"""You are optimizing heuristics for the General Number Field Sieve (GNFS) factorization algorithm.

## Task
Propose a mutation to improve a strategy that identifies "smooth numbers" (numbers with many small prime factors).

## Parent Strategy Performance
- Current fitness: {fitness} candidates found in 0.1 seconds
- Generation: {generation}
- Recent fitness trend: {recent_history}

## Current Strategy Parameters
- **Power**: {parent_strategy["power"]} (range: 2-5)
  - Determines the polynomial: x^power - N
  - Lower powers (2-3) are faster, higher powers (4-5) may find better candidates

- **Modulus filters**: {parent_strategy["modulus_filters"]}
  - Format: [(modulus, [allowed_residues]), ...]
  - Example: [(3, [0, 1])] means "candidate % 3 must be 0 or 1"
  - These quickly reject non-smooth candidates

- **Smoothness bound**: {parent_strategy["smoothness_bound"]}
  - Maximum prime to check for divisibility
  - Available primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37

- **Min small prime hits**: {parent_strategy["min_small_prime_hits"]}
  - Minimum count of small prime factors required
  - Higher = more selective, lower = more permissive

## Mutation Strategy Guidelines

**If fitness is LOW (<20)**: Try dramatic changes
- Change power completely
- Add/remove filters to change selectivity
- Adjust smoothness threshold significantly

**If fitness is MODERATE (20-50)**: Experiment with filters
- Add new modulus filters to be more selective
- Adjust existing residue sets
- Fine-tune smoothness parameters

**If fitness is HIGH (>50)**: Make careful refinements
- Small adjustments to existing filters
- Minor smoothness tuning
- Preserve what's working

**Generation context**:
- Early (0-3): Explore diverse approaches, try different powers
- Mid (4-7): Exploit promising patterns, refine filters
- Late (8+): Fine-tune, make minimal changes

## Available Mutations
1. **power**: Change polynomial power
2. **add_filter**: Add new modulus filter (max 4 total)
3. **modify_filter**: Change existing filter's modulus or residues
4. **remove_filter**: Remove a filter (if >1 exists)
5. **adjust_smoothness**: Tweak smoothness_bound and/or min_hits

## Your Task
Analyze the current strategy and propose ONE mutation that is most likely to improve fitness.
Consider: What mathematical properties make numbers smooth? How can modular arithmetic help identify them faster?"""

    def _calculate_temperature(self, generation: int) -> float:
        """Scale temperature: early gens explore (high temp), later exploit (low temp)"""
        progress = min(generation / self.config.temperature_scaling_generations, 1.0)
        # Start high (max) for exploration, decrease to low (base) for exploitation
        temp: float = (
            self.config.temperature_max
            - (self.config.temperature_max - self.config.temperature_base) * progress
        )
        return temp

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Gemini 2.5 Flash Lite pricing:
        - Free tier: $0.00 (unlimited within rate limits)
        - Paid tier: $0.10/M input, $0.40/M output

        We estimate paid tier costs for transparency
        """
        input_cost = (input_tokens / 1_000_000) * 0.10
        output_cost = (output_tokens / 1_000_000) * 0.40
        return input_cost + output_cost

    def _convert_response(self, mutation_data: dict) -> dict:
        """Convert structured response to mutation parameters"""
        mutation_type = mutation_data["mutation_type"]

        result = {"mutation_type": mutation_type, "parameters": {}}

        # Map mutation type to corresponding params key
        params_key = f"{mutation_type}_params"
        if params_key in mutation_data and mutation_data[params_key] is not None:
            result["parameters"] = mutation_data[params_key]

        return result

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def input_tokens(self) -> int:
        return self._input_tokens

    @property
    def output_tokens(self) -> int:
        return self._output_tokens

    def propose_mutation_with_feedback(
        self,
        strategy: Strategy,
        fitness: float,
        feedback_text: str,
        fitness_history: List[float],
        generation: int,
        max_generations: int,
    ) -> LLMResponse:
        """
        Propose mutation using EvaluationSpecialist feedback.

        Enhanced version of propose_mutation() that incorporates feedback context
        from the EvaluationSpecialist agent for more informed mutation decisions.

        Args:
            strategy: Parent Strategy object to mutate
            fitness: Current fitness score (candidate count)
            feedback_text: Feedback from EvaluationSpecialist
            fitness_history: Recent fitness scores (last 5)
            generation: Current generation number
            max_generations: Total generations for temperature scaling

        Returns:
            LLMResponse with mutation parameters and reasoning
        """
        if self._call_count >= self.config.max_llm_calls:
            return LLMResponse(
                success=False, mutation_params={}, error="API call limit reached"
            )

        # Convert Strategy object to dict for prompt
        parent_strategy = {
            "power": strategy.power,
            "modulus_filters": strategy.modulus_filters,
            "smoothness_bound": strategy.smoothness_bound,
            "min_small_prime_hits": strategy.min_small_prime_hits,
        }

        prompt = self._build_feedback_prompt(
            parent_strategy,
            fitness,
            feedback_text,
            fitness_history,
            generation,
            max_generations,
        )
        temperature = self._calculate_temperature_with_progress(
            generation, max_generations
        )

        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=self.config.max_tokens,
                    response_mime_type="application/json",
                    response_schema=MutationResponse,
                ),
            )

            # Parse structured JSON response
            mutation_data = json.loads(response.text)

            # Track token usage with defensive access
            try:
                usage = getattr(response, "usage_metadata", None)
                if usage:
                    input_tokens = getattr(usage, "prompt_token_count", 0)
                    output_tokens = getattr(usage, "candidates_token_count", 0)
                    self._input_tokens += input_tokens
                    self._output_tokens += output_tokens
                    cost = self._calculate_cost(input_tokens, output_tokens)
                    self._total_cost += cost
                else:
                    input_tokens = output_tokens = 0
                    cost = 0.0
            except (AttributeError, TypeError) as e:
                # Expected if API response structure changes
                logger.debug(f"Token tracking failed: {e}")
                input_tokens = output_tokens = 0
                cost = 0.0

            # Convert to standard mutation format
            mutation_params = self._convert_response(mutation_data)

            return LLMResponse(
                success=True,
                mutation_params=mutation_params,
                reasoning=mutation_data.get("reasoning", ""),
                cost=cost,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            return LLMResponse(
                success=False,
                mutation_params={},
                error=f"Invalid JSON response: {str(e)}",
            )
        except (ValueError, KeyError) as e:
            logger.error(f"Invalid LLM response structure: {e}")
            return LLMResponse(
                success=False,
                mutation_params={},
                error=f"Invalid response structure: {str(e)}",
            )
        except Exception as e:
            # Catch unexpected errors with full logging
            logger.exception(f"Unexpected error in LLM API call: {e}")
            return LLMResponse(
                success=False, mutation_params={}, error=f"API error: {str(e)}"
            )
        finally:
            # Increment call count regardless of success/failure to prevent infinite retries
            self._call_count += 1

    def _build_feedback_prompt(
        self,
        parent_strategy: dict,
        fitness: float,
        feedback_text: str,
        fitness_history: List[float],
        generation: int,
        max_generations: int,
    ) -> str:
        """Build enhanced prompt with EvaluationSpecialist feedback"""
        recent_history = fitness_history[-5:] if len(fitness_history) > 0 else []

        return f"""You are a SearchSpecialist in an AI civilization working on GNFS factorization.

**Your Role**: Generate novel strategies to find "smooth numbers" (numbers with many small prime factors).

**Current Strategy**:
- Power: {parent_strategy["power"]} (polynomial degree: x^power - N)
- Modulus filters: {parent_strategy["modulus_filters"]} (quick rejection via x mod m)
- Smoothness bound: {parent_strategy["smoothness_bound"]} (max prime factor to check)
- Min small prime hits: {parent_strategy["min_small_prime_hits"]} (required small factor count)

**Current Performance**:
- Fitness: {fitness} candidates found in evaluation window
- Recent trend: {recent_history}
- Generation: {generation}/{max_generations}

**EvaluationSpecialist's Feedback**:
{feedback_text}

**Domain Knowledge**:
- Higher power → more candidates but slower evaluation
- More filters → fewer candidates but potentially higher quality
- Higher smoothness_bound → more thorough prime checking (slower)
- Higher min_hits → stricter acceptance (fewer candidates)

**Your Task**:
Analyze the feedback and propose a mutation that addresses the issues raised.

**Reasoning Process**:
1. What is the feedback telling you? (performance, quality, efficiency issues?)
2. What is the root cause? (too restrictive? too permissive? inefficient?)
3. What parameter change would address this?
4. What are the tradeoffs of this change?

**Output Format**:
Propose ONE mutation type:
- power: Change polynomial degree
- add_filter: Add modulus filter
- modify_filter: Change existing filter
- remove_filter: Remove filter
- adjust_smoothness: Change smoothness_bound or min_hits

Provide clear reasoning for your choice."""

    def _calculate_temperature_with_progress(
        self, generation: int, max_generations: int
    ) -> float:
        """
        Scale temperature based on generation progress.

        Early generations explore (high temp), later exploit (low temp).
        Uses max_generations for accurate progress calculation.
        """
        progress = min(generation / max_generations, 1.0)
        # Start high (max) for exploration, decrease to low (base) for exploitation
        temp: float = (
            self.config.temperature_max
            - (self.config.temperature_max - self.config.temperature_base) * progress
        )
        return temp
