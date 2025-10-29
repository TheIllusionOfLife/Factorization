"""Strategy representation and generators for GNFS optimization."""

import logging
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

logger = logging.getLogger(__name__)

# Constants
SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]


@dataclass
class Strategy:
    power: int
    modulus_filters: List[Tuple[int, List[int]]]
    smoothness_bound: int
    min_small_prime_hits: int

    def __post_init__(self) -> None:
        self._normalize()

    def copy(self) -> "Strategy":
        return Strategy(
            power=self.power,
            modulus_filters=[
                (mod, residues[:]) for mod, residues in self.modulus_filters
            ],
            smoothness_bound=self.smoothness_bound,
            min_small_prime_hits=self.min_small_prime_hits,
        )

    def describe(self) -> str:
        filters = (
            ", ".join(
                f"%{mod} in {tuple(residues)}" for mod, residues in self.modulus_filters
            )
            or "<none>"
        )
        return (
            f"power={self.power}, filters=[{filters}],"
            f" bound<={self.smoothness_bound}, hits>={self.min_small_prime_hits}"
        )

    def __call__(self, x: int, n: int) -> bool:
        candidate = abs(pow(x, self.power) - n)
        # Skip zero candidates (consistent with evaluate_strategy_detailed)
        if candidate == 0:
            return False

        for modulus, residues in self.modulus_filters:
            if candidate % modulus not in residues:
                return False

        return self._count_small_prime_hits(candidate) >= self.min_small_prime_hits

    def _normalize(self) -> None:
        self.power = max(2, min(5, self.power))
        normalized_filters: List[Tuple[int, List[int]]] = []
        for modulus, residues in self.modulus_filters:
            modulus = max(2, modulus)
            residues = sorted({residue % modulus for residue in residues})
            if residues:
                normalized_filters.append((modulus, residues))
        self.modulus_filters = normalized_filters[:4]
        self.smoothness_bound = max(3, min(self.smoothness_bound, SMALL_PRIMES[-1]))
        self.min_small_prime_hits = max(1, min(self.min_small_prime_hits, 6))

    def _count_small_prime_hits(self, candidate: int) -> int:
        hits = 0
        remainder = candidate
        for prime in SMALL_PRIMES:
            if prime > self.smoothness_bound:
                break
            while remainder % prime == 0:
                remainder //= prime
                hits += 1
                if hits >= self.min_small_prime_hits:
                    return hits
        return hits


def blend_modulus_filters(
    filters1: List[Tuple[int, List[int]]],
    filters2: List[Tuple[int, List[int]]],
    max_filters: int = 4,
) -> List[Tuple[int, List[int]]]:
    """
    Blend modulus filters from two parents, prioritizing diversity.

    Combines filters from both parents:
    - Merges filters with same modulus (union of residues)
    - Keeps unique filters from each parent
    - Limits total to max_filters (prioritizes smaller moduli)

    Args:
        filters1: Modulus filters from parent 1
        filters2: Modulus filters from parent 2
        max_filters: Maximum number of filters to keep (default: 4)

    Returns:
        Blended list of (modulus, residues) tuples
    """
    # Merge filters by modulus using set operations for efficiency
    filter_dict: dict[int, List[int]] = {}

    for modulus, residues in filters1 + filters2:
        if modulus in filter_dict:
            # Merge residues for same modulus using set union
            filter_dict[modulus] = sorted(set(filter_dict[modulus]) | set(residues))
        else:
            filter_dict[modulus] = sorted(set(residues))

    # Convert back to list, sorted by modulus (prioritize smaller moduli)
    blended = [(mod, res) for mod, res in sorted(filter_dict.items())]

    # Limit to max_filters
    return blended[:max_filters]


def crossover_strategies(parent1: Strategy, parent2: Strategy) -> Strategy:
    """
    Uniform crossover: combine strategies from two parents.

    Each component has 50% chance to come from either parent:
    - power: Random choice from {parent1.power, parent2.power}
    - modulus_filters: Blend filters from both parents
    - smoothness_bound: Random choice from {parent1, parent2}.smoothness_bound
    - min_small_prime_hits: Random choice from {parent1, parent2}.min_small_prime_hits

    The resulting strategy is automatically normalized by Strategy.__post_init__.

    Args:
        parent1: First parent strategy
        parent2: Second parent strategy

    Returns:
        New strategy combining traits from both parents
    """
    # Randomly select discrete parameters from either parent
    power = random.choice([parent1.power, parent2.power])
    smoothness_bound = random.choice(
        [parent1.smoothness_bound, parent2.smoothness_bound]
    )
    min_small_prime_hits = random.choice(
        [parent1.min_small_prime_hits, parent2.min_small_prime_hits]
    )

    # Blend modulus filters from both parents
    modulus_filters = blend_modulus_filters(
        parent1.modulus_filters, parent2.modulus_filters, max_filters=4
    )

    # Create new strategy (automatically normalized)
    return Strategy(
        power=power,
        modulus_filters=modulus_filters,
        smoothness_bound=smoothness_bound,
        min_small_prime_hits=min_small_prime_hits,
    )


class StrategyGenerator:
    def __init__(self, primes: Sequence[int] = SMALL_PRIMES) -> None:
        self.primes = list(primes)

    def random_strategy(self) -> Strategy:
        power = random.choice([2, 2, 2, 3, 3, 4])
        filter_count = random.randint(1, 3)
        filters: List[Tuple[int, List[int]]] = []
        for _ in range(filter_count):
            modulus = random.choice(self.primes)
            residue_count = random.randint(1, min(3, modulus))
            residues = random.sample(range(modulus), residue_count)
            filters.append((modulus, residues))

        smoothness_bound = random.choice(self.primes[3:])
        min_hits = random.randint(1, 4)

        return Strategy(
            power=power,
            modulus_filters=filters,
            smoothness_bound=smoothness_bound,
            min_small_prime_hits=min_hits,
        )

    def mutate_strategy(self, parent: Strategy) -> Strategy:
        child = parent.copy()
        mutation_roll = random.random()

        if mutation_roll < 0.3:
            child.power = random.choice([2, 2, 3, 3, 4, 5])
        elif mutation_roll < 0.6 and child.modulus_filters:
            index = random.randrange(len(child.modulus_filters))
            modulus, residues = child.modulus_filters[index]
            if random.random() < 0.5:
                modulus = random.choice(self.primes)
                residues = [r % modulus for r in residues]
            else:
                choices = list(range(modulus))
                if random.random() < 0.5 and len(residues) > 1:
                    residues.pop(random.randrange(len(residues)))
                else:
                    candidate = random.choice(choices)
                    if candidate not in residues:
                        residues.append(candidate)
            child.modulus_filters[index] = (modulus, residues)
        else:
            adjustment = random.choice([-2, -1, 1, 2])
            child.smoothness_bound = child.smoothness_bound + adjustment
            child.min_small_prime_hits = max(
                1, child.min_small_prime_hits + random.choice([-1, 0, 1])
            )

        if random.random() < 0.15 and len(child.modulus_filters) < 4:
            modulus = random.choice(self.primes)
            residues = random.sample(range(modulus), random.randint(1, min(3, modulus)))
            child.modulus_filters.append((modulus, residues))
        elif random.random() < 0.05 and len(child.modulus_filters) > 1:
            child.modulus_filters.pop(random.randrange(len(child.modulus_filters)))

        child._normalize()
        return child


class LLMStrategyGenerator(StrategyGenerator):
    """
    LLM統合版の戦略生成器。
    LLMによる戦略提案を試み、失敗時は従来のルールベース手法にフォールバックする。
    """

    def __init__(self, llm_provider=None, primes=SMALL_PRIMES):
        if not primes:
            raise ValueError("primes list cannot be empty")
        super().__init__(primes)
        self.llm_provider = llm_provider
        self.fitness_history = []

    def mutate_strategy_with_context(
        self, parent: Strategy, fitness: int, generation: int
    ) -> Strategy:
        """
        文脈（fitness、世代数）を考慮した戦略の変異。
        LLMが利用可能な場合はLLMによる提案を試み、失敗時はルールベースにフォールバック。
        """
        # LLMが利用可能な場合は提案を試みる
        if self.llm_provider:
            response = self.llm_provider.propose_mutation(
                parent_strategy={
                    "power": parent.power,
                    "modulus_filters": parent.modulus_filters,
                    "smoothness_bound": parent.smoothness_bound,
                    "min_small_prime_hits": parent.min_small_prime_hits,
                },
                fitness=fitness,
                generation=generation,
                fitness_history=self.fitness_history[-5:],  # 直近5世代の履歴
            )

            if response.success:
                child = self._apply_llm_mutation(parent, response.mutation_params)
                print(f"    [LLM] {response.reasoning}")
                return child

        # LLMが失敗またはNoneの場合は従来のルールベース手法
        return super().mutate_strategy(parent)

    def _apply_llm_mutation(self, parent: Strategy, mutation_params: dict) -> Strategy:
        """LLMの提案した変異パラメータを実際の戦略に適用"""
        mutation_type = mutation_params["mutation_type"]
        params = mutation_params.get("parameters", {})

        if mutation_type == "power":
            return Strategy(
                power=params["new_power"],
                modulus_filters=parent.modulus_filters[:],
                smoothness_bound=parent.smoothness_bound,
                min_small_prime_hits=parent.min_small_prime_hits,
            )

        elif mutation_type == "add_filter":
            new_filters = parent.modulus_filters[:]
            if len(new_filters) < 4:  # 最大4フィルタまで
                new_filters.append((params["modulus"], params["residues"]))
            else:
                logger.warning(
                    "Cannot add filter: maximum limit (4) reached, keeping parent strategy"
                )
            return Strategy(
                power=parent.power,
                modulus_filters=new_filters,
                smoothness_bound=parent.smoothness_bound,
                min_small_prime_hits=parent.min_small_prime_hits,
            )

        elif mutation_type == "modify_filter":
            new_filters = parent.modulus_filters[:]
            idx = params["index"]
            if 0 <= idx < len(new_filters):
                new_filters[idx] = (params["modulus"], params["residues"])
            else:
                logger.warning(
                    f"Invalid filter index {idx} (valid range: 0-{len(new_filters) - 1}), keeping parent strategy"
                )
            return Strategy(
                power=parent.power,
                modulus_filters=new_filters,
                smoothness_bound=parent.smoothness_bound,
                min_small_prime_hits=parent.min_small_prime_hits,
            )

        elif mutation_type == "remove_filter":
            new_filters = parent.modulus_filters[:]
            idx = params["index"]
            if 0 <= idx < len(new_filters) and len(new_filters) > 1:
                del new_filters[idx]
            else:
                if len(new_filters) <= 1:
                    logger.warning(
                        "Cannot remove filter: minimum 1 filter required, keeping parent strategy"
                    )
                else:
                    logger.warning(
                        f"Invalid filter index {idx} (valid range: 0-{len(new_filters) - 1}), keeping parent strategy"
                    )
            return Strategy(
                power=parent.power,
                modulus_filters=new_filters,
                smoothness_bound=parent.smoothness_bound,
                min_small_prime_hits=parent.min_small_prime_hits,
            )

        elif mutation_type == "adjust_smoothness":
            # smoothness_boundの変更（利用可能な素数内に制限）
            bound_delta = params.get("bound_delta", 0)
            new_bound = parent.smoothness_bound + bound_delta
            # SMALL_PRIMESに含まれる素数に制限
            new_bound = max(min(self.primes), min(new_bound, max(self.primes)))

            # min_small_prime_hitsの変更
            hits_delta = params.get("hits_delta", 0)
            new_hits = max(1, parent.min_small_prime_hits + hits_delta)

            return Strategy(
                power=parent.power,
                modulus_filters=parent.modulus_filters[:],
                smoothness_bound=new_bound,
                min_small_prime_hits=new_hits,
            )

        # 未知の変異タイプの場合は親をそのまま返す
        return parent


__all__ = [
    "SMALL_PRIMES",
    "Strategy",
    "blend_modulus_filters",
    "crossover_strategies",
    "StrategyGenerator",
    "LLMStrategyGenerator",
]
