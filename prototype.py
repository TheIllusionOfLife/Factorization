import random
import time
import math
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Strategy representation
# ------------------------------------------------------------------------------
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
            modulus_filters=[(mod, residues[:]) for mod, residues in self.modulus_filters],
            smoothness_bound=self.smoothness_bound,
            min_small_prime_hits=self.min_small_prime_hits,
        )

    def describe(self) -> str:
        filters = ", ".join(
            f"%{mod} in {tuple(residues)}" for mod, residues in self.modulus_filters
        ) or "<none>"
        return (
            f"power={self.power}, filters=[{filters}],"
            f" bound<={self.smoothness_bound}, hits>={self.min_small_prime_hits}"
        )

    def __call__(self, x: int, n: int) -> bool:
        candidate = abs(pow(x, self.power) - n)
        if candidate == 0:
            return True

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
            child.min_small_prime_hits = max(1, child.min_small_prime_hits + random.choice([-1, 0, 1]))

        if random.random() < 0.15 and len(child.modulus_filters) < 4:
            modulus = random.choice(self.primes)
            residues = random.sample(range(modulus), random.randint(1, min(3, modulus)))
            child.modulus_filters.append((modulus, residues))
        elif random.random() < 0.05 and len(child.modulus_filters) > 1:
            child.modulus_filters.pop(random.randrange(len(child.modulus_filters)))

        child._normalize()
        return child

# ------------------------------------------------------------------------------
# るつぼ (Crucible) - AI文明が挑戦する環境
# ------------------------------------------------------------------------------
class FactorizationCrucible:
    """
    素因数分解の「ふるい分け」ステップを模倣した環境。
    文明が提案した戦略の有効性をテストする。
    """
    def __init__(self, number_to_factor: int):
        self.N = number_to_factor
        self.search_space_root = int(math.sqrt(self.N))

    def evaluate_strategy(self, strategy: Callable[[int, int], bool], duration_seconds: float) -> int:
        """
        指定された戦略を一定時間実行し、どれだけ「スムーズな数」に近い候補を見つけられたかを評価する。
        ここでは「スコア」として、候補を発見した回数を返す。
        """
        score = 0
        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            # 探索空間からランダムに候補を選択
            x = self.search_space_root + random.randint(1, 1000)
            
            # AI文明が生成した戦略（ヒューリスティック）を実行
            try:
                if strategy(x, self.N):
                    score += 1
            except Exception:
                # 不正なコードが生成された場合はスコア0
                pass
        return score

# ------------------------------------------------------------------------------
# 進化的エンジン - 文明を進化させる淘汰圧
# ------------------------------------------------------------------------------
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
            from src.llm.base import LLMResponse

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
                logger.warning(f"Cannot add filter: maximum limit (4) reached, keeping parent strategy")
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
                logger.warning(f"Invalid filter index {idx} (valid range: 0-{len(new_filters)-1}), keeping parent strategy")
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
                    logger.warning(f"Cannot remove filter: minimum 1 filter required, keeping parent strategy")
                else:
                    logger.warning(f"Invalid filter index {idx} (valid range: 0-{len(new_filters)-1}), keeping parent strategy")
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


class EvolutionaryEngine:
    """
    文明の世代交代を司る。優れた戦略を選択し、次世代の戦略を生み出す。
    """
    def __init__(
        self,
        crucible: FactorizationCrucible,
        population_size: int = 10,
        llm_provider=None,
        evaluation_duration: float = 0.1,
    ):
        self.crucible = crucible
        self.population_size = population_size
        self.evaluation_duration = evaluation_duration
        self.civilizations: Dict[str, Dict] = {}
        self.generation = 0

        # LLM統合版または従来版のジェネレータを選択
        if llm_provider:
            self.generator = LLMStrategyGenerator(llm_provider=llm_provider)
        else:
            self.generator = LLMStrategyGenerator()  # llm_provider=None で従来と同じ動作

    def initialize_population(self):
        """最初の文明（戦略）群を生成する"""
        for i in range(self.population_size):
            civ_id = f"civ_{self.generation}_{i}"
            strategy = self.generator.random_strategy()
            self.civilizations[civ_id] = {"strategy": strategy, "fitness": 0}

    def run_evolutionary_cycle(self):
        """1世代分の進化（評価、選択、繁殖）を実行する"""
        print(f"\n===== Generation {self.generation}: Evaluating Strategies =====")

        # 評価: 全ての文明の戦略を評価する
        for civ_id, civ_data in self.civilizations.items():
            strategy = civ_data["strategy"]
            fitness = self.crucible.evaluate_strategy(
                strategy, duration_seconds=self.evaluation_duration
            )
            civ_data["fitness"] = fitness

            print(
                f"  Civilization {civ_id}: Fitness = {fitness:<5} | Strategy: {strategy.describe()}"
            )

        # 選択: フィットネスが高い上位20%の文明を選択
        sorted_civs = sorted(
            self.civilizations.items(), key=lambda item: item[1]["fitness"], reverse=True
        )
        num_elites = max(1, int(self.population_size * 0.2))
        elites = sorted_civs[:num_elites]

        print(
            f"\n--- Top performing civilization in Generation {self.generation}: "
            f"{elites[0][0]} with fitness {elites[0][1]['fitness']} ---"
        )

        # フィットネス履歴を更新（LLM用）
        if isinstance(self.generator, LLMStrategyGenerator):
            self.generator.fitness_history.append(elites[0][1]["fitness"])

        # 繁殖: エリート戦略を基に、次世代の文明（戦略）を生成
        next_generation_civs = {}
        for i in range(self.population_size):
            parent_civ = random.choice(elites)
            parent_strategy = parent_civ[1]["strategy"]
            parent_fitness = parent_civ[1]["fitness"]

            new_civ_id = f"civ_{self.generation + 1}_{i}"
            if random.random() < 0.2:
                new_strategy = self.generator.random_strategy()
            else:
                # LLM統合版の場合は文脈を渡す
                if isinstance(self.generator, LLMStrategyGenerator):
                    new_strategy = self.generator.mutate_strategy_with_context(
                        parent_strategy, parent_fitness, self.generation
                    )
                else:
                    new_strategy = self.generator.mutate_strategy(parent_strategy)

            next_generation_civs[new_civ_id] = {"strategy": new_strategy, "fitness": 0}

        self.civilizations = next_generation_civs
        self.generation += 1

# ------------------------------------------------------------------------------
# メイン実行ブロック
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evolutionary GNFS strategy optimizer with optional LLM integration"
    )
    parser.add_argument(
        "--number",
        type=int,
        default=961730063,
        help="Number to factor (default: 961730063)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=5,
        help="Number of generations to evolve (default: 5)",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=10,
        help="Population size per generation (default: 10)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.1,
        help="Evaluation duration in seconds (default: 0.1)",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Enable LLM-guided mutations (requires GEMINI_API_KEY in .env)",
    )

    args = parser.parse_args()

    # Initialize LLM provider if requested
    llm_provider = None
    if args.llm:
        try:
            from src.config import load_config
            from src.llm.gemini import GeminiProvider

            config = load_config()
            if not config.api_key:
                print("❌ ERROR: GEMINI_API_KEY not set in .env file")
                print("Please create .env file with your API key (see .env.example)")
                exit(1)

            llm_provider = GeminiProvider(config.api_key, config)
            print(f"✅ LLM mode enabled (Gemini 2.5 Flash Lite)")
            print(f"   Max API calls: {config.max_llm_calls}")
        except ImportError as e:
            print(f"❌ ERROR: Missing dependencies for LLM mode: {e}")
            print("Please run: pip install -r requirements.txt")
            exit(1)
    else:
        print("📊 Rule-based mode (no LLM)")

    # Create and run evolutionary engine
    crucible = FactorizationCrucible(args.number)
    engine = EvolutionaryEngine(
        crucible,
        population_size=args.population,
        llm_provider=llm_provider,
        evaluation_duration=args.duration,
    )

    print(f"\n🎯 Target number: {args.number}")
    print(f"🧬 Generations: {args.generations}, Population: {args.population}")
    print(f"⏱️  Evaluation duration: {args.duration}s per strategy\n")

    engine.initialize_population()

    for _ in range(args.generations):
        engine.run_evolutionary_cycle()

    # Display LLM cost summary if used
    if llm_provider:
        print("\n💰 LLM Cost Summary:")
        print(f"   Total API calls: {llm_provider.call_count}")
        print(f"   Total tokens: {llm_provider.input_tokens} in, {llm_provider.output_tokens} out")
        print(f"   Estimated cost: ${llm_provider.total_cost:.6f}")
