import random
import time
import math
from typing import List, Dict, Callable

# ------------------------------------------------------------------------------
# Mock LLM - 課題解決のための「戦略」を提案する創造主
# ------------------------------------------------------------------------------
def LLM_propose_strategy(base_strategy: str, generation: int) -> str:
    """
    LLMが新しい探索戦略（ヒューリスティック）をPythonコードとして生成・進化させるプロセスを模倣する。
    世代が進むにつれて、より洗練された戦略を提案する可能性がある。
    """
    # 新しい戦略を生成（ここでは既存の戦略に摂動を加えることで模倣）
    if generation < 2: # 初期世代は単純な戦略
        return f"lambda x, n: (x**2 - n) % {random.choice([3, 5, 7, 11, 13])} == 0"
    else: # 進化した世代は、より複雑な戦略を試みる
        new_strategy_code = base_strategy.replace(
            f"{random.choice([3, 5, 7, 11, 13, 17, 19])}",
            f"{random.choice([3, 5, 7, 11, 13, 17, 19, 23, 29])}"
        )
        if random.random() < 0.3:
             new_strategy_code += " or (x**3 - n) % 2 == 0"
        return new_strategy_code

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
class EvolutionaryEngine:
    """
    文明の世代交代を司る。優れた戦略を選択し、次世代の戦略を生み出す。
    """
    def __init__(self, crucible: FactorizationCrucible, population_size: int = 10):
        self.crucible = crucible
        self.population_size = population_size
        self.civilizations: Dict[str, Dict] = {}
        self.generation = 0

    def initialize_population(self):
        """最初の文明（戦略）群を生成する"""
        for i in range(self.population_size):
            civ_id = f"civ_{self.generation}_{i}"
            strategy_code = LLM_propose_strategy("lambda x, n: (x**2 - n) % 2 == 0", self.generation)
            self.civilizations[civ_id] = {"strategy_code": strategy_code, "fitness": 0}

    def run_evolutionary_cycle(self):
        """1世代分の進化（評価、選択、繁殖）を実行する"""
        print(f"\n===== Generation {self.generation}: Evaluating Strategies =====")
        
        # 評価: 全ての文明の戦略を評価する
        for civ_id, civ_data in self.civilizations.items():
            strategy_code = civ_data["strategy_code"]
            try:
                strategy_func = eval(strategy_code)
                fitness = self.crucible.evaluate_strategy(strategy_func, duration_seconds=0.1)
                self.civilizations[civ_id]["fitness"] = fitness
            except Exception as e:
                print(f"  Error evaluating {civ_id}: {e}")
                self.civilizations[civ_id]["fitness"] = 0

            print(f"  Civilization {civ_id}: Fitness = {self.civilizations[civ_id]['fitness']:<5} | Strategy: {strategy_code}")

        # 選択: フィットネスが高い上位20%の文明を選択
        sorted_civs = sorted(self.civilizations.items(), key=lambda item: item[1]['fitness'], reverse=True)
        num_elites = max(1, int(self.population_size * 0.2))
        elites = sorted_civs[:num_elites]
        
        print(f"\n--- Top performing civilization in Generation {self.generation}: {elites[0][0]} with fitness {elites[0][1]['fitness']} ---")

        # 繁殖: エリート戦略を基に、次世代の文明（戦略）を生成
        next_generation_civs = {}
        for i in range(self.population_size):
            parent_civ = random.choice(elites)
            parent_strategy_code = parent_civ[1]['strategy_code']
            
            new_civ_id = f"civ_{self.generation + 1}_{i}"
            new_strategy_code = LLM_propose_strategy(parent_strategy_code, self.generation + 1)
            next_generation_civs[new_civ_id] = {"strategy_code": new_strategy_code, "fitness": 0}

        self.civilizations = next_generation_civs
        self.generation += 1

# ------------------------------------------------------------------------------
# メイン実行ブロック
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    NUMBER_TO_FACTOR = 961730063
    NUM_GENERATIONS = 5

    crucible = FactorizationCrucible(NUMBER_TO_FACTOR)
    engine = EvolutionaryEngine(crucible)
    
    engine.initialize_population()
    
    for gen in range(NUM_GENERATIONS):
        engine.run_evolutionary_cycle()
