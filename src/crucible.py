"""Factorization evaluation environment (Crucible)."""

import math
import random
import time
from typing import Callable

from src.metrics import (
    MAX_EXAMPLE_CANDIDATES_TO_STORE,
    MAX_SMOOTHNESS_SCORES_TO_STORE,
    EvaluationMetrics,
)
from src.strategy import SMALL_PRIMES, Strategy


class FactorizationCrucible:
    """
    素因数分解の「ふるい分け」ステップを模倣した環境。
    文明が提案した戦略の有効性をテストする。
    """

    def __init__(self, number_to_factor: int):
        self.N = number_to_factor
        self.search_space_root = int(math.sqrt(self.N))

    def evaluate_strategy(
        self, strategy: Callable[[int, int], bool], duration_seconds: float
    ) -> int:
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

    def evaluate_strategy_detailed(
        self, strategy: Strategy, duration_seconds: float
    ) -> EvaluationMetrics:
        """
        Evaluate strategy with detailed instrumentation.

        Returns comprehensive metrics including:
        - Candidate count
        - Smoothness scores
        - Timing breakdown by phase
        - Rejection statistics
        - Example smooth candidates found
        """
        candidates_found = []
        smoothness_scores = []
        rejections = {"modulus_filter": 0, "min_hits": 0, "passed": 0}

        timing = {
            "candidate_generation": 0.0,
            "modulus_filtering": 0.0,
            "smoothness_check": 0.0,
        }

        start_time = time.perf_counter()
        end_time = start_time + duration_seconds

        while time.perf_counter() < end_time:
            # Time candidate generation
            gen_start = time.perf_counter()
            x = self.search_space_root + random.randint(1, 1000)
            candidate = abs(pow(x, strategy.power) - self.N)
            timing["candidate_generation"] += time.perf_counter() - gen_start

            if candidate == 0:
                continue

            # Time modulus filtering
            filter_start = time.perf_counter()
            passes_filter = True
            for modulus, residues in strategy.modulus_filters:
                if candidate % modulus not in residues:
                    passes_filter = False
                    rejections["modulus_filter"] += 1
                    break
            timing["modulus_filtering"] += time.perf_counter() - filter_start

            if not passes_filter:
                continue

            # Time smoothness check (combined with prime factorization)
            smooth_start = time.perf_counter()
            hits = 0
            prime_product = 1
            temp = candidate

            for prime in SMALL_PRIMES:
                if prime > strategy.smoothness_bound:
                    break
                while temp % prime == 0:
                    temp //= prime
                    prime_product *= prime
                    hits += 1

            timing["smoothness_check"] += time.perf_counter() - smooth_start

            if hits >= strategy.min_small_prime_hits:
                rejections["passed"] += 1
                candidates_found.append(candidate)

                # Smoothness: ratio of candidate to smooth part
                # Lower = smoother (more factors removed)
                # prime_product > 1 guaranteed here (hits >= min_small_prime_hits)
                smoothness = candidate / prime_product
                smoothness_scores.append(smoothness)
            else:
                rejections["min_hits"] += 1

        return EvaluationMetrics(
            candidate_count=len(candidates_found),
            smoothness_scores=smoothness_scores[:MAX_SMOOTHNESS_SCORES_TO_STORE],
            timing_breakdown=timing,
            rejection_stats=rejections,
            example_candidates=candidates_found[:MAX_EXAMPLE_CANDIDATES_TO_STORE],
        )


__all__ = ["FactorizationCrucible"]
