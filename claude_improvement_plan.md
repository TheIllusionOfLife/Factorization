# Implementation Plan: Full LLM Integration (Option B)

## Overview
Transform prototype.py from mock to production-ready with real LLM integration, implementing the Project Prometheus vision of AI civilizations discovering novel factorization strategies.

## Phase 1: Project Setup & Dependencies
1. Create `requirements.txt` with dependencies:
   - `anthropic` (Claude API - recommended for code generation)
   - `openai` (optional fallback)
   - `python-dotenv` (environment variables)
   - `pytest` (testing)
2. Create `.env.example` template for API keys
3. Add `.env` and `__pycache__/` to `.gitignore`
4. Create project structure:
   ```
   src/
   ├── llm/
   │   ├── __init__.py
   │   ├── base.py (abstract LLM interface)
   │   └── claude.py (Claude implementation)
   ├── crucible/
   │   ├── __init__.py
   │   └── factorization.py (move Crucible here)
   ├── evolution/
   │   ├── __init__.py
   │   └── engine.py (move EvolutionaryEngine)
   └── config.py (constants & configuration)
   tests/
   └── test_integration.py
   ```

## Phase 2: Safe Code Execution Framework
1. Create `src/sandbox/executor.py`:
   - Use `ast.parse()` to validate Python syntax
   - Whitelist allowed operations (arithmetic, comparison, boolean logic)
   - Implement timeout mechanism (5 seconds max per strategy)
   - Restrict to pure functions (no I/O, no imports)
2. Replace `eval()` with safe executor
3. Add comprehensive error handling

## Phase 3: LLM Integration
1. Design prompt template for strategy generation:
   - Context: GNFS sieving optimization goal
   - Input: parent strategy code, generation number, fitness history
   - Output: Python lambda expression for candidate filtering
   - Constraints: must be stateless, fast evaluation, no external dependencies
2. Implement `src/llm/claude.py`:
   - API client initialization with retry logic
   - Temperature scaling (0.7 → 0.9 as generations progress)
   - Token limits (500 max for strategy generation)
   - Cost tracking per API call
3. Replace `LLM_propose_strategy()` with real API calls
4. Add fallback to simpler mutations if API fails

## Phase 4: Real Factorization Logic
1. Implement actual GNFS sieving evaluation:
   - Smooth number detection (B-smooth for configurable bound B)
   - Quadratic sieve as starting point (simpler than full GNFS)
   - Count successful "smooth" candidates found
2. Update `FactorizationCrucible`:
   - Real mathematical evaluation instead of random scoring
   - Proper search space exploration around √N
   - Track quality metrics (smoothness degree, prime factorization)

## Phase 5: Enhanced Evolution Engine
1. Add diversity preservation:
   - Track strategy uniqueness (AST similarity)
   - Penalize duplicates to prevent premature convergence
2. Implement fitness history:
   - Store performance across generations
   - Detect stagnation (trigger higher mutation rates)
3. Add checkpointing:
   - Save best strategies to JSON after each generation
   - Resume from checkpoint if interrupted

## Phase 6: Testing & Validation
1. Unit tests for safe executor
2. Integration test with real Claude API (small number factorization)
3. Verify strategy evolution produces working code
4. Cost estimation for full run (N generations)

## Phase 7: Documentation & CLI
1. Update README with:
   - Setup instructions (API key configuration)
   - Usage examples
   - Expected costs and runtime
2. Add CLI arguments:
   - `--number` (number to factor)
   - `--generations` (evolution cycles)
   - `--population` (civilization size)
   - `--api-key` (or use .env)
3. Add progress visualization (rich/tqdm)

## Deliverables
- Production-ready codebase with real LLM integration
- Comprehensive test suite
- Documentation for setup and usage
- Cost and performance benchmarks

## Estimated Effort
- ~3-4 hours for full implementation
- ~30-50 API calls for testing (≈$0.50-$2.00)

## Risks & Mitigations
- **API costs**: Start with small population (5) and few generations (3)
- **LLM hallucination**: Strict syntax validation catches invalid code
- **No convergence**: Implement diversity metrics and adaptive mutation rates
