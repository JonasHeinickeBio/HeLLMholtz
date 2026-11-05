# Benchmarking and the leaderboard

HeLLMholtz benchmarks every available Blablador chat model on a fixed prompt suite and
publishes the result as an auto-updating leaderboard.

- **Live dashboard:** <https://jonasheinickebio.github.io/HeLLMholtz/benchmark/>
- **Static copy:** the *Model Leaderboard* section of the README (rewritten on every run)
- **Raw data:** `docs/benchmark/data.json`, history in `reports/history.json`

## How a response is scored

Every run gets a score from 0 to 100 built from two independent signals.

| Signal | What it is | Used for |
|---|---|---|
| **Checks** | Deterministic: exact final answers, executed unit tests, JSON schema, word/sentence limits, required/forbidden content | Anything verifiable (math, code, instruction following, extraction) |
| **Judge panel** | Blind LLM judges grade correctness, completeness, instruction following and clarity (1-5 each) against a reference answer and a per-prompt rubric | Open-ended answers (creative, medical, knowledge, summaries) |

* Prompts that have checks and a rubric blend them: `60% checks + 40% judge`.
* Prompts fully covered by checks are **not** judged - deterministic results are authoritative.
* A failed, truncated or empty run scores **0**; a model that errors on half the prompts cannot
  outrank one that answers all of them. Reliability is shown separately.
* The overall score is the mean of the **category** means, so a category with many prompts does
  not dominate. The dashboard shows a 95% confidence interval.

### Judge design

The judge never sees which model wrote the answer, receives a reference and rubric when the
prompt defines them, must answer in JSON (analysis first, then scores), and runs at temperature 0.
Judges form a panel (default: `alias-huge` + `alias-deepseek-v41-flash-booster`); scores are
averaged, the disagreement is stored as `judge_spread`, and **a judge never grades its own
model** (self-preference bias).

### Adding a prompt

Append a `_p(...)` call to `src/hellmholtz/benchmark/prompts.py` with a stable `id`, a category,
`checks` and/or `reference` + `rubric`, then add a gold answer to
`tests/benchmark/test_checks_and_scoring.py` - the test suite verifies that every prompt
accepts a correct answer. See `src/hellmholtz/benchmark/checks.py` for the check types.

## Speed and reasoning behaviour

Quality and speed are reported separately; the speed probe never touches the score.

* **Speed probe.** After the quality prompts, every model gets two streamed requests
  with a fixed short task (`hellm bench --speed-probe`, on by default;
  `--no-speed-probe` skips it). It measures **time to first token (TTFT)** and **decode
  throughput** (tokens after the first, per second). End-to-end latency from the quality
  prompts is kept too, but it mixes in answer length and thinking time.
* **Speed score (0-100).** Mean of two log-scaled parts: throughput (10 tok/s = 0,
  200 tok/s = 100) and TTFT (10 s = 0, 0.3 s = 100). Without a probe the score falls back
  to throughput from the quality runs.
* **Reasoning prompts.** Twelve multi-step problems with unique, machine-checkable
  answers (rate/probability puzzles, ordering and river-crossing planning, ...). Beyond
  the final answer, the judge also rates the working through a process rubric, so a lucky
  guess does not score like a sound derivation.
* **Thinking behaviour.** `Think %` is the share of output tokens spent on reasoning,
  from the provider's `reasoning_tokens` or estimated from `reasoning_content` /
  `<think>` blocks. `Cut-off %` is the share of runs that hit the token limit
  (`finish_reason=length`), typically reasoning models running out of budget.

## Tiers: quick vs. full

| Tier | Prompts | When |
|---|---|---|
| `quick` (`hellm bench --quick`) | 20, at least one per category | every Sunday |
| full (`--all-prompts`) | 58 | 1st of the month, newly available models, manual runs |

Results merge per (model, prompt), so a quick run refreshes the quick prompts and the rest of
the leaderboard keeps the values from the last full run. Trigger either manually with the
workflow's `mode` input.

## Running it yourself

```bash
# Everything available, 8 workers, judged by the default panel (~1 h)
hellm bench --models "$(python -P scripts/select_models.py)" --all-prompts \
  --temperatures 0.2 --replications 1 --max-tokens 4096 --concurrency 8 --evaluate-with default

# Only models that appeared since the last committed models_status.yaml
python -P scripts/select_models.py --new-since HEAD

# Re-score and (re-)judge an existing results file; resumable, never self-judges
python -P scripts/run_llm_judge.py results/benchmark_X.json --workers 4

# Rebuild dashboard, README leaderboard and history from a judged file
python scripts/build_dashboard.py results/benchmark_X_evaluated.json
```

`-P` keeps the repository root off `sys.path`: it contains a test stub of `aisuite` that would
otherwise shadow the real client. Reasoning models need a generous `--max-tokens` (4096) or
they spend the budget thinking and return an empty answer. `HELLM_REQUEST_TIMEOUT` (default
120 s) bounds each request.

## How GitHub keeps it current

| Workflow | When | What |
|---|---|---|
| `daily-model-check.yml` | daily 02:00 UTC | Syncs the model registry and sends one 5-token request per listed model; commits `models_status.yaml`. No benchmarks |
| `weekly-benchmark.yml` | Sundays (quick), 1st of month (full), or manually | Benchmarks all (or only new / selected) models, judges, merges into the rolling `reports/results_latest.json`, rebuilds the dashboard + README leaderboard, commits, deploys to Pages |

New models are merged per (model, prompt), so an "only new models" run adds them to the
leaderboard without discarding everyone else. Models that disappear from the API are dropped.

**One-time setup:** set the `BLABLADOR_API_KEY` repository secret and enable
*Settings → Pages → Source: GitHub Actions*.
