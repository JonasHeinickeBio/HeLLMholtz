# Model performance

_Updated 2026-11-05 · 41 models · 58 prompts · judges: alias-deepseek-v41-flash-booster, alias-huge_

- **Best overall:** `deepseek-v4.1-flash-at-hzdr` with **98.4** points
- **Fastest of the top group (within 5 points of #1):** `alias-fast` - 0.14 s to first token, 209 tok/s
- **Best quality/speed balance (70 % quality, 30 % speed):** `deepseek-v4.1-flash-at-hzdr` - 98.4 points, speed 96

Scores are 0-100: each prompt blends deterministic checks (exact answers, executed unit
tests, format rules) with a blind LLM-judge panel; failed runs count as 0. Speed comes from
a separate streamed probe per model (time to first token and decode throughput) and never
changes the quality score. Details: [Benchmarking](https://jonasheinickebio.github.io/HeLLMholtz/benchmarking/). The interactive version with
filters, charts and per-prompt drill-down is the [live dashboard](https://jonasheinickebio.github.io/HeLLMholtz/benchmark/).

## Which model for which job

| Use case | Model | Score | TTFT / throughput |
|:--|:--|:--|--:|
| Highest quality | `deepseek-v4.1-flash-at-hzdr` | 98.4 | 0.41 s / 451 tok/s |
| Best fast model (≥ 150 tok/s) | `deepseek-v4.1-flash-at-hzdr` | 98.4 | 0.41 s / 451 tok/s |
| Best mid-speed model (60-150 tok/s) | `alias-mimo-v2.6-flash` | 98.2 | – / 74 tok/s |
| Coding | `deepseek-v4.1-flash-at-hzdr` | 98.4, coding 100 | 0.41 s / 451 tok/s |
| Math | `deepseek-v4.1-flash-at-hzdr` | 98.4, math 100 | 0.41 s / 451 tok/s |
| Reasoning | `alias-hyperqwen` | 97.8, reasoning 100 | 0.26 s / 50 tok/s |
| Medical | `alias-glm-5.3-flash` | 94.5, medical 100 | 0.30 s / 78 tok/s |
| Multilingual | `alias-mimo-v2.6-flash` | 98.2, multilingual 100 | – / 74 tok/s |
| Long context | `deepseek-v4.1-flash-at-hzdr` | 98.4, long context 100 | 0.41 s / 451 tok/s |

## Leaderboard

| # | Model | Family | Score | Speed | TTFT s | Tok/s | Reliab. |
|:--|:--|:--|--:|--:|--:|--:|--:|
| 🥇 | `deepseek-v4.1-flash-at-hzdr` | DeepSeek | **98.4** | 96 | 0.41 | 451 | 100% |
| 🥈 | `alias-mimo-v2.6-flash` | MiMo | **98.2** | 67 | – | 74 | 100% |
| 🥉 | `alias-mimo-v2.6-pro` | MiMo | **97.9** | 66 | 0.79 | 60 | 100% |
| 4 | `alias-hyperqwen` | Qwen | **97.8** | 77 | 0.26 | 50 | 98% |
| 5 | `DeepSeek-V4.1-Flash on JUWELS Booster` | DeepSeek | **97.7** | 28 | 5.22 | 31 | 100% |
| 6 | `alias-qwen3.8-flash-next` | Qwen | **97.5** | 93 | 0.15 | 130 | 98% |
| 7 | `alias-qwen38-quick` | Qwen | **97.4** | 75 | 0.21 | 46 | 98% |
| 8 | `alias-code` | Other | **97.3** | 62 | 0.40 | 27 | 98% |
| 9 | `alias-deepseek-v41-flash-booster` | DeepSeek | **97.0** | 63 | 0.52 | 35 | 100% |
| 10 | `alias-large` | Other | **97.0** | 88 | 0.15 | 97 | 98% |
| 11 | `MiMo-V2.6-Pro-RL on Juwels Booster` | MiMo | **96.9** | 71 | 0.69 | 70 | 100% |
| 12 | `alias-qwen38-27b` | Qwen | **96.9** | 60 | 0.35 | 20 | 98% |
| 13 | `Qwen3.8-27B-FP8` | Qwen | **96.7** | 72 | 0.24 | 37 | 96% |
| 14 | `Aleph Kolibri` | Other | **96.7** | 54 | 2.30 | 74 | 98% |
| 15 | `Qwen 3.8-Flash-Next-125B-AWQ-W4A16 on Juwels Booster` | Qwen | **96.6** | 90 | 0.29 | 113 | 96% |

_Top 15 of 41; the full ranking is in the dashboard._

## Best per category

| Category | 1st | 2nd | 3rd |
|:--|:--|:--|:--|
| Coding | `deepseek-v4.1-flash-at-hzdr` (100) | `DeepSeek-V4.1-Flash on JUWELS Booster` (100) | `alias-deepseek-v41-flash-booster` (100) |
| Creative | `DeepSeek-V4.1-Flash on JUWELS Booster` (100) | `alias-qwen3.8-flash-next` (100) | `alias-qwen38-quick` (100) |
| Extraction | `deepseek-v4.1-flash-at-hzdr` (100) | `alias-mimo-v2.6-flash` (100) | `alias-mimo-v2.6-pro` (100) |
| Instruction | `deepseek-v4.1-flash-at-hzdr` (96) | `alias-mimo-v2.6-flash` (96) | `alias-mimo-v2.6-pro` (96) |
| Knowledge | `Qwen 3.8-Flash-Next-125B-AWQ-W4A16 on Juwels Booster` (100) | `alias-medgemma` (100) | `deepseek-v4.1-flash-at-hzdr` (96) |
| Long context | `deepseek-v4.1-flash-at-hzdr` (100) | `alias-mimo-v2.6-flash` (100) | `alias-mimo-v2.6-pro` (100) |
| Math | `deepseek-v4.1-flash-at-hzdr` (100) | `alias-mimo-v2.6-flash` (100) | `alias-mimo-v2.6-pro` (100) |
| Medical | `alias-glm-5.3-flash` (100) | `Mistral 4 Large` (100) | `MiMo-V2.6-Pro-RL on Juwels Booster` (99) |
| Multilingual | `alias-mimo-v2.6-flash` (100) | `alias-mimo-v2.6-pro` (100) | `alias-hyperqwen` (100) |
| Reasoning | `alias-hyperqwen` (100) | `alias-qwen38-quick` (100) | `Qwen3.8-27B-FP8` (100) |
| Safety | `deepseek-v4.1-flash-at-hzdr` (100) | `alias-mimo-v2.6-flash` (100) | `alias-hyperqwen` (100) |

## Speed ranking

Speed score = mean of throughput (10 tok/s = 0, 200 tok/s = 100) and time to first token (10 s = 0, 0.3 s = 100), both log-scaled.

| # | Model | Speed | TTFT s | Tok/s | Latency s | Score |
|:--|:--|--:|--:|--:|--:|--:|
| 1 | `alias-fast` | 100 | 0.14 | 209 | 1.5 | 94.5 |
| 2 | `deepseek-v4.1-flash-at-hzdr` | 96 | 0.41 | 451 | 1.0 | 98.4 |
| 3 | `alias-qwen3.8-flash-next` | 93 | 0.15 | 130 | 3.3 | 97.5 |
| 4 | `Qwen 3.8-Flash-Next-125B-AWQ-W4A16 on Juwels Booster` | 90 | 0.29 | 113 | 3.3 | 96.6 |
| 5 | `alias-large` | 88 | 0.15 | 97 | 14.1 | 97.0 |
| 6 | `alias-glm-5.3-flash` | 84 | 0.30 | 78 | 18.4 | 94.5 |
| 7 | `alias-aleph-kolibri-1` | 83 | 0.15 | 74 | 6.6 | 92.0 |
| 8 | `alias-hyperqwen` | 77 | 0.26 | 50 | 12.1 | 97.8 |
| 9 | `alias-hyperqwen-27b` | 77 | 0.25 | 50 | 9.7 | 96.1 |
| 10 | `alias-qwen38-quick` | 75 | 0.21 | 46 | 12.2 | 97.4 |
| 11 | `Qwen 3.8-27B on Juwels Booster` | 75 | 0.19 | 46 | 6.5 | 96.4 |
| 12 | `alias-qwen3.8-27b` | 75 | 0.22 | 46 | 16.0 | 96.0 |

## Thinking behaviour

Share of output tokens spent on reasoning and how often a model ran into the token limit before finishing.

_Thinking-token counts are not in this run yet (they are recorded from the next benchmark on); models that ran into the token limit:_

| Model | Hit token limit | Reasoning score |
|:--|--:|--:|
| `alias-aleph-kolibri-1` | 6% | 95 |
| `Qwen 3.8-Flash-Next-125B-AWQ-W4A16 on Juwels Booster` | 4% | 95 |
| `alias-qwen3.8-flash-next` | 2% | 95 |
| `alias-large` | 2% | 95 |
| `Aleph Kolibri` | 2% | 95 |

## What changed

_No new models or score changes since the previous run._
