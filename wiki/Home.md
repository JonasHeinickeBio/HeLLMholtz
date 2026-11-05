# HeLLMholtz

Unified LLM access, benchmarking and reporting for the Helmholtz Blablador API.

## Current state

_Updated 2026-11-05 · 41 models · 58 prompts · judges: alias-deepseek-v41-flash-booster, alias-huge_

- **Best overall:** `deepseek-v4.1-flash-at-hzdr` with **98.4** points
- **Fastest of the top group (within 5 points of #1):** `alias-fast` - 0.14 s to first token, 209 tok/s
- **Best quality/speed balance (70 % quality, 30 % speed):** `deepseek-v4.1-flash-at-hzdr` - 98.4 points, speed 96

### Top 5

| # | Model | Score | Speed |
|:--|:--|--:|--:|
| 🥇 | `deepseek-v4.1-flash-at-hzdr` | **98.4** | 96 |
| 🥈 | `alias-mimo-v2.6-flash` | **98.2** | 67 |
| 🥉 | `alias-mimo-v2.6-pro` | **97.9** | 66 |
| 4 | `alias-hyperqwen` | **97.8** | 77 |
| 5 | `DeepSeek-V4.1-Flash on JUWELS Booster` | **97.7** | 28 |

## Pages

- [[Performance]] - leaderboard, best model per job, speed ranking
- [[Models]] - every known model with context size, availability and score
- [Live dashboard](https://jonasheinickebio.github.io/HeLLMholtz/benchmark/) - interactive charts and drill-down
- [Documentation](https://jonasheinickebio.github.io/HeLLMholtz/) - installation, usage, configuration
- [Source](https://github.com/JonasHeinickeBio/HeLLMholtz)

_This wiki is generated from the benchmark data by `scripts/build_docs.py`; changes made here are overwritten._
