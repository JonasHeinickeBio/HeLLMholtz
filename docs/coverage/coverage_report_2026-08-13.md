# Test Coverage Report

**Date**: 2026-08-13
**Project**: helmholtz_llm_suite
**Target**: 80% coverage
**Achieved**: 82%

## Coverage Summary

```
TOTAL                                             3290    503   1034     76    82%
```

## Detailed Coverage by Module

### Core Modules (Excellent Coverage)

| Module | Coverage | Status |
|--------|----------|--------|
| core/config.py | 91% | ✅ |
| core/exporters.py | 97% | ✅ |
| core/prompts.py | 97% | ✅ |

### CLI Modules (Good Coverage)

| Module | Coverage | Status |
|--------|----------|--------|
| cli/__init__.py | 85% | ✅ |
| cli/benchmark.py | 92% | ✅ |
| cli/chat.py | 88% | ✅ |
| cli/model_manager.py | 99% | ✅ |
| cli/models.py | 61% | ⚠️ Needs work |
| cli/setup.py | 18% | ❌ Low coverage |

### Provider Modules (Mixed Coverage)

| Module | Coverage | Status |
|--------|----------|--------|
| providers/blablador_provider.py | 94% | ✅ |
| providers/litellm.py | 87% | ✅ |
| providers/monitoring.py | 89% | ✅ |
| providers/monitoring_core.py | 96% | ✅ |
| providers/blablador_config.py | 49% | ❌ Needs work |

### Reporting Modules (Excellent Coverage)

| Module | Coverage | Status |
|--------|----------|--------|
| reporting/chart.py | 99% | ✅ |
| reporting/html.py | 94% | ✅ |
| reporting/stats.py | 97% | ✅ |

### Benchmark & Evaluation Modules (Good Coverage)

| Module | Coverage | Status |
|--------|----------|--------|
| benchmark/evaluator.py | 87% | ✅ |
| benchmark/runner.py | 85% | ✅ |
| evaluation_analysis.py | 87% | ✅ |
| export.py | 97% | ✅ |

### Integration Modules (Good Coverage)

| Module | Coverage | Status |
|--------|----------|--------|
| integrations/lm_eval.py | 86% | ✅ |
| monitoring.py | 97% | ✅ |

### Client Module (Excellent Coverage)

| Module | Coverage | Status |
|--------|----------|--------|
| client.py | 95% | ✅ |

## Coverage Analysis

### High Coverage Modules (90%+)
- 12 modules achieve 90%+ coverage
- Core functionality is well-tested

### Moderate Coverage Modules (70-90%)
- 6 modules achieve 70-90% coverage
- Main functionality covered, some edge cases missing

### Low Coverage Modules (<70%)
- `cli/models.py` (61%): CLI command implementation needs tests
- `cli/setup.py` (18%): Setup command flows need tests
- `providers/blablador_config.py` (49%): Model configuration needs tests

## Recommendations

1. **Add CLI tests**: Focus on `cli/models.py` and `cli/setup.py`
2. **Add provider tests**: Focus on `providers/blablador_config.py` edge cases
3. **Integration tests**: Add more end-to-end tests for CLI interactions

## Test Suite Summary

- Total test files: 23
- Total tests: 193
- Passed: 192
- XFAIL: 1
- Skipped: 0
- Failed: 0
