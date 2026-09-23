# Anonymizer prompt comparison

_Generated: 2026-09-23T13:13:46+00:00_

| Variant | p50 (ms) | p95 (ms) | mean (ms) | Pass rate | Recall | Leak rate | Footer |
|---|---|---|---|---|---|---|---|
| default | 1612.1 | 5213.2 | 2678.2 | 100.0% | 0.944 | 0.056 | yes |
| framed | 296.2 | 430.2 | 294.3 | 100.0% | 0.944 | 0.056 | yes |
| strict | 309.0 | 435.7 | 342.4 | 100.0% | 0.944 | 0.056 | yes |
| multilingual | 453.3 | 54460.0 | 20371.9 | 100.0% | 0.944 | 0.056 | yes |
| footer_off | 314.4 | 439.1 | 305.0 | 100.0% | 0.944 | 0.056 | no |

## default

_Body + entity footer, single user message (smoke-test contract)._

- **sample_short_en**: p50 5613.3 ms, p95 5613.3 ms, 67.9 chars/s, entities [9], pass 100%, recall 0.889, leaked: 12.03.2025
- **sample_german**: p50 1612.1 ms, p95 1612.1 ms, 189.2 chars/s, entities [8], pass 100%, recall 1.0
- **sample_no_pii**: p50 809.2 ms, p95 809.2 ms, 191.5 chars/s, entities [0], pass 100%

## framed

_System message carries the contract; user message is the bare text._

- **sample_short_en**: p50 445.1 ms, p95 445.1 ms, 856.0 chars/s, entities [9], pass 100%, recall 0.889, leaked: 12.03.2025
- **sample_german**: p50 296.2 ms, p95 296.2 ms, 1029.8 chars/s, entities [8], pass 100%, recall 1.0
- **sample_no_pii**: p50 141.7 ms, p95 141.7 ms, 1093.6 chars/s, entities [0], pass 100%

## strict

_Hardened wording: verbatim preservation, no commentary, no additions._

- **sample_short_en**: p50 449.7 ms, p95 449.7 ms, 847.2 chars/s, entities [9], pass 100%, recall 0.889, leaked: 12.03.2025
- **sample_german**: p50 309.0 ms, p95 309.0 ms, 987.2 chars/s, entities [8], pass 100%, recall 1.0
- **sample_no_pii**: p50 268.6 ms, p95 268.6 ms, 577.1 chars/s, entities [0], pass 100%

## multilingual

_Explicit language-preservation instructions for mixed-language documents._

- **sample_short_en**: p50 453.3 ms, p95 453.3 ms, 840.4 chars/s, entities [9], pass 100%, recall 0.889, leaked: 12.03.2025
- **sample_german**: p50 60460.7 ms, p95 60460.7 ms, 5.0 chars/s, entities [8], pass 100%, recall 1.0
- **sample_no_pii**: p50 201.6 ms, p95 201.6 ms, 768.8 chars/s, entities [0], pass 100%

## footer_off

_No entity footer: measures the latency/cost of the footer contract._

- **sample_short_en**: p50 452.9 ms, p95 452.9 ms, 841.2 chars/s, entities [9], pass 100%, recall 0.889, leaked: 12.03.2025
- **sample_german**: p50 314.4 ms, p95 314.4 ms, 970.0 chars/s, entities [8], pass 100%, recall 1.0
- **sample_no_pii**: p50 147.6 ms, p95 147.6 ms, 1050.1 chars/s, entities [0], pass 100%
