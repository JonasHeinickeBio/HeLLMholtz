# System-One reliability benchmark

_Generated: 2026-09-30T16:17:46+00:00_

- **Model:** alias-laya
- **Endpoint:** https://laya.blablador.fz-juelich.de/v1/systemone
- **Replications per scenario:** 3
- **Overall success:** 100.0% (12/12)
- **Overall decision stability:** 1.00
- **Latency (all calls):** mean 554 ms, p95 661 ms

## Scenarios

| Scenario | Success | Mean (ms) | p95 | Stability | Confidence | Routing |
|---|---|---|---|---|---|---|
| fatigue_triage | 100.0% (3/3) | 576 | 612 | 1.00 | 0.60 | english |
| antibiotic_stewardship | 100.0% (3/3) | 512 | 513 | 1.00 | 0.11 | english |
| referral_urgency | 100.0% (3/3) | 648 | 707 | 1.00 | 0.10 | english |
| followup_frequency | 100.0% (3/3) | 478 | 508 | 1.00 | 0.00 | english |

### fatigue_triage

_State: Adult patient with persistent, severe post-exertional fatigue for over 12 months, not explained by another condition, presenting for an initial diagnostic work-up._

- **post_exertional_malaise** — Does the history strongly suggest post-exertional malaise?: majority **yes**, stability 1.00, answer rate 100%, counts: yes=3

### antibiotic_stewardship

_State: Patient with a suspected viral prodrome and low-grade fever who is requesting antibiotics at the visit._

- **antibiotics** — Should antibiotics be prescribed at this time?: majority **yes**, stability 1.00, answer rate 100%, counts: yes=3

### referral_urgency

_State: Patient with worsening neurological symptoms, including brain fog and orthostatic intolerance._

- **neurology** — Is a neurology referral warranted, and how urgent?: majority **urgent**, stability 1.00, answer rate 100%, counts: urgent=3
- **followup** — How soon should the next follow-up happen?: majority **week**, stability 1.00, answer rate 100%, counts: week=3

### followup_frequency

_State: Stable ME/CFS patient on a graded management plan, reviewing the care pathway._

- **frequency** — What follow-up cadence is appropriate?: majority **quarterly**, stability 1.00, answer rate 100%, counts: quarterly=3
