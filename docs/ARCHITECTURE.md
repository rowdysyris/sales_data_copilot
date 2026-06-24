# Architecture

## Design principle

The application is evidence-first. LLMs are optional and may only improve wording; all metrics and recommendations are grounded in Pandas calculations.

## Data flow

```text
User CSV/Excel
→ load_dataset()
→ detect_columns()
→ build_semantic_model()
→ _run_core()
→ analysis modules
→ app tabs + reports + Q&A
```

## Core services

- **Ingestion:** file loading, cleanup, metadata.
- **Column detection:** maps dataset columns to business roles.
- **Semantic layer:** determines dataset type, metrics, dimensions, available analyses.
- **Analytics engines:** KPI, EDA, relationships, profit, customer, product, region.
- **Brain layer:** entity resolution, planning, answer generation.
- **Exports:** Excel/PDF manager reports.

## Q&A pipeline

```text
question
→ semantic model
→ entity resolver
→ analysis planner
→ analyst brain
→ deterministic calculations
→ evidence tables
→ optional LLM wording
→ answer with limitations
```

## Safety and reliability rules

- No unsafe `eval` or arbitrary code execution.
- No raw dataset sent to LLM.
- Missing columns produce limitations, not crashes.
- Forecasts/scenarios state assumptions.
- Tests cover core manager questions.
