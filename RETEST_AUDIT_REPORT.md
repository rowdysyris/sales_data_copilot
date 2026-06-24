# Retest Audit Report

Project: AI Sales Analyst Copilot
Base ZIP: ai_sales_analyst_copilot_final_no_warnings.zip
Retest scope: 16-section senior-engineer audit prompt covering upload, profiling, column detection, manual overrides, data quality, missing-value treatment, KPI engine, EDA, Tier 1-4 Q&A, advanced intelligence, dashboard, exports, no-API/LLM mode, cleanup, and GitHub readiness.

## Validation Method

Tests were executed with warnings treated as errors:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -W error <test chunks>
```

This means pandas warnings, datetime warnings, deprecation warnings, and runtime warnings fail the test run instead of being ignored.

## Results

- Compile check: PASS
- Warning policy: PASS, tests were run with `-W error`
- Total tests executed: 138
- Total failures: 0
- Warning failures: 0
- Code changes required during retest: 0

## Section Status

- SECTION 1 — Dataset Upload: PASS
- SECTION 2 — Dataset Profiling: PASS
- SECTION 3 — Smart Column Detection: PASS
- SECTION 4 — Manual Column Override: PASS
- SECTION 5 — Data Quality Analysis: PASS
- SECTION 6 — Missing Value Treatment: PASS
- SECTION 7 — KPI Engine: PASS
- SECTION 8 — Auto EDA: PASS
- SECTION 9 — Tier 1 Q1-Q15: PASS
- SECTION 10 — Tier 2 Q16-Q22: PASS
- SECTION 11 — Tier 3 Q23-Q29: PASS
- SECTION 12 — Tier 4 Q30-Q39: PASS
- SECTION 13 — Advanced Features: PASS
- SECTION 14 — Dashboard: PASS
- SECTION 15 — Export Features: PASS
- SECTION 16 — No API Mode: PASS
- Cleanup / GitHub readiness: PASS

## Notes

The full test suite was executed in chunks because this environment terminates long-running shell commands. Each chunk was run independently with warnings treated as errors. The chunked approach matches the included validation script design and avoids environment timeout limits while still covering every test file.

No source-code fix was needed after this retest.
