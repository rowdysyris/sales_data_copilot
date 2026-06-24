# Validation Log

Latest production validation:

- `python -m compileall .` — PASS
- Tier 1 sanity checks — PASS
- Tier 2 analyst questions — PASS
- Tier 3 hard questions — PASS
- Tier 4 messy real-world questions — PASS
- 29-question E2E audit — PASS
- 39-question E2E audit — PASS
- Production audit Sections 1–16 — PASS
- Export smoke tests — PASS
- Dashboard model/filter smoke tests — PASS
- No-API mode and LLM fallback tests — PASS

Use `python scripts/validate_project.py` to run the full local validation suite in stable chunks.
