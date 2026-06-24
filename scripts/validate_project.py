"""Run production-readiness validation.

Usage:
    python scripts/validate_project.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_step(name: str, command: list[str], env: dict[str, str] | None = None, timeout: int = 240) -> bool:
    print(f"\n== {name} ==", flush=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            env=merged_env,
            timeout=timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        if output:
            print(output, end="" if output.endswith("\n") else "\n", flush=True)
        print(f"Timed out after {timeout} seconds.", flush=True)
        return False
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n", flush=True)
    print(f"Exit code: {result.returncode}", flush=True)
    return result.returncode == 0


def main() -> int:
    results: list[tuple[str, bool]] = []
    results.append(("Generate sample data", run_step("Generate sample data", [sys.executable, "sample_data/generate_sample_sales_data.py"], timeout=60)))
    results.append(("Compile Python files", run_step("Compile Python files", [sys.executable, "-m", "compileall", "-q", "."], timeout=120)))

    # Treat warnings as errors so CI fails if pandas/date parsing warnings return.
    pytest_env = {"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    pytest_base = [sys.executable, "-m", "pytest", "-q", "-W", "error"]
    test_chunks = [
        ["tests/test_basic_sanity_broad_question_set.py", "tests/test_basic_sanity_full_question_set.py", "tests/test_basic_sanity_questions.py"],
        ["tests/test_superstore_29_end_to_end_qa.py", "tests/test_superstore_39_end_to_end_qa.py", "tests/test_tier4_question_engine.py"],
        ["tests/test_dynamic_question_engine.py", "tests/test_tier2_question_engine.py", "tests/test_tier2_final_polish.py"],
        ["tests/test_tier3_question_engine.py", "tests/test_tier123_blackbox_fixes.py", "tests/test_final_reaudit_regressions.py"],
        ["tests/test_core.py", "tests/test_data_quality_business_signals.py", "tests/test_missing_value_treatment.py"],
        ["tests/test_must_fix_edge_cases.py", "tests/test_remaining_logic_edge_cases.py", "tests/test_final_batch_fixes.py"],
        ["tests/test_board_summary_formatting_fix.py", "tests/test_context_safe_root_cause.py"],
        ["tests/test_enhanced_features.py", "tests/test_manager_text_formatting_guard.py"],
        ["tests/test_ui_quality_fixes.py", "tests/test_ui_theme_and_llm.py"],
        ["tests/test_powerbi_exporter.py", "tests/test_enhanced_pdf_report.py", "tests/test_embedded_bi_dashboard.py"],
        ["tests/test_manager_demo_readiness.py"],
        # The full audit export test is isolated because report generation mutates global
        # document/font state in some CI runners. Running the audit checks one by one
        # keeps validation deterministic while still testing every section.
        ["tests/test_full_app_production_audit.py::test_section_1_upload_handler_behaviour"],
        ["tests/test_full_app_production_audit.py::test_sections_2_3_7_profile_column_detection_and_kpis"],
        ["tests/test_full_app_production_audit.py::test_section_3_alternate_column_names"],
        ["tests/test_full_app_production_audit.py::test_sections_5_6_8_data_quality_missing_treatment_and_eda"],
        ["tests/test_full_app_production_audit.py::test_sections_9_to_12_all_39_questions_have_required_grounding"],
        ["tests/test_full_app_production_audit.py::test_section_13_advanced_features_and_scenarios"],
        ["tests/test_full_app_production_audit.py::test_sections_14_15_dashboard_and_export_features"],
        ["tests/test_full_app_production_audit.py::test_section_16_no_api_mode_and_llm_fallback"],
    ]
    all_tests_ok = True
    for idx, chunk in enumerate(test_chunks, start=1):
        ok = run_step(f"Pytest chunk {idx}", [*pytest_base, *chunk], env=pytest_env, timeout=300)
        all_tests_ok = all_tests_ok and ok
    results.append(("Run pytest chunks with warnings-as-errors", all_tests_ok))

    print("\n== Summary ==")
    for name, ok in results:
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
    return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
