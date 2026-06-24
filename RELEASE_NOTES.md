# Release Notes — Manager Demo Ready Build

## Release summary

This build packages the AI Sales Analyst Copilot as a manager-demo-ready and GitHub-upload-ready project.

## Added in this release

- Manager demo guide
- GitHub upload guide
- Project report
- Architecture documentation
- Feature matrix
- Demo checklist
- Demo question list
- Demo script
- GitHub Actions CI workflow
- MIT license
- Contribution guide
- Changelog
- Streamlit theme configuration
- Windows and macOS/Linux demo launch scripts
- Validation script
- Manager-demo readiness tests

## Validation

- `python sample_data/generate_sample_sales_data.py` — PASS
- `python -m compileall .` — PASS
- `python scripts/validate_project.py` — PASS with warnings treated as errors
- `streamlit run app.py --server.headless true` — PASS

## Recommended first GitHub commit message

```text
Initial manager-demo release: AI Sales Analyst Copilot
```
