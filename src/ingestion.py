from __future__ import annotations

from pathlib import Path
import pandas as pd
from src.utils import safe_numeric


def _read_csv(uploaded_file):
    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
    warnings = []
    for enc in encodings:
        try:
            if hasattr(uploaded_file, "seek"):
                uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding=enc), warnings
        except UnicodeDecodeError:
            warnings.append(f"CSV decoding failed with {enc}.")
        except Exception as exc:
            if enc == encodings[-1]:
                raise
            warnings.append(f"CSV read failed with {enc}: {exc}")
    raise ValueError("Could not read CSV.")


def _dedupe_columns(columns: list[str]) -> tuple[list[str], list[str]]:
    """Return unique column names and warnings for duplicates.

    Pandas allows duplicate column labels, but many downstream operations expect a
    single Series per name. We keep the first name unchanged and suffix later
    duplicates with __2, __3, etc.
    """
    seen: dict[str, int] = {}
    out: list[str] = []
    warnings: list[str] = []
    for raw in columns:
        base = str(raw).strip() or "Unnamed Column"
        # pandas CSV reader may already mangle duplicates as Profit.1, Profit.2.
        # Convert those back to the logical base name before applying our stable suffix.
        import re
        m = re.match(r"^(.*)\.(\d+)$", base)
        if m and m.group(1) in seen:
            base = m.group(1)
        count = seen.get(base, 0) + 1
        seen[base] = count
        if count == 1:
            out.append(base)
        else:
            new_name = f"{base}__{count}"
            out.append(new_name)
            warnings.append(f"Duplicate column '{base}' renamed to '{new_name}'.")
    return out, warnings


def _safe_numeric_conversion(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            numeric = safe_numeric(out[col])
            if numeric.notna().mean() >= 0.85:
                out[col] = numeric
    return out


def load_dataset(uploaded_file):
    """Load CSV/Excel dataset and return dataframe plus metadata."""
    if uploaded_file is None:
        raise ValueError("No file provided.")

    name = getattr(uploaded_file, "name", None) or str(uploaded_file)
    suffix = Path(name).suffix.lower()
    warnings = []

    if suffix == ".csv":
        df, csv_warnings = _read_csv(uploaded_file)
        warnings.extend(csv_warnings)
        file_type = "csv"
    elif suffix in {".xlsx", ".xls"}:
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        df = pd.read_excel(uploaded_file)
        file_type = "excel"
    else:
        raise ValueError("Unsupported file type. Upload CSV, XLSX, or XLS.")

    original_columns = list(df.columns)
    df = df.copy()
    df.columns, duplicate_warnings = _dedupe_columns([str(c).strip() for c in df.columns])
    warnings.extend(duplicate_warnings)
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df = _safe_numeric_conversion(df)

    metadata = {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "columns": list(df.columns),
        "original_columns": original_columns,
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "file_type": file_type,
        "load_warnings": warnings,
    }
    return df, metadata
