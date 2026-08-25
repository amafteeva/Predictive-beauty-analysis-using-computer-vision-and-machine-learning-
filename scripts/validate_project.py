"""Static validation that does not require the project's heavy dependencies."""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def validate_python_files() -> list[str]:
    errors: list[str] = []
    for path in sorted((ROOT / "src").rglob("*.py")) + sorted(
        (ROOT / "tests").rglob("*.py")
    ):
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as error:
            errors.append(f"{path.relative_to(ROOT)}: {error}")
    return errors


def validate_notebooks() -> list[str]:
    errors: list[str] = []
    notebook_paths = sorted((ROOT / "notebooks").glob("*.ipynb")) + sorted(
        (ROOT / "experiments").glob("*.ipynb")
    )
    for path in notebook_paths:
        relative = path.relative_to(ROOT)
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            errors.append(f"{relative}: invalid JSON: {error}")
            continue

        if document.get("nbformat") != 4:
            errors.append(f"{relative}: expected nbformat 4")
        cells = document.get("cells", [])
        if not cells or cells[0].get("cell_type") != "markdown":
            errors.append(f"{relative}: first cell must explain the notebook")

        for cell_number, cell in enumerate(cells):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            try:
                compile(source, f"{relative}:cell-{cell_number}", "exec")
            except SyntaxError as error:
                errors.append(f"{relative}: cell {cell_number}: {error}")
            if cell.get("outputs"):
                errors.append(f"{relative}: cell {cell_number} contains stale output")
    return errors


def validate_required_files() -> list[str]:
    required = [
        "README.md",
        "requirements.txt",
        "pyproject.toml",
        "src/foundation_matcher/__init__.py",
        "notebooks/01_foundation_matcher_demo.ipynb",
        "notebooks/02_fairface_pipeline_evaluation.ipynb",
        "notebooks/03_shade_clustering.ipynb",
        "experiments/04_review_satisfaction_baseline.ipynb",
    ]
    return [f"Missing required file: {path}" for path in required if not (ROOT / path).is_file()]


def main() -> None:
    errors = validate_required_files() + validate_python_files() + validate_notebooks()
    if errors:
        raise SystemExit("\n".join(errors))
    print("Static validation passed: Python syntax, notebook JSON, and code cells.")


if __name__ == "__main__":
    main()
