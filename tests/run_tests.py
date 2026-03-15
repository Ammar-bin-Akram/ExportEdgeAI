"""
Run the full test suite from the Code/ directory.

Usage (from Code/):
    python tests/run_tests.py          # default: verbose output
    python tests/run_tests.py --html   # also generate HTML report (requires pytest-html)
"""
import sys
from pathlib import Path

# Ensure Code/ is on sys.path
CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))


def main():
    import pytest

    args = [
        str(CODE_DIR / "tests"),
        "-v",
        "--tb=short",
        "-rA",                       # show summary of All test outcomes
        "--color=yes",
    ]

    if "--html" in sys.argv:
        report_path = CODE_DIR / "tests" / "report.html"
        args += [f"--html={report_path}", "--self-contained-html"]
        print(f"HTML report will be saved to {report_path}")

    sys.exit(pytest.main(args))


if __name__ == "__main__":
    main()
