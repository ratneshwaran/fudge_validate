"""Regression tests for the --results spec parsing (bare path used to crash)."""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "experiments"))

import length_matched_reanalysis as lmr  # noqa: E402


def test_parse_result_specs_label_form():
    assert lmr.parse_result_specs(["seg=a/b.json"]) == [("seg", "a/b.json")]


def test_parse_result_specs_bare_path():
    # regression: this form crashed with FileNotFoundError('') before the fix
    assert lmr.parse_result_specs(["a/my_results.json"]) == \
        [("my_results", "a/my_results.json")]


def test_parse_result_specs_mixed():
    assert lmr.parse_result_specs(["x=1.json", "dir/2.json"]) == \
        [("x", "1.json"), ("2", "dir/2.json")]
