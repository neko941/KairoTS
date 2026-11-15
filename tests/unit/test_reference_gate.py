"""Missing reference inputs are optional locally and failures in the strict gate."""

from pathlib import Path

import pytest


@pytest.mark.parametrize("strict", [False, True])
def test_reference_skips_fail_only_in_strict_gate(
    *, pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, strict: bool
) -> None:
    monkeypatch.setenv("KAIROTS_REQUIRE_REFERENCES", "1" if strict else "0")
    pytester.makeconftest(
        (Path(__file__).parents[1] / "conftest.py").read_text(encoding="utf-8")
    )
    pytester.makepyfile(
        test_models=(
            "import pytest\ndef test_reference(): pytest.skip('reference absent')"
        )
    )
    result = pytester.runpytest_subprocess("-q", "-p", "no:cacheprovider")
    if strict:
        result.assert_outcomes(failed=1)
        result.stdout.fnmatch_lines(["*required reference check skipped*"])
    else:
        result.assert_outcomes(skipped=1)
