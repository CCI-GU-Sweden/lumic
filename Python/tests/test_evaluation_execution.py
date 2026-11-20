"""Prerequisite and completion contracts for standalone evaluation runs."""

import json

import pytest

from evaluations._execution import (
    prepare_prerequisites, print_case_summary, record_evaluation,
)


def test_fresh_run_builds_prerequisites_and_reuse_does_not_overwrite(tmp_path):
    calls = []

    @record_evaluation("output_dir")
    def baseline(output_dir):
        calls.append(output_dir)
        (output_dir / "result.txt").write_text("original result")

    stage = "test_evaluation_execution"
    arguments = dict(stage=stage, required=("result.txt",), build=lambda: baseline(tmp_path))
    prepare_prerequisites(tmp_path, **arguments)
    assert len(calls) == 1
    manifest = json.loads((tmp_path / f"run_{stage}.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["python"]
    assert "numpy" in manifest["packages"]
    assert manifest["arguments"]["output_dir"] == str(tmp_path)

    (tmp_path / "result.txt").write_text("reviewed result")
    prepare_prerequisites(tmp_path, reuse=True, **arguments)
    assert len(calls) == 1
    assert (tmp_path / "result.txt").read_text() == "reviewed result"

    # Rebuilding is the default, even if old artifacts are present.
    prepare_prerequisites(tmp_path, **arguments)
    assert len(calls) == 2
    assert (tmp_path / "result.txt").read_text() == "original result"


@pytest.mark.parametrize("status", [None, "running", "failed", "interrupted"])
def test_reuse_rejects_unfinished_runs_even_when_artifacts_exist(tmp_path, status):
    (tmp_path / "result.txt").write_text("partial output")
    if status is not None:
        (tmp_path / "run_baseline.json").write_text(json.dumps(dict(status=status)))
    with pytest.raises(ValueError, match="without --reuse-prerequisites"):
        prepare_prerequisites(tmp_path, stage="baseline", required=("result.txt",),
                              build=lambda: pytest.fail("reuse must not rebuild"), reuse=True)


def test_reuse_rejects_missing_artifacts_and_a_different_input(tmp_path):
    (tmp_path / "run_baseline.json").write_text(json.dumps(dict(status="completed")))
    arguments = dict(stage="baseline", required=("settings.json",),
                     build=lambda: pytest.fail("reuse must not rebuild"), reuse=True)
    with pytest.raises(ValueError, match="missing artifacts"):
        prepare_prerequisites(tmp_path, **arguments)
    (tmp_path / "settings.json").write_text(json.dumps(dict(input_path=str(tmp_path / "old.czi"))))
    with pytest.raises(ValueError, match="--input differs"):
        prepare_prerequisites(tmp_path, input_path=tmp_path / "new.czi", **arguments)


@pytest.mark.parametrize("error,status", [(RuntimeError("fit failed"), "failed"),
                                        (KeyboardInterrupt(), "interrupted")])
def test_failed_run_invalidates_previous_completion(tmp_path, error, status):
    manifest = tmp_path / "run_test_evaluation_execution.json"
    manifest.write_text(json.dumps(dict(status="completed")))

    @record_evaluation("output_dir")
    def fail(output_dir):
        raise error

    with pytest.raises(type(error)):
        fail(tmp_path)
    metadata = json.loads(manifest.read_text())
    assert metadata["status"] == status
    assert metadata["finished_at"]
    assert metadata["seconds"] >= 0


def test_case_summary_distinguishes_expected_failures_and_cap_hits(capsys):
    print_case_summary([
        dict(status="ok", stop="converged"),
        dict(status="ok", stop="max_iterations"),
        dict(status="failed: ValueError: unusable SVD coordinate"),
    ])
    output = capsys.readouterr().out
    assert "2/3 completed, 1 failed" in output
    assert "1 completed fits reached the iteration limit" in output
