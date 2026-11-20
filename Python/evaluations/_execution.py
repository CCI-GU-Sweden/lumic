"""Execution and provenance helpers for standalone evaluation scripts."""

from datetime import datetime, timezone
from functools import wraps
from importlib.metadata import PackageNotFoundError, version
import inspect
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter


def _git_output(source, *arguments):
    try:
        return subprocess.check_output(
            ["git", "-C", str(source.parent), *arguments],
            stderr=subprocess.DEVNULL, text=True, timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _json_value(value):
    if isinstance(value, Path):
        return str(value.resolve())
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    raise TypeError(f"Unsupported configuration value: {type(value).__name__}")


def record_evaluation(output_argument):
    """Record settings/environment and completion, including interrupted runs.

    Each script has its own manifest, so a dependent evaluation and the stages
    it prepares can share one results tree without overwriting provenance.
    """
    def decorate(function):
        signature = inspect.signature(function)
        source = Path(inspect.getfile(function)).resolve()
        name = source.stem

        @wraps(function)
        def recorded(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            output = Path(bound.arguments[output_argument]).resolve()
            output.mkdir(parents=True, exist_ok=True)
            constants = {}
            for key, value in function.__globals__.items():
                if key.isupper():
                    try:
                        constants[key] = _json_value(value)
                    except TypeError:
                        pass
            packages = {}
            for package in ("numpy", "scipy", "scikit-learn", "scikit-image", "matplotlib",
                            "tifffile", "pylibCZIrw", "threadpoolctl"):
                try:
                    packages[package] = version(package)
                except PackageNotFoundError:
                    packages[package] = None
            metadata = dict(
                evaluation=name, status="running", source=str(source),
                git_commit=_git_output(source, "rev-parse", "HEAD"),
                git_status=_git_output(source, "status", "--porcelain"),
                python=platform.python_version(), packages=packages,
                arguments=_json_value(bound.arguments), constants=constants,
                started_at=datetime.now(timezone.utc).isoformat(),
            )
            manifest = output / f"run_{name}.json"
            manifest.write_text(json.dumps(metadata, indent=2) + "\n")
            start = perf_counter()
            try:
                result = function(*args, **kwargs)
            except BaseException as error:
                metadata.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                error=f"{type(error).__name__}: {error}")
                raise
            else:
                metadata["status"] = "completed"
                print(f"Completed {name}. Results: {output}", flush=True)
                return result
            finally:
                metadata.update(finished_at=datetime.now(timezone.utc).isoformat(),
                                seconds=perf_counter() - start)
                manifest.write_text(json.dumps(metadata, indent=2) + "\n")

        return recorded
    return decorate


def prepare_prerequisites(root, *, stage, required, build, reuse=False, input_path=None):
    """Build a stage by default; reuse existing complete artifacts only on request."""
    root = Path(root)
    if reuse:
        manifest = root / f"run_{stage}.json"
        missing = [str(path) for path in required if not (root / path).is_file()]
        if missing or not manifest.is_file():
            raise ValueError(
                f"Cannot reuse {stage} in {root}: missing artifacts or completion manifest. "
                "Run without --reuse-prerequisites to generate them."
            )
        metadata = json.loads(manifest.read_text())
        if metadata.get("status") != "completed":
            raise ValueError(
                f"Cannot reuse incomplete {stage} in {root}. "
                "Run without --reuse-prerequisites to regenerate it."
            )
        if input_path is not None:
            settings = json.loads((root / "settings.json").read_text())
            if Path(settings["input_path"]).resolve() != Path(input_path).resolve():
                raise ValueError("--input differs from the saved prerequisite input; run without --reuse-prerequisites")
        print(f"Reusing {stage} in {root} (requested explicitly).", flush=True)
    else:
        print(f"Preparing {stage} in {root}...", flush=True)
        build()
        missing = [str(path) for path in required if not (root / path).is_file()]
        if missing:
            raise RuntimeError(f"{stage} did not produce required artifacts: {', '.join(missing)}")


def print_case_summary(rows):
    """Distinguish completed experiments from failures within their case sweeps."""
    failed = sum(str(row.get("status", "")).startswith("failed") for row in rows)
    capped = sum(row.get("stop", row.get("status")) == "max_iterations" for row in rows)
    print(f"Cases: {len(rows) - failed}/{len(rows)} completed, {failed} failed; "
          f"{capped} completed fits reached the iteration limit.", flush=True)
