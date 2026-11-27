"""Explore how noise affects automatic selection of fluorophore spectra.

We take three saved FPbase emission spectra, sample them using an approximate
32-channel LSM 980 detector, and create pixels containing random mixtures of
the three fluorophores. We repeat the same experiment after adding small random
measurement variations. For each data set, we ask the convex-hull selector to
use progressively more SVD components (``npc``).

The noiseless mixtures contain only three independent spectral signatures.
Asking for more dimensions can make the hull fail because the data do not fill
those dimensions. Noise adds variation, so a higher-dimensional hull may run,
but can select many pixels or become slow. This is a simplified simulation:
it does not measure unmixing accuracy or the number of fluorophores that can
be distinguished in a real experiment.

Run from Python/: python -m evaluations.evaluate_hull_fluorophores
Each hull runs in a separate process with a per-case time limit.
"""
import argparse
import contextlib
import csv
import io
import multiprocessing as mp
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial import QhullError
from threadpoolctl import threadpool_limits

from lumic import essential_spectral_pixels
from lumic.simulation import SpectralDetector, load_offline_emission_spectrum
from ._execution import record_evaluation

SEED = 42
N_PIXELS = 150
N_CHANNELS = 32
FLUOROPHORES = ("EBFP2", "EGFP", "mCherry")
DETECTOR_PROFILE = "980_32ch"
NOISE_FRACTION_OF_PEAK = 0.001
NPC_VALUES = (3, 4, 6, 8, 12, 16, 24, 32)
TIMEOUT_SECONDS = 8.0
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "hull_fluorophores"


def make_data():
    """Sample offline FPbase spectra and make random nonnegative mixtures."""
    rng = np.random.default_rng(SEED)
    detector = SpectralDetector.from_profile(DETECTOR_PROFILE)
    spectra = np.asarray([detector.sample(*load_offline_emission_spectrum(name))
                          for name in FLUOROPHORES])
    spectra /= spectra.sum(axis=1, keepdims=True)
    clean = rng.dirichlet([1, 1, 1], N_PIXELS) @ spectra
    noise = rng.normal(0, NOISE_FRACTION_OF_PEAK * clean.max(), clean.shape)
    return {"noiseless": clean, "noisy": np.maximum(clean + noise, 0)}


def _hull_worker(data, npc, connection):
    """Send a compact result, including Qhull failures, to the parent process."""
    try:
        np.random.seed(SEED)
        with threadpool_limits(limits=1), contextlib.redirect_stdout(io.StringIO()):
            start = perf_counter()
            _, indices = essential_spectral_pixels(data, npc=npc)
        connection.send(("completed", len(indices), perf_counter() - start, ""))
    except QhullError as error:
        connection.send(("qhull_error", "", "", str(error).splitlines()[0]))
    except Exception as error:
        connection.send(("error", "", "", f"{type(error).__name__}: {error}"))
    finally:
        connection.close()


def run_case(data, npc, timeout):
    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_hull_worker, args=(data, npc, child))
    process.start()
    child.close()
    # The time limit includes process startup and library imports.
    if parent.poll(timeout):
        try:
            result = parent.recv()
        except EOFError:
            result = ("error", "", "", "Worker exited without a result")
        process.join()
    else:
        process.terminate()
        process.join()
        result = ("timeout", "", "", f"Exceeded {timeout:g} seconds including startup")
    parent.close()
    return result


@record_evaluation("output_dir")
def main(output_dir=OUTPUT_DIR, timeout=TIMEOUT_SECONDS):
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    output_dir = Path(output_dir)
    rows = []
    for label, data in make_data().items():
        rank = int(np.linalg.matrix_rank(data))
        for npc in NPC_VALUES:
            status, vertices, seconds, detail = run_case(data, npc, timeout)
            row = dict(data=label, matrix_rank=rank, npc=npc, hull_dimensions=npc-1,
                       status=status, vertices=vertices, seconds=seconds, detail=detail)
            rows.append(row)
            print(f"{label:9s} npc={npc:2d}: {status}, vertices={vertices}", flush=True)
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Convex hull dimensions: three simulated signatures", "",
             f"{N_PIXELS} random mixtures of {', '.join(FLUOROPHORES)}, "
             f"approximate {DETECTOR_PROFILE} detector, seed {SEED}; "
             f"noise standard deviation = {NOISE_FRACTION_OF_PEAK:g} × maximum noiseless value. "
             f"Each case has a {timeout:g} s limit including worker startup.", "",
             "| Data | Matrix rank | npc | Hull dimensions | Outcome | Vertices |",
             "| --- | ---: | ---: | ---: | --- | ---: |"]
    for row in rows:
        lines.append(f"| {row['data']} | {row['matrix_rank']} | {row['npc']} | "
                     f"{row['hull_dimensions']} | {row['status']} | {row['vertices']} |")
    lines += ["", "A timeout is not proof that a hull cannot be computed. "
              "Noise can create extra numerical dimensions without adding fluorophores. "
              "This experiment does not evaluate unmixing or establish a fluorophore limit."]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--timeout", type=float, default=TIMEOUT_SECONDS,
                        help="seconds per case including process startup")
    args = parser.parse_args()
    main(args.output_dir, args.timeout)
