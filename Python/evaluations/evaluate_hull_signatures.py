"""Show how the number of independent signatures affects the convex hull.

We create sets of 3, 4, 6, 8 and 10 artificial, well-separated spectral curves
and sample them using an approximate 32-channel LSM 980 detector. Each set
produces noiseless pixels: one pure pixel for each signature and additional
random mixtures. We ask the hull selector to use first the same number of SVD
components (``npc``) as signatures, then one more.

This demonstrates a geometric principle: when the signatures are independent,
additional signatures can support additional hull dimensions. Asking for a
dimension that the data do not contain can make the hull fail. The idealized
curves and pure pixels make this a favorable case, not a test of how many real
fluorophores can be unmixed accurately.

Run from Python/: python -m evaluations.evaluate_hull_signatures
Each hull runs in a separate process with a per-case time limit.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from lumic.simulation import SpectralDetector
from ._execution import record_evaluation
from .evaluate_hull_fluorophores import run_case

SEED = 42
SIGNATURE_COUNTS = (3, 4, 6, 8, 10)
N_PIXELS = 60
GAUSSIAN_WIDTH_NM = 14.0
CENTER_RANGE_NM = (420.0, 690.0)
TIMEOUT_SECONDS = 8.0
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "hull_signatures"


def make_data(count):
    """Sample Gaussian signatures on the approximate 980 detector and mix them."""
    detector = SpectralDetector.from_profile("980_32ch")
    wavelengths = np.linspace(350, 750, 1601)
    centers = np.linspace(*CENTER_RANGE_NM, count)
    curves = np.exp(-0.5 * ((wavelengths[None, :] - centers[:, None]) / GAUSSIAN_WIDTH_NM) ** 2)
    spectra = np.asarray([detector.sample(wavelengths, curve) for curve in curves])
    spectra /= spectra.sum(axis=1, keepdims=True)
    rng = np.random.default_rng(SEED + count)
    fractions = np.vstack((np.eye(count), rng.dirichlet(np.ones(count), N_PIXELS - count)))
    return fractions @ spectra


@record_evaluation("output_dir")
def main(output_dir=OUTPUT_DIR, timeout=TIMEOUT_SECONDS):
    if not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    rows = []
    for count in SIGNATURE_COUNTS:
        data = make_data(count)
        singular_values = np.linalg.svd(data, compute_uv=False)
        rank = int(np.linalg.matrix_rank(data))
        for npc in (count, count + 1):
            status, vertices, seconds, detail = run_case(data, npc, timeout)
            row = dict(signatures=count, matrix_rank=rank, npc=npc, dimensions=npc - 1,
                       status=status, vertices=vertices, seconds=seconds, detail=detail,
                       smallest_signal_singular_value=singular_values[count - 1])
            rows.append(row)
            print(f"{count} signatures, npc={npc}: {status}, vertices={vertices}", flush=True)
    output_dir = Path(output_dir)
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Hull dimensions as the number of signatures increases", "",
             f"{N_PIXELS} noiseless pixels including one pure pixel per signature; "
             f"synthetic Gaussian spectra sampled by the approximate 980_32ch detector. "
             f"Each case has a {timeout:g} s limit including process startup.", "",
             "| Signatures | Matrix rank | npc | Hull dimensions | Outcome | Vertices |",
             "| ---: | ---: | ---: | ---: | --- | ---: |"]
    for row in rows:
        lines.append(f"| {row['signatures']} | {row['matrix_rank']} | {row['npc']} | "
                     f"{row['dimensions']} | {row['status']} | {row['vertices']} |")
    lines += ["", "The well-separated synthetic spectra and pure pixels form an idealized case. "
              "A timeout does not prove that a hull is impossible. This tests hull selection, "
              "not recovery accuracy or how many fluorophores can be separated in an acquisition."]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--timeout", type=float, default=TIMEOUT_SECONDS)
    args = parser.parse_args()
    main(args.output_dir, args.timeout)
