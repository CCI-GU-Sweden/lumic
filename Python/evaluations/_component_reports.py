"""Tables and plots shared by the exploratory binary and ternary runs."""

import csv

import numpy as np
from matplotlib.figure import Figure


def component_metric_rows(max_hidden_fraction, names, initial, final):
    """Keep one row per component so improvements cannot hide in a mean."""
    rows = []
    for index, name in enumerate(names):
        rows.append({
            "max_hidden_fraction": max_hidden_fraction,
            "component": name,
            "mcr_status": "completed" if final is not None else "failed",
            "initial_estimated_index": int(initial.component_order[index]),
            "final_estimated_index": (
                int(final.component_order[index]) if final is not None else ""
            ),
            "initial_matching_cost": initial.matching_costs[index],
            "final_matching_cost": (
                final.matching_costs[index] if final is not None else float("nan")
            ),
            "initial_spectral_relative_l2": initial.spectral_relative_l2_per_component[index],
            "final_spectral_relative_l2": (
                final.spectral_relative_l2_per_component[index]
                if final is not None else float("nan")
            ),
            "initial_abundance_rmse_counts": initial.abundance_rmse_per_component[index],
            "final_abundance_rmse_counts": (
                final.abundance_rmse_per_component[index]
                if final is not None else float("nan")
            ),
            "initial_abundance_relative_l2": initial.abundance_relative_l2_per_component[index],
            "final_abundance_relative_l2": (
                final.abundance_relative_l2_per_component[index]
                if final is not None else float("nan")
            ),
        })
    return rows


def save_component_metrics(path, rows):
    """Write a tidy table for comparing both scripts and all titration levels."""
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: f"{value:.6g}" if isinstance(value, (float, np.floating))
                else value for key, value in row.items()
            })


def save_component_recovery_figure(path, rows, names, hidden_component):
    """Show each component's spectral and count recovery before and after MCR."""
    fig = Figure(figsize=(4.4 * len(names), 7), dpi=150)
    axes = fig.subplots(2, len(names), squeeze=False)
    for index, name in enumerate(names):
        selected = [row for row in rows if row["component"] == name]
        levels = np.array([row["max_hidden_fraction"] * 100 for row in selected])
        for axis, metric, ylabel in (
            (axes[0, index], "spectral_relative_l2", "Spectral error, unit L2 (%)"),
            (axes[1, index], "abundance_relative_l2", "Abundance error (%)"),
        ):
            initial_values = np.array(
                [row[f"initial_{metric}"] for row in selected]
            )
            final_values = np.array(
                [row[f"final_{metric}"] for row in selected]
            )
            # Numerical round-off is still present in the CSV. On a figure it
            # would make an effectively zero error look like a trend.
            initial_values[np.abs(initial_values) < 1e-8] = 0
            final_values[np.abs(final_values) < 1e-8] = 0
            axis.plot(
                levels, initial_values * 100,
                "o--", label="Selected start",
            )
            axis.plot(
                levels, final_values * 100,
                "o-", label="After MCR-ALS",
            )
            axis.invert_xaxis()
            axis.set_ylim(bottom=0)
            axis.grid(alpha=0.2)
            axis.set_xlabel(f"Maximum {hidden_component} fraction (%)")
            axis.set_ylabel(ylabel)
            axis.legend(frameon=False, fontsize=8)
        axes[0, index].set_title(name)
    fig.suptitle("Recovery of each component across the hidden-fraction titration")
    if any(np.isnan(row["final_spectral_relative_l2"]) for row in rows):
        fig.text(
            0.5, 0.015, "Missing fitted points indicate an MCR-ALS run that failed.",
            ha="center", fontsize=9,
        )
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.09, top=0.88,
                        hspace=0.36, wspace=0.38)
    fig.savefig(path, facecolor="white", bbox_inches="tight")
