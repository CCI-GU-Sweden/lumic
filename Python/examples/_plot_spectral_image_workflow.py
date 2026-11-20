"""Figure layout for the readable spectral_image_workflow example."""
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure


def save_workflow_figure(image, restored, maps, reconstruction, errors, *,
                         inside, expected_counts, shot_noise, output_path):
    """Compare image kinds, component maps and the fitted spectral residual.

    Inputs follow the main example's YXC/YXK layouts. The same RGB scale is used
    throughout; all diagnostic values were calculated from unscaled float data.
    This helper only plots existing results and performs no fitting.
    """
    figure = Figure(figsize=(16, 10), dpi=150)
    axes = figure.subplots(2, 4)
    figure.subplots_adjust(left=.03, right=.97, top=.86, bottom=.12,
                           hspace=.48, wspace=.40)
    display_limits = image.to_rgb().limits
    for ax, result, title in zip(
        axes[0, :3], [image, restored, reconstruction],
        ["Simulated measurement\nkind = measured", "Restored pixels\nkind = measured",
         "Fitted reconstruction\nkind = reconstructed"],
    ):
        rendering = result.to_rgb(limits=display_limits)
        rgb = rendering.data
        rgb[~inside] = 255
        ax.imshow(rgb)
        ax.set_title(title)
        ax.text(.5, .98, maps.component_names[0], transform=ax.transAxes,
                ha="center", va="top", fontsize=9)
        ax.text(.02, .02, maps.component_names[1], transform=ax.transAxes,
                va="bottom", fontsize=9)
        ax.text(.98, .02, maps.component_names[2], transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9)

    for k, name in enumerate(maps.component_names):
        panel = axes[1, k].imshow(maps.data[..., k], cmap="viridis",
                                   vmin=0, vmax=expected_counts)
        axes[1, k].set_title(f"{name} contribution\nkind = abundance; axes = {maps.axes}")
        figure.colorbar(panel, ax=axes[1, k], shrink=.7, label="Estimated counts")

    # A fixed fraction of signal sets a minimum scale so roundoff is not
    # displayed as large error when shot noise is disabled.
    residual_limit = max(errors.rms.max(), expected_counts * 1e-9)
    panel = axes[0, 3].imshow(errors.rms, cmap="magma", vmin=0, vmax=residual_limit)
    axes[0, 3].set_title("Restored − reconstructed\nRMS across detector channels")
    figure.colorbar(panel, ax=axes[0, 3], shrink=.7, label="Counts/channel")

    # One interior pixel shows why equal-shaped spectral images can differ.
    row, column = np.array(inside.shape) // 2
    axes[1, 3].plot(image.wavelengths_nm, restored.data[row, column],
                    ".-", label="Restored measurement", linewidth=1)
    axes[1, 3].plot(image.wavelengths_nm, reconstruction.data[row, column],
                    label="Fitted reconstruction", linewidth=1.5)
    axes[1, 3].set_title(f"One pixel: y={row}, x={column}")
    axes[1, 3].set_xlabel("Wavelength (nm)")
    axes[1, 3].set_ylabel("Counts/channel")
    axes[1, 3].legend(fontsize=8)
    for ax in axes.flat:
        if ax is not axes[1, 3]:
            ax.set_axis_off()

    figure.suptitle("SpectralImage → unfold → restore / unmix / reconstruct", fontsize=16)
    noise_label = "Poisson shot noise" if shot_noise else "no noise"
    figure.text(.5, .04,
                f"Offline FPbase: {' / '.join(maps.component_names)} · approximate 980_32ch detector · {noise_label}\n"
                "Restoration preserves the measured pixels exactly. Reconstruction predicts their signal from component contributions.",
                ha="center", fontsize=10)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, facecolor="white")
