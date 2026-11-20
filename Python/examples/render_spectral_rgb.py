"""Render a spectral CZI as RGB and select an independent copy of one plane.

Run from Python/: python -m examples.render_spectral_rgb
Uses public COS-7 data by default. Supply a Z stack to compare planes on one scale.
"""
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from lumic.data import download_cos7_data
from lumic.io import load_czi


# 1. USER SETTINGS
FILE_PATH = None  # Default to the public COS-7 CZI, or supply your own path.
# To try a small Z stack:
# FILE_PATH = Path("data/io-data/980/ex405-range411_694-res8p8nm_zstack3.czi")
Z_SLICE = 0
OUTPUT_PATH = Path(__file__).resolve().parent / "output" / "spectral-rgb.png"

if FILE_PATH is None:
    image_path, = download_cos7_data(snr="high", progress=True)
else:
    image_path = Path(FILE_PATH).expanduser()


# 2. LOAD ALL Z SLICES AND RENDER WITH ONE SHARED DISPLAY SCALE
image = load_czi(image_path, z=None)
rendering = image.to_rgb()
print(f"Spectral image: axes={image.axes}, shape={image.data.shape}")
print(f"RGB rendering: axes={rendering.axes}, shape={rendering.data.shape}")
print(f"Shared display limits: {rendering.limits}")
print("This is a channel-order preview, not spectral unmixing or true color.")


# 3. COPY ONE PLANE, PRESERVING ITS METADATA AND KIND
plane = image.slice(z=Z_SLICE)
plane_rendering = plane.to_rgb(limits=rendering.limits)
print(f"Copied plane: axes={plane.axes}, shape={plane.data.shape}")
has_stack = "Z" in rendering.axes
rgb_planes = (np.moveaxis(rendering.data, rendering.axes.index("Z"), 0)
              if has_stack else rendering.data[np.newaxis])
# Reusing limits makes the copied plane match its position in the full rendering.
np.testing.assert_array_equal(plane_rendering.data, rgb_planes[Z_SLICE])
assert not np.shares_memory(plane.data, image.data)


# 4. SAVE THE RGB PREVIEW
figure = Figure(figsize=(10, 4), dpi=150)
axes = figure.subplots(1, len(rgb_planes), squeeze=False)[0]
for z, (ax, rgb) in enumerate(zip(axes, rgb_planes)):
    ax.imshow(rgb, interpolation="nearest")
    ax.set_title(f"Z = {z}" if has_stack else "Single spectral plane")
    ax.set_axis_off()
figure.suptitle("One shared RGB scale across channels and planes")
figure.text(.5, .03, "Channel-order preview only · not component abundance maps", ha="center")
figure.tight_layout(rect=(0, .07, 1, .93))
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(OUTPUT_PATH, facecolor="white")
print(f"Saved preview to {OUTPUT_PATH}")
