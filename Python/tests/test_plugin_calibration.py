"""Physical registration of selected-plane viewer layers."""
from types import SimpleNamespace

import numpy as np
import pytest
from napari.components import ViewerModel

from lumic import SpectralImage
from plugin.plugin import SpectralUnmixingWidget


@pytest.mark.parametrize("sizes, scale, unit", [
    ({"Y": .3, "X": .2}, (.3, .2), "micrometer"),
    ({"Y": .3}, (1, 1), "pixel"),
    ({}, (1, 1), "pixel"),
])
def test_plane_layers_reuse_and_click_registration(sizes, scale, unit):
    viewer = ViewerModel()
    widget = SimpleNamespace(viewer=viewer, image=SpectralImage(
        np.arange(60).reshape(3, 4, 5), "CYX", pixel_sizes_um={"Y": .7, "X": .4}))
    widget.raw_img = widget.image.data
    widget._layer_is_in_viewer = lambda layer: layer is not None and layer in viewer.layers
    widget._image_layer = lambda *args, **kw: SpectralUnmixingWidget._image_layer(widget, *args, **kw)
    widget._points_layer = lambda *args, **kw: SpectralUnmixingWidget._points_layer(widget, *args, **kw)
    widget.raw_layer = widget._image_layer(None, widget.raw_img[0], "raw")
    widget.rgb_layer = widget._image_layer(None, np.zeros((4, 5, 3)), "rgb", rgb=True)
    widget.reference_points_layer = widget._points_layer(None, np.array([[2, 3]]), "references", "red", 5)
    widget.hull_layer = widget._points_layer(None, np.array([[2, 3]]), "hull", "white", 5)
    widget.abundance_layers = [widget._image_layer(None, np.zeros((4, 5)), "abundance")]
    widget.residual_layer = widget._image_layer(None, np.zeros((4, 5)), "residual")
    widget.initial_distance_layer = widget._image_layer(None, np.zeros((4, 5)), "distance")
    widget.optimized_distance_layer = widget._image_layer(None, np.zeros((4, 5)), "distance optimized")
    widget.pca_layer = widget._image_layer(None, np.zeros((4, 5)), "PC")

    widget.image = SpectralImage(widget.raw_img, "CYX", pixel_sizes_um=sizes)
    SpectralUnmixingWidget._update_layer_transforms(widget)
    layers = [widget.raw_layer, widget.rgb_layer, widget.reference_points_layer, widget.hull_layer,
              *widget.abundance_layers, widget.residual_layer, widget.initial_distance_layer,
              widget.optimized_distance_layer, widget.pca_layer]
    for layer in layers:
        np.testing.assert_allclose(layer.scale[-2:], scale)
        assert all(str(value) == unit for value in layer.units[-2:])
    world = widget.raw_layer.data_to_world((2, 3))
    assert SpectralUnmixingWidget._event_pixel(widget, SimpleNamespace(position=world)) == (2, 3)
    np.testing.assert_allclose(widget.reference_points_layer.data_to_world((2, 3)), world)
    labels = []
    widget.coord_label = SimpleNamespace(setText=labels.append)
    widget._event_pixel = lambda event: SpectralUnmixingWidget._event_pixel(widget, event)
    widget._active_layer_value = lambda event: SpectralUnmixingWidget._active_layer_value(widget, event)
    viewer.layers.selection.active = widget.raw_layer
    SpectralUnmixingWidget._show_pixel_coordinates(widget, viewer, SimpleNamespace(position=world))
    if unit == "micrometer":
        assert labels[-1] == "y=2, x=3  |  y=0.6 µm, x=0.6 µm  |  Intensity: 13"
    else:
        assert labels[-1] == "y=2, x=3  |  Intensity: 13"
    for layer, expected in ((widget.rgb_layer, "Intensity: (0, 0, 0)"),
                            (widget.initial_distance_layer, "D<sub>sc</sub>: 0"),
                            (widget.optimized_distance_layer, "D<sub>sc</sub>: 0"),
                            (widget.abundance_layers[0], "Rel. abundance: 0"),
                            (widget.residual_layer, "Intensity: 0"),
                            (widget.pca_layer, "Intensity: 0"),
                            (widget.hull_layer, None)):
        viewer.layers.selection.active = layer
        assert widget._active_layer_value(SimpleNamespace(position=world)) == expected
    assert widget.image.pixel_sizes_um == sizes
    reused = widget._image_layer(widget.abundance_layers[0], np.ones((4, 5)), "abundance")
    np.testing.assert_allclose(reused.scale, scale)
    points = widget._points_layer(widget.reference_points_layer, np.array([[1, 2]]), "references", "red", 5)
    np.testing.assert_allclose(points.scale, scale)


def test_pc_controls_remain_available_and_change_displayed_plane():
    viewer = ViewerModel()
    raw = viewer.add_image(np.zeros((2, 3)))
    pc = viewer.add_image(np.zeros((2, 3)))
    visibility, enabled, labels = [], [], []
    scores = np.arange(12).reshape(2, 3, 2)
    widget = SimpleNamespace(
        viewer=viewer, raw_layer=raw, raw_img=raw.data, residual_layer=None, residual_img=None,
        pca_layer=pc, pca_result=SimpleNamespace(scores=scores), is_busy=False,
        pc_controls=SimpleNamespace(setVisible=visibility.append),
        channel_slider=SimpleNamespace(setEnabled=enabled.append),
        pc_label=SimpleNamespace(setText=labels.append), _plot_pca=lambda: None,
        _layer_is_in_viewer=lambda layer: layer is not None and layer in viewer.layers,
    )
    viewer.layers.selection.active = pc
    SpectralUnmixingWidget._update_channel_slider_state(widget)
    assert visibility[-1]
    SpectralUnmixingWidget._set_residual_pc(widget, 2)
    np.testing.assert_array_equal(pc.data, scores[..., 1])
    assert labels[-1] == "PC 2" and pc.metadata["pc_number"] == 2
    viewer.layers.selection.active = raw
    SpectralUnmixingWidget._update_channel_slider_state(widget)
    assert visibility[-1] and enabled[-1]
    widget.pca_result = None
    viewer.layers.selection.active = pc
    SpectralUnmixingWidget._update_channel_slider_state(widget)
    assert not visibility[-1]
