from types import SimpleNamespace
import numpy as np
import pytest
from lumic import SpectralImage
from plugin.plugin import SpectralUnmixingWidget, canonicalize_detector_plane

@pytest.mark.parametrize('axes', ['CYX', 'YXC', 'XCY'])
def test_canonical_plane_preserves_pixels_and_metadata(axes):
    original = np.arange(24).reshape(3, 2, 4)
    source = original.transpose(['CYX'.index(a) for a in axes]).copy()
    before = source.copy()
    image = SpectralImage(source, axes, kind='reconstructed', wavelengths_nm=[450, 500, 550],
                          detection_bands_nm=[[440, 460], [490, 510], [540, 560]],
                          pixel_sizes_um={'X': 0.2, 'Y': 0.3}, provenance={'source': 'test'})
    result = canonicalize_detector_plane(image)
    assert result.axes == 'CYX' and result.kind == image.kind
    np.testing.assert_array_equal(result.data, original)
    np.testing.assert_array_equal(result.unfold().data, original.transpose(1, 2, 0).reshape(-1, 3))
    np.testing.assert_array_equal(source, before)
    np.testing.assert_array_equal(result.wavelengths_nm, image.wavelengths_nm)
    np.testing.assert_array_equal(result.detection_bands_nm, image.detection_bands_nm)
    assert result.pixel_sizes_um == image.pixel_sizes_um and result.provenance == image.provenance

def test_rejects_abundances_and_unselected_stacks():
    with pytest.raises(ValueError, match='kind.*abundance'):
        canonicalize_detector_plane(SpectralImage(np.ones((2, 3, 4)), 'KYX', kind='abundance'))
    with pytest.raises(ValueError, match='selected detector plane'):
        canonicalize_detector_plane(SpectralImage(np.ones((2, 3, 4, 5)), 'TCYX'))

def test_plane_selection_requests_indices_before_canonicalization(monkeypatch):
    calls = []
    class Reader:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, *, t, z):
            calls.append((t, z))
            return SpectralImage(np.arange(24).reshape(2, 4, 3), 'YXC')
    monkeypatch.setattr('plugin.plugin.open_image', lambda path: Reader())
    slider = lambda value: SimpleNamespace(value=lambda: value)
    received = []
    widget = SimpleNamespace(image_path='image', is_busy=False,
        plane_controls={'T': (None, slider(2), None), 'Z': (None, slider(1), None)},
        _apply_plane=lambda *args: received.append(args))
    SpectralUnmixingWidget._select_plane(widget)
    assert calls == [(2, 1)] and widget.plane_position == (2, 1)
    assert received[0][0].axes == 'CYX' and received[0][2:] == (2, 1)

@pytest.mark.parametrize('channels', [1, 2, 3, 4, 18, 32])
def test_selected_plane_uses_shared_display_rendering(monkeypatch, channels):
    pixels = np.arange(channels * 6, dtype=float).reshape(channels, 2, 3)
    image = SpectralImage(pixels, 'CYX')
    class Reader:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, *, t, z): return image
    monkeypatch.setattr('plugin.plugin.open_image', lambda path: Reader())
    slider = SimpleNamespace(value=lambda: 0)
    received = []
    widget = SimpleNamespace(image_path='image', is_busy=False,
        plane_controls={'T': (None, slider, None), 'Z': (None, slider, None)},
        _apply_plane=lambda *args: received.append(args))
    SpectralUnmixingWidget._select_plane(widget)
    selected, rendering, _, _ = received[0]
    np.testing.assert_array_equal(selected.unfold().data, pixels.transpose(1, 2, 0).reshape(-1, channels))
    if channels < 3:
        assert rendering is None
    else:
        expected = selected.to_rgb(limits=rendering.limits)
        np.testing.assert_array_equal(rendering.data, expected.data)
        assert rendering.channel_groups == expected.channel_groups
    np.testing.assert_array_equal(image.data, pixels)
