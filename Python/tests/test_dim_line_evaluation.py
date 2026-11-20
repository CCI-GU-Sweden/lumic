"""Scene/audit contracts, not an assertion that blind recovery must succeed."""

import numpy as np

from evaluations.evaluate_dim_line_recovery import (
    BACKGROUND_COUNTS, COMPONENTS, DIM_COUNTS, IMAGE_SHAPE, PIXEL_SIZE_UM,
    build_scene, detector_spectra, audit_fit,
)
from lumic.simulation import GaussianPSF, SamplingGrid, SpectralDetector, render_scene


def test_dim_line_exists_separately_from_bright_spokes_over_uniform_background():
    scene=build_scene()
    assert scene.fluorophore_names==COMPONENTS
    assert sum("mNeptune" in shape.counts for shape in scene.shapes)==1
    source=render_scene(scene,SamplingGrid(IMAGE_SHAPE,PIXEL_SIZE_UM)).source_counts
    line=source[...,3]>0
    assert line.any()
    np.testing.assert_array_equal(source[...,4],np.full(IMAGE_SHAPE,BACKGROUND_COUNTS))
    assert source[...,3].max()==DIM_COUNTS
    assert not source[line,:3].any()
    blurred=render_scene(scene,SamplingGrid(IMAGE_SHAPE,PIXEL_SIZE_UM),
                         GaussianPSF({name:.6 for name in COMPONENTS})).blurred_counts
    assert 0<blurred[...,3].max()<source[...,3].max()


def test_offline_detector_signatures_use_unit_sum_detected_counts():
    spectra=detector_spectra(SpectralDetector.from_profile("980_32ch"))
    assert spectra.shape==(5,32)
    assert np.isfinite(spectra).all() and np.all(spectra>=0)
    np.testing.assert_allclose(spectra.sum(axis=1),1)


def test_rectangular_truth_audit_reports_omitted_component(tmp_path):
    counts=np.arange(1,31,dtype=float).reshape(2,3,5)
    truth_spectra=np.eye(5)
    keep=[0,1,2,4]
    np.savez(tmp_path/"fit.npz",abundances=counts[...,keep])
    fitted=dict(folder=tmp_path,n_components=4,spectra=truth_spectra[keep])
    rows,recovered,_=audit_fit(counts,truth_spectra,fitted)
    assert not rows[3]["assigned"] and rows[3]["shape_distance"] is None
    assert rows[3]["count_rmse"]==np.sqrt(np.mean(counts[...,3]**2))
    assert not recovered[...,3].any()
    np.testing.assert_allclose(recovered[...,keep],counts[...,keep])
