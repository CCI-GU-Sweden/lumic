from types import SimpleNamespace

import numpy as np
import pytest

import fpbase as fpbase_client
from lumic.fpbase import fetch_emission_spectrum


@pytest.mark.parametrize(
    ("protein_name", "emission_spectrum_id"),
    [("EGFP", 18), ("mCherry", 80)],
)
def test_fetch_emission_spectrum_returns_fpbase_curve(
    monkeypatch, protein_name, emission_spectrum_id
):
    curve = [[460.0, 0.1], [461.0, 0.4], [462.0, 0.2]]
    protein = SimpleNamespace(
        default_state=SimpleNamespace(
            spectra=[
                SimpleNamespace(id=emission_spectrum_id - 1, subtype="EX"),
                SimpleNamespace(id=emission_spectrum_id, subtype="EM"),
            ]
        )
    )
    query_calls = []

    monkeypatch.setattr(fpbase_client, "get_protein", lambda name: protein)

    def fake_graphql_query(query, variables):
        query_calls.append((query, variables))
        return {"data": {"spectrum": {"data": curve}}}

    monkeypatch.setattr(fpbase_client, "graphql_query", fake_graphql_query)

    wavelengths, intensities = fetch_emission_spectrum(protein_name)

    np.testing.assert_array_equal(wavelengths, [460.0, 461.0, 462.0])
    np.testing.assert_array_equal(intensities, [0.1, 0.4, 0.2])
    assert len(query_calls) == 1
    assert query_calls[0][1] == {"id": emission_spectrum_id}


def test_fetch_emission_spectrum_requires_one_default_emission_curve(monkeypatch):
    protein = SimpleNamespace(
        default_state=SimpleNamespace(
            spectra=[SimpleNamespace(id=17, subtype="EX")]
        )
    )
    monkeypatch.setattr(fpbase_client, "get_protein", lambda name: protein)

    with pytest.raises(ValueError, match="one default emission spectrum"):
        fetch_emission_spectrum("EGFP")
