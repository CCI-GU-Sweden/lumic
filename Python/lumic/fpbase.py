"""Small adapter for retrieving fluorophore emission spectra from FPbase."""

import numpy as np


_SPECTRUM_QUERY = """
query GetSpectrum($id: Int!) {
    spectrum(id: $id) {
        id
        subtype
        data
    }
}
"""


def fetch_emission_spectrum(protein_name):
    """Fetch a protein's default emission spectrum from FPbase.

    The returned arrays contain the wavelength and intensity values as stored
    by FPbase. Intensities are not normalized or resampled.

    Parameters
    ----------
    protein_name : str
        A protein name recognized by FPbase, such as ``"EGFP"`` or
        ``"mCherry"``.

    Returns
    -------
    wavelengths, intensities : tuple of numpy.ndarray
        One-dimensional arrays containing wavelength and intensity values.

    Raises
    ------
    ValueError
        If FPbase has no single default emission spectrum or returns invalid
        spectrum data.
    """
    try:
        import fpbase as fpbase_client
    except ModuleNotFoundError as error:
        if error.name != "fpbase":
            raise
        raise ImportError(
            "Online spectrum retrieval requires the FPbase client. "
            "Install it with pip install 'lumic[fpbase]'."
        ) from error

    protein = fpbase_client.get_protein(protein_name)
    state = getattr(protein, "default_state", None)
    spectra = getattr(state, "spectra", None) or []
    emission_spectra = [
        spectrum for spectrum in spectra if getattr(spectrum, "subtype", None) == "EM"
    ]
    if len(emission_spectra) != 1:
        raise ValueError(
            f"Expected one default emission spectrum for {protein_name!r}; "
            f"found {len(emission_spectra)}"
        )

    spectrum_id = int(emission_spectra[0].id)
    response = fpbase_client.graphql_query(
        _SPECTRUM_QUERY, variables={"id": spectrum_id}
    )
    if response.get("errors"):
        raise ValueError(f"FPbase could not return spectrum {spectrum_id}")

    try:
        curve = np.asarray(response["data"]["spectrum"]["data"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"FPbase returned invalid spectrum data for {protein_name!r}") from exc

    if (
        curve.ndim != 2
        or curve.shape[1] != 2
        or curve.shape[0] < 2
        or not np.isfinite(curve).all()
        or np.any(np.diff(curve[:, 0]) <= 0)
        or np.any(curve[:, 1] < 0)
    ):
        raise ValueError(f"FPbase returned invalid spectrum data for {protein_name!r}")

    return curve[:, 0], curve[:, 1]
