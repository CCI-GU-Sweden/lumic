"""Download curated microscopy datasets for Lumic examples."""

import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen


__all__ = ["download_cos7_data"]

_ACCESSION = "S-BIAD4448"
_ARCHIVE_URL = f"https://www.ebi.ac.uk/biostudies/files/{_ACCESSION}/lumic-data-submission"
_LOGGER = logging.getLogger(__name__)

# Relative paths and byte sizes from the public BioImage Archive file listing.
_FILES = {
    "high_SNR/mixed_avg_16.czi": 145122176,
    "high_SNR/DAPI_avg_16.czi": 145050432,
    "high_SNR/OregonGreen488_avg_16.czi": 145025152,
    "high_SNR/TMR_avg_16.czi": 145044480,
    "medium_SNR/mixed_avg_4.czi": 145142272,
    "medium_SNR/DAPI_avg_1.czi": 145077216,
    "medium_SNR/OregonGreen488_avg_1.czi": 145046592,
    "medium_SNR/TMR_avg_1.czi": 145054208,
    "low_SNR/mixed_avg_1.czi": 145165952,
    "low_SNR/DAPI_avg_1.czi": 145099072,
    "low_SNR/OregonGreen488_avg_1.czi": 145076448,
    "low_SNR/TMR_avg_1.czi": 145089024,
}


def download_cos7_data(*, snr="high", include_references=False, data_dir=None, progress=False):
    """Fetch selected public COS-7 CZI acquisitions from BioImage Archive.

    Parameters
    ----------
    snr : {"high", "medium", "low", "all"}, default "high"
        Acquisition condition to download. "all" selects high, medium, and
        low in that order.
    include_references : bool, default False
        Also download the three single-stain acquisitions for each selected
        condition. Each mixed acquisition is listed before its references.
    data_dir : str or pathlib.Path, optional
        Local directory for this accession, with one subfolder per SNR
        condition. Defaults to ~/.lumic/data/bioimage-archive/S-BIAD4448.
    progress : bool, default False
        Print percentage updates for each download and a message for each
        file reused from disk.

    Returns
    -------
    list of pathlib.Path
        Local CZI paths. Existing files matching the archive's byte sizes are
        reused without accessing the network. Incomplete files are replaced
        only after a successful download and size check.

    Notes
    -----
    The COS-7 data are available under CC0 from BioImage Archive accession
    S-BIAD4448. A single CZI is about 145 MB. Downloading all conditions
    with their single-stain references takes about 1.74 GB.
    """
    if snr not in ("high", "medium", "low", "all"):
        raise ValueError("snr must be 'high', 'medium', 'low', or 'all'")
    if data_dir is None:
        data_dir = Path.home() / ".lumic" / "data" / "bioimage-archive" / _ACCESSION
    data_dir = Path(data_dir).expanduser()

    paths = []
    for relative_path, expected_bytes in _FILES.items():
        if snr != "all" and not relative_path.startswith(f"{snr}_SNR/"):
            continue
        if not include_references and not Path(relative_path).name.startswith("mixed_"):
            continue
        destination = data_dir / relative_path
        if destination.is_file() and destination.stat().st_size == expected_bytes:
            _LOGGER.info("Using existing file: %s", destination)
            if progress:
                print(f"Using existing file: {destination}", flush=True)
        else:
            _download(relative_path, destination, expected_bytes, progress=progress)
        paths.append(destination)
    return paths


def _download(relative_path, destination, expected_bytes, *, progress):
    destination.parent.mkdir(parents=True, exist_ok=True)
    _LOGGER.info("Downloading %s (%.0f MB)", relative_path, expected_bytes / 1e6)
    if progress:
        print(f"Downloading {relative_path} (0%, {expected_bytes / 1e6:.0f} MB)", flush=True)

    # Publish the final filename only after the transfer and the size check.
    temp = NamedTemporaryFile(dir=destination.parent, suffix=".part", delete=False)
    temp_path = Path(temp.name)
    received = 0
    next_update = 10
    try:
        with temp, urlopen(f"{_ARCHIVE_URL}/{relative_path}", timeout=60) as response:
            while chunk := response.read(1024 * 1024):
                received += temp.write(chunk)
                if progress:
                    percent = 100 * received // expected_bytes
                    if next_update <= percent < 100:
                        print(f"{relative_path}: {percent}%", flush=True)
                        next_update = (percent // 10 + 1) * 10
        if received != expected_bytes:
            raise OSError(f"{relative_path}: expected {expected_bytes} bytes, got {received}")
        temp_path.replace(destination)
    finally:
        temp_path.unlink(missing_ok=True)
    _LOGGER.info("Saved %s", destination)
    if progress:
        print(f"{relative_path}: 100% — saved to {destination}", flush=True)
