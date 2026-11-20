from io import BytesIO

import pytest

from lumic import data


@pytest.mark.parametrize(
    ("options", "mixed_names", "file_count"),
    [
        ({}, ["mixed_avg_16.czi"], 1),
        ({"snr": "medium"}, ["mixed_avg_4.czi"], 1),
        ({"snr": "low", "include_references": True}, ["mixed_avg_1.czi"], 4),
        ({"snr": "all"}, ["mixed_avg_16.czi", "mixed_avg_4.czi", "mixed_avg_1.czi"], 3),
        (
            {"snr": "all", "include_references": True},
            ["mixed_avg_16.czi", "mixed_avg_4.czi", "mixed_avg_1.czi"],
            12,
        ),
    ],
)
def test_select_cos7_acquisitions(tmp_path, monkeypatch, options, mixed_names, file_count):
    def unexpected_request(*args, **kwargs):
        pytest.fail("Existing files should be reused without a network request")

    monkeypatch.setattr(data, "urlopen", unexpected_request)
    monkeypatch.setattr(data, "_FILES", dict.fromkeys(data._FILES, 3))
    for relative in data._FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"CZI")

    paths = data.download_cos7_data(data_dir=tmp_path, **options)

    assert len(paths) == file_count
    assert all(path.is_file() and path.parent.parent == tmp_path for path in paths)
    assert [path.name for path in paths if path.name.startswith("mixed_")] == mixed_names
    if options.get("include_references"):
        for index in range(0, len(paths), 4):
            assert [path.name.split("_")[0] for path in paths[index:index + 4]] == [
                "mixed", "DAPI", "OregonGreen488", "TMR",
            ]


def test_replace_incomplete_cos7_download(tmp_path, monkeypatch, capsys):
    relative = "high_SNR/mixed_avg_16.czi"
    target = tmp_path / relative
    target.parent.mkdir()
    target.write_bytes(b"old")
    monkeypatch.setattr(data, "_FILES", {relative: 8})
    monkeypatch.setattr(data, "urlopen", lambda *args, **kwargs: BytesIO(b"short"))

    with pytest.raises(OSError, match="expected 8 bytes, got 5"):
        data.download_cos7_data(data_dir=tmp_path, progress=True)
    assert target.read_bytes() == b"old"
    assert list(target.parent.iterdir()) == [target]
    assert "100%" not in capsys.readouterr().out

    monkeypatch.setattr(data, "urlopen", lambda *args, **kwargs: BytesIO(b"complete"))
    assert data.download_cos7_data(data_dir=tmp_path) == [target]
    assert target.read_bytes() == b"complete"


def test_report_download_progress(tmp_path, monkeypatch, capsys):
    class ChunkedResponse(BytesIO):
        def read(self, size):
            return super().read(min(size, 2))

    relative = "high_SNR/mixed_avg_16.czi"
    monkeypatch.setattr(data, "_FILES", {relative: 10})
    monkeypatch.setattr(data, "urlopen", lambda *args, **kwargs: ChunkedResponse(b"1234567890"))

    data.download_cos7_data(data_dir=tmp_path, progress=True)
    lines = capsys.readouterr().out.splitlines()
    assert "(0%" in lines[0]
    assert [line.rsplit(" ", 1)[-1] for line in lines[1:-1]] == ["20%", "40%", "60%", "80%"]
    assert "100%" in lines[-1]

    data.download_cos7_data(data_dir=tmp_path, progress=True)
    assert capsys.readouterr().out.startswith("Using existing file:")


def test_invalid_snr(tmp_path):
    with pytest.raises(ValueError, match="snr must be"):
        data.download_cos7_data(snr="unknown", data_dir=tmp_path)
