import io
import zipfile

import pandas as pd
import pytest
import shapefile

from loanfair.references import cached_download, lookup_apor, parse_apor, tract_table


def test_apor_pipe_format_and_effective_week():
    row = lambda day, rate: day + "|" + "|".join([str(rate)] * 50)
    table = parse_apor("header\n" + row("01/01/2024", 5.0) + "\n" + row("01/08/2024", 5.5), "fixed")
    assert lookup_apor(table, "2024-01-07") == (5.0, "2024-01-01")
    assert lookup_apor(table, "2024-01-08") == (5.5, "2024-01-08")
    with pytest.raises(ValueError, match="effective week"):
        lookup_apor(table, "2024-01-15")
    with pytest.raises(ValueError, match="No historical"):
        lookup_apor(table, "2023-12-31")
    with pytest.raises(ValueError, match="duplicate"):
        parse_apor(row("01/01/2024", 5.0) + "\n" + row("01/01/2024", 5.0), "fixed")


def test_offline_cache_reuse_and_missing_error(tmp_path, monkeypatch):
    path = tmp_path / "cached.txt"
    path.write_text("already downloaded")
    def fail(*args, **kwargs):
        raise AssertionError("Network should not be used")
    monkeypatch.setattr("requests.Session.get", fail)
    assert cached_download("https://example.invalid", path, offline=True) == path
    with pytest.raises(FileNotFoundError, match="Missing cached"):
        cached_download("https://example.invalid", tmp_path / "missing", offline=True)


def test_census_dbf_coordinates(tmp_path):
    stream = io.BytesIO()
    with shapefile.Writer(dbf=stream) as writer:
        writer.field("GEOID", "C", size=11)
        writer.field("INTPTLAT", "C", size=12)
        writer.field("INTPTLON", "C", size=12)
        writer.record("51001000100", "+37.100000", "-77.100000")
    path = tmp_path / "tract.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("tract.dbf", stream.getvalue())
    result = tract_table(path, 2017)
    assert result.latitude.iloc[0] == 37.1
    assert result.tract_vintage.iloc[0] == 2017
