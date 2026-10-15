"""Download official references; cache raw files and validated normalized tables."""
from __future__ import annotations

import io
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import shapefile
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .common import fingerprint, log, write_json

TRACT_URLS = {
    2017: "https://www2.census.gov/geo/tiger/TIGER2017/TRACT/tl_2017_51_tract.zip",
    2020: "https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_51_tract.zip",
}
APOR_URLS = {
    kind: f"https://files.ffiec.cfpb.gov/apor/YieldTable{kind.title()}.txt"
    for kind in ("fixed", "adjustable")
}


def cached_download(url, path, offline=False, refresh=False):
    path = Path(path)
    if path.exists() and not refresh:
        return path
    if offline:
        raise FileNotFoundError(f"Missing cached reference: {path}. Run fetch_references.py online first.")
    path.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))
    log(f"Downloading {url}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with session.get(url, timeout=(15, 120), stream=True) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for block in response.iter_content(1024 * 1024):
                    output.write(block)
        temporary.replace(path)
    except requests.RequestException as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Reference download failed: {url}. Retry online or supply a previously downloaded cache.") from exc
    finally:
        session.close()
    return path


def tract_table(path, vintage):
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.endswith(".dbf")]
        if len(names) != 1:
            raise ValueError("Expected exactly one Census tract DBF in archive.")
        with shapefile.Reader(dbf=io.BytesIO(archive.read(names[0]))) as reader:
            records = [r.as_dict() for r in reader.iterRecords()]
    data = pd.DataFrame(records)
    result = data[["GEOID", "INTPTLAT", "INTPTLON"]].rename(
        columns={"GEOID": "census_tract", "INTPTLAT": "latitude", "INTPTLON": "longitude"}
    )
    result["tract_vintage"] = vintage
    result["census_tract"] = result.census_tract.astype(str)
    for col in ("latitude", "longitude"):
        result[col] = pd.to_numeric(result[col], errors="raise")
    if result.census_tract.duplicated().any() or not result.census_tract.str.match(r"^51\d{9}$").all():
        raise ValueError("Invalid or duplicate Virginia Census tract identifiers.")
    if not (result.latitude.between(36, 40.1) & result.longitude.between(-84, -75)).all():
        raise ValueError("Census coordinates are outside Virginia's bounding box.")
    return result


def parse_apor(text, amortization):
    """Official files contain one date followed by rates for terms 1 through 50."""
    rows = []
    for line in text.splitlines():
        tokens = re.split(r"[\s,|]+", line.strip())
        if not tokens or not re.fullmatch(r"\d{2}/\d{2}/\d{4}", tokens[0]):
            continue
        if len(tokens) != 51:
            raise ValueError(f"Unexpected APOR column count on {tokens[0]}: {len(tokens)}")
        date = pd.to_datetime(tokens[0], format="%m/%d/%Y")
        for term, rate in enumerate(tokens[1:], start=1):
            rows.append((date, amortization, term, float(rate)))
    result = pd.DataFrame(rows, columns=["effective_date", "amortization", "term_years", "apor"])
    if result.empty or result.duplicated(["effective_date", "amortization", "term_years"]).any():
        raise ValueError("APOR table is empty or has ambiguous duplicate effective dates.")
    if not result.apor.between(0, 30).all():
        raise ValueError("Invalid APOR values.")
    return result.sort_values(["effective_date", "term_years"]).reset_index(drop=True)


def fetch_references(config, offline=False, refresh=False):
    directory = Path(config["reference_dir"])
    manifest_path = directory / "manifest.json"
    old_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest = {}
    tracts, rates = [], []
    for vintage, url in TRACT_URLS.items():
        path = cached_download(url, directory / f"tracts_{vintage}.zip", offline, refresh)
        tracts.append(tract_table(path, vintage))
        manifest[path.name] = source_record(path, url, old_manifest)
    for kind, url in APOR_URLS.items():
        path = cached_download(url, directory / f"apor_{kind}.txt", offline, refresh)
        rates.append(parse_apor(path.read_text(encoding="utf-8-sig"), kind))
        manifest[path.name] = source_record(path, url, old_manifest)
    pd.concat(tracts, ignore_index=True).to_parquet(directory / "tracts.parquet", index=False)
    apor = pd.concat(rates, ignore_index=True)
    apor.to_parquet(directory / "apor.parquet", index=False)
    write_json(manifest_path, manifest)
    log(f"References ready: {sum(len(t) for t in tracts):,} tract/vintage pairs; {len(apor):,} APOR entries.")


def source_record(path, url, old):
    digest = fingerprint(path)
    previous = old.get(path.name, {})
    return {
        "url": url, "sha256": digest,
        "retrieved_at_utc": previous.get("retrieved_at_utc") if previous.get("sha256") == digest else datetime.now(timezone.utc).isoformat(),
    }


def load_tracts(config):
    path = Path(config["reference_dir"]) / "tracts.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run fetch_references.py first.")
    table = pd.read_parquet(path)
    if table.duplicated(["census_tract", "tract_vintage"]).any():
        raise ValueError("Duplicate tract/vintage keys in reference cache.")
    return table


def lookup_apor(table, rate_set_date):
    date = pd.Timestamp(rate_set_date)
    candidates = table[(table.amortization == "fixed") & (table.term_years == 30) & (table.effective_date <= date)]
    if candidates.empty:
        raise ValueError("No historical fixed 30-year APOR for that rate-set date.")
    row = candidates.sort_values("effective_date").iloc[-1]
    if (date - row.effective_date).days > 6:
        raise ValueError("APOR cache does not cover that effective week; refresh reference data.")
    return float(row.apor), row.effective_date.strftime("%Y-%m-%d")
