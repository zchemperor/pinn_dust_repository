"""Rebuild the record-level identity table for MAR_LGM.csv.

Design rules (deliberate, to avoid over-claiming identity):

* Keep every one of the 317 rows that survive the upstream filter
  ``(lat > -90) & (lat < 90) & (lon > -180) & (lon <= 180)``,
  with the original CSV row number and values that are NOT rounded.
* Never collapse rows by rounding. Records are keyed on the exact
  (lon, lat, MAR) triple, so Vostok 0.01295 and 0.013 stay two records.
* Enumerate *all* DIRTMAP v2 candidates near a record, not just the first hit.
  DIRTMAP v2 itself contains repeated coordinates, so "one coordinate" does not
  imply "one measurement".
* Separate coordinate evidence from value evidence from source-chain evidence.
  A coordinate candidate is never promoted to an identity on its own.

Outputs
-------
Data/metadata/p2_identity_reaudit_20261116/crosscheck/input_record_identity_v2.csv   one row per distinct record
Data/metadata/p2_identity_reaudit_20261116/crosscheck/input_record_candidates_v2.csv one row per (record, candidate)
Data/metadata/p2_identity_reaudit_20261116/crosscheck/identity_tier_summary.csv       tier counts

Run from the repository root.
"""

import csv
import math
import os
import pickle
from collections import Counter, defaultdict

BASE = "Data/metadata/p2_identity_reaudit_20261116/crosscheck"
RAW = "Data/original_data/MAR_LGM.csv"
V2_DIR = "tmp/dirtmap_search/dirtmap_v2"
S1_CSV = "Data/metadata/lambert2015_si/TableS1_Lambert2015_63records.csv"

TABLES = {
    "table1_ice_core.txt": "ice_core",
    "table2_marine_sediment.txt": "marine_sediment",
    "table3_marine_sediment_trap.txt": "marine_sediment_trap",
    "table4_non_carb_marine_sed.txt": "marine_noncarbonate",
    "table5_terrestrial_sediment.txt": "terrestrial",
    "table6_all_data.txt": "aggregate_table6",
    "table7_flagged_data.txt": "flagged_as_contaminated",
}

KG_TO_G_YR = 1000.0 * 365.0 * 24.0 * 3600.0  # 3.1536e10 g m-2 yr-1 per kg m-2 s-1
TOL_EXACT_KM = 1.0     # coordinate treated as the same location
TOL_NEAR_KM = 10.0     # looser screening radius, flagged separately
REL_VALUE = 0.005      # |candidate - input| / input <= 0.5 %

SAMPLE_TYPE = {
    "1": "ice core",
    "2": "marine sediment",
    "3": "marine sediment trap",
    "4": "terrestrial (loess)",
    "5": "marine sediment, noncarbonate",
}

# per-archive tables carry the archive identity in the table name, not a code column
TABLE_ARCHIVE = {
    "ice_core": "ice core",
    "marine_sediment": "marine sediment",
    "marine_sediment_trap": "marine sediment trap",
    "marine_noncarbonate": "marine sediment, noncarbonate",
    "terrestrial": "terrestrial (loess)",
    "aggregate_table6": "unknown (table6)",
    "flagged_as_contaminated": "unknown (table7)",
}


def load_table(path):
    for enc in ("cp1252", "utf-8-sig", "latin-1"):
        try:
            with open(path, encoding=enc) as fh:
                rows = list(csv.reader(fh, delimiter="\t"))
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError("cannot decode " + path)
    head = rows[0]
    out = []
    for row in rows[1:]:
        if len(row) >= len(head) - 1 and any(c.strip() for c in row):
            out.append(dict(zip(head, row)))
    return out


def great_circle_km(lon1, lat1, lon2, lat2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def load_input():
    with open(RAW, newline="") as fh:
        rows = list(csv.reader(fh))
    body = rows[1:]
    kept = []
    for idx, row in enumerate(body, start=1):
        lon, lat = float(row[0]), float(row[1])
        if (lat > -90) and (lat < 90) and (lon > -180) and (lon <= 180):
            kept.append(dict(
                csv_row=idx + 1,          # 1-based line number in MAR_LGM.csv
                rid="R%03d" % (len(kept) + 1),
                lon_raw=row[0], lat_raw=row[1], mar_raw=row[2], logmar_raw=row[3],
                lon=lon, lat=lat,
                mar_kg=float(row[2]),
                mar_g=float(row[2]) * KG_TO_G_YR,
            ))
    return kept


def load_v2():
    recs = []
    for fname, label in TABLES.items():
        for row in load_table(os.path.join(V2_DIR, fname)):
            try:
                lon, lat = float(row["Lon"]), float(row["Lat"])
            except (KeyError, ValueError):
                continue
            recs.append(dict(
                core=(row.get("Core ID") or "").strip().strip('"'),
                lon=lon, lat=lat,
                modern=(row.get("Modern g/m2/yr") or "").strip(),
                lgm=(row.get("LGM g/m2/yr") or "").strip(),
                ref1=(row.get("Reference 1") or "").strip().strip('"'),
                ref2=(row.get("Reference 2") or "").strip().strip('"'),
                age_begin=(row.get("LGM Begin Age") or "").strip(),
                age_end=(row.get("LGM End Age") or "").strip(),
                age_model=(row.get("Age Model Comment") or "").strip(),
                sample_type=(row.get("Sample Type Code") or "").strip(),
                data_type=(row.get("Data Type Code") or "").strip(),
                flag=(row.get("Flag") or "").strip(),
                comment=(row.get("Comments") or "").strip()[:90],
                table=label,
            ))
    # collapse identical candidate rows; prefer a per-archive table over table6
    uniq = {}
    for r in recs:
        key = (r["core"], round(r["lon"], 3), round(r["lat"], 3), r["lgm"])
        if key not in uniq or (uniq[key]["table"] == "aggregate_table6"
                               and r["table"] != "aggregate_table6"):
            uniq[key] = r
    return list(uniq.values())


def load_s1():
    """Lambert (2015) Table S1, keyed by rounded coordinate."""
    out = defaultdict(list)
    with open(S1_CSV, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            try:
                lon, lat = float(row["longitude_degE"]), float(row["latitude_degN"])
            except ValueError:
                continue
            out[(round(lon, 2), round(lat, 2))].append(row)
    return out


def classify(rec, cands, s1_hits):
    exact = [c for c in cands if c["d_km"] <= TOL_EXACT_KM]
    cores = {c["core"] for c in exact}
    value_ok = [c for c in exact if c["value_ok"]]

    s1_site = s1_hits[0].get("site_location", "") if s1_hits else ""
    s1_ref = s1_hits[0].get("reference", "") if s1_hits else ""
    s1_raw = s1_hits[0].get("LGM_flux_g_m-2_yr-1", "") if s1_hits else ""
    s1_pm10 = s1_hits[0].get("PM10_fraction", "") if s1_hits else ""
    s1_numeric = None
    if s1_raw:
        try:
            s1_numeric = float(s1_raw.replace("*", "").strip())
        except ValueError:
            s1_numeric = None

    factor = ""
    pm10 = None
    try:
        pm10 = float(s1_pm10)
    except (TypeError, ValueError):
        pm10 = None
    if s1_numeric:
        factor = round(rec["mar_g"] / s1_numeric, 4)
        if abs(factor - 1) <= REL_VALUE:
            tier, note = "S1_VALUE_VERIFIED", "source LGM reproduces the input value"
        elif pm10 is not None and abs(factor - pm10) <= 0.01:
            tier, note = "S1_RESCALED_PM10", "input = source LGM x tabulated PM10 fraction"
        else:
            tier, note = "S1_VALUE_DIFFERS", "source LGM present, input differs by factor %.3f" % factor
    elif s1_hits:
        tier, note = "S1_VALUE_UNVERIFIABLE", "site and reference known, source LGM is N/A"
    elif len(value_ok) == 1 and len(cores) == 1:
        tier, note = "V2_VALUE_UNIQUE", "single v2 candidate, value within 0.5 %"
    elif len(value_ok) >= 1:
        tier, note = "V2_VALUE_AMBIGUOUS", "several v2 candidates share the value"
    elif len(cores) == 1:
        tier, note = "V2_COORD_ONLY_UNIQUE", "single v2 candidate, value differs or missing"
    elif len(cores) > 1:
        tier, note = "V2_COORD_ONLY_AMBIGUOUS", "%d candidate core IDs at this location" % len(cores)
    elif cands:
        tier, note = "NEAR_CANDIDATE_ONLY", "only candidates between 1 and 10 km"
    else:
        tier, note = "NO_CANDIDATE", "no DIRTMAP v2 row within 10 km"

    return dict(
        identity_tier=tier, tier_note=note,
        n_candidates_exact=len(exact),
        n_distinct_cores_exact=len(cores),
        n_value_consistent=len(value_ok),
        nearest_km=round(cands[0]["d_km"], 2) if cands else "",
        candidate_cores=";".join(sorted(cores)) if cores else "",
        s1_site=s1_site, s1_reference=s1_ref, s1_lgm_raw=s1_raw,
        s1_pm10_fraction=s1_pm10,
        s1_ratio=("" if factor == "" else factor),
    )


def main():
    os.makedirs(BASE, exist_ok=True)
    inp = load_input()
    v2 = load_v2()
    s1 = load_s1()

    groups = defaultdict(list)
    for r in inp:
        groups[(r["lon"], r["lat"], r["mar_kg"])].append(r)

    rows, cand_rows = [], []
    for key in sorted(groups, key=lambda k: (k[1], k[0], k[2])):
        members = groups[key]
        lon, lat, mar_kg = key
        mar_g = members[0]["mar_g"]
        cands = []
        for c in v2:
            d = great_circle_km(lon, lat, c["lon"], c["lat"])
            if d <= TOL_NEAR_KM:
                try:
                    cv = float(c["lgm"])
                    ok = abs(cv - mar_g) / max(mar_g, 1e-12) <= REL_VALUE
                except ValueError:
                    cv, ok = None, False
                cands.append(dict(c, d_km=d, lgm_val=cv, value_ok=ok))
        cands.sort(key=lambda x: x["d_km"])

        s1_hits = s1.get((round(lon, 2), round(lat, 2)), [])
        rec = dict(mar_g=mar_g)
        cls = classify(rec, cands, s1_hits)

        rows.append(dict(
            record_key="%.4f|%.4f|%.6e" % (lon, lat, mar_kg),
            input_row_ids=";".join(m["rid"] for m in members),
            input_csv_lines=";".join(str(m["csv_row"]) for m in members),
            n_input_rows=len(members),
            lon_raw=members[0]["lon_raw"], lat_raw=members[0]["lat_raw"],
            mar_kg_m2_s=members[0]["mar_raw"], logmar_raw=members[0]["logmar_raw"],
            mar_g_m2_yr="%.10g" % mar_g,
            **cls))

        for c in cands:
            cand_rows.append(dict(
                record_key=rows[-1]["record_key"],
                input_row_ids=rows[-1]["input_row_ids"],
                distance_km=round(c["d_km"], 3),
                core_id=c["core"],
                v2_lon=c["lon"], v2_lat=c["lat"],
                v2_lgm_g_m2_yr=c["lgm"], v2_lgm_value=c["lgm_val"],
                input_mar_g_m2_yr="%.10g" % mar_g,
                value_ratio=("" if not c["lgm_val"] else round(mar_g / c["lgm_val"], 4)),
                value_consistent=c["value_ok"],
                v2_modern_g_m2_yr=c["modern"],
                reference_1=c["ref1"], reference_2=c["ref2"],
                lgm_begin_ka=c["age_begin"], lgm_end_ka=c["age_end"],
                age_model_comment=c["age_model"],
                sample_type_code=c["sample_type"],
                sample_type_meaning=SAMPLE_TYPE.get(c["sample_type"], "")
                or TABLE_ARCHIVE.get(c["table"], "unknown"),
                data_type_code=c["data_type"],
                contaminated_flag=c["flag"],
                v2_table=c["table"],
                comment=c["comment"]))

    # near-duplicate audit: records describing the same location but different values
    coords = [(float(r["lon_raw"]), float(r["lat_raw"]), float(r["mar_g_m2_yr"])) for r in rows]
    parent = list(range(len(rows)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if abs(coords[i][1] - coords[j][1]) > 0.2:
                continue
            if great_circle_km(coords[i][0], coords[i][1], coords[j][0], coords[j][1]) <= 3.0:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[find(i)].append(i)
    gid = 0
    for members in groups.values():
        if len(members) < 2:
            for i in members:
                rows[i]["near_duplicate_group"] = ""
                rows[i]["near_duplicate_values"] = ""
            continue
        gid += 1
        vals = sorted({round(coords[i][2], 6) for i in members})
        label = "ND%02d" % gid
        for i in members:
            rows[i]["near_duplicate_group"] = label
            rows[i]["near_duplicate_values"] = ";".join("%.4f" % v for v in vals)
    print("near-duplicate groups (<=3 km):", gid)

    identity = os.path.join(BASE, "input_record_identity_v2.csv")
    with open(identity, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    candidates = os.path.join(BASE, "input_record_candidates_v2.csv")
    with open(candidates, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cand_rows[0].keys()))
        w.writeheader()
        w.writerows(cand_rows)

    order = ["S1_VALUE_VERIFIED", "S1_RESCALED_PM10", "S1_VALUE_DIFFERS", "S1_VALUE_UNVERIFIABLE",
             "V2_VALUE_UNIQUE", "V2_VALUE_AMBIGUOUS",
             "V2_COORD_ONLY_UNIQUE", "V2_COORD_ONLY_AMBIGUOUS",
             "NEAR_CANDIDATE_ONLY", "NO_CANDIDATE"]
    cnt = Counter(r["identity_tier"] for r in rows)
    nrows = Counter()
    for r in rows:
        nrows[r["identity_tier"]] += r["n_input_rows"]
    summary = os.path.join(BASE, "identity_tier_summary.csv")
    with open(summary, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["identity_tier", "distinct_records", "input_rows", "meaning"])
        meaning = {
            "S1_VALUE_VERIFIED": "site name + reference in Lambert S1, LGM value identical",
            "S1_RESCALED_PM10": "site name + reference in Lambert S1, input = source x PM10 fraction",
            "S1_VALUE_DIFFERS": "site name in Lambert S1, input differs by a non-PM10 factor",
            "S1_VALUE_UNVERIFIABLE": "site name in Lambert S1 but source LGM is N/A",
            "V2_VALUE_UNIQUE": "single DIRTMAP v2 row at this coordinate with matching value",
            "V2_VALUE_AMBIGUOUS": "several v2 rows share the coordinate and the value",
            "V2_COORD_ONLY_UNIQUE": "single v2 row at this coordinate, value not reproduced",
            "V2_COORD_ONLY_AMBIGUOUS": "several v2 rows at this coordinate, value not reproduced",
            "NEAR_CANDIDATE_ONLY": "only candidates at 1-10 km, coordinate not identical",
            "NO_CANDIDATE": "no v2 row within 10 km",
        }
        for t in order:
            w.writerow([t, cnt.get(t, 0), nrows.get(t, 0), meaning[t]])

    print("input rows kept            :", len(inp))
    print("distinct records (unrounded):", len(rows))
    print("distinct coordinates       :", len({(r["lon_raw"], r["lat_raw"]) for r in rows}))
    print("candidate rows written     :", len(cand_rows))
    for t in order:
        print("  %-26s %3d records / %3d rows" % (t, cnt.get(t, 0), nrows.get(t, 0)))
    print("written:", identity)
    print("written:", candidates)
    print("written:", summary)


if __name__ == "__main__":
    main()
