"""Export the submission tables from existing audits; never fit or alter inputs.

Run with python -B Code/export_supplementary_tables.py
The delivered manifest supplies source paths and expected output hashes.
This exports existing results; it does not regenerate the underlying audits.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'manuscript/supplementary_tables')
    args = parser.parse_args()
    catalog = ROOT/'manuscript/supplementary_tables'
    manifest = json.loads((catalog/'manifest.json').read_text(encoding='utf-8'))
    args.output.mkdir(parents=True, exist_ok=True)
    for item in manifest:
        source = ROOT/item['source']
        assert hashlib.sha256(source.read_bytes()).hexdigest() == item['source_sha256'], source
        destination = args.output/item['file']
        if item['file'] == 'Table_S3_value_ratios.csv':
            with source.open(encoding='utf-8-sig', newline='') as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                try:
                    ratio = float(row['input_g_m2_yr'])/float(row['candidate_LGM_g_m2_yr'])
                except ValueError:
                    ratio = None
                row['input_over_source'] = '' if ratio is None else repr(ratio)
            with destination.open('w', encoding='utf-8-sig', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        else:
            shutil.copy2(source, destination)
        assert hashlib.sha256(destination.read_bytes()).hexdigest() == item['sha256'], destination
    for name in ['README.md', 'manifest.json']:
        if (catalog/name).resolve() != (args.output/name).resolve():
            shutil.copy2(catalog/name, args.output/name)
    print(f'Verified export of all {len(manifest)} supplementary tables; no input files changed.')


if __name__ == '__main__':
    main()
