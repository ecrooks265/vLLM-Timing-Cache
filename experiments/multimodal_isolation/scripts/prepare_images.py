#!/usr/bin/env python3
"""Download a small public natural-image subset; do not use private photos.

The dataset is leaf photography, NOT a broad natural-image benchmark. The
manifest preserves row IDs, revision, labels, source hashes and normalized hashes.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import random
import requests
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DATASET = 'AI-Lab-Makerere/beans'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default=str(ROOT / 'data'))
    args = ap.parse_args()
    out = Path(args.output)
    if (out / 'manifest.json').exists():
        manifest = json.loads((out / 'manifest.json').read_text())
        for image in manifest['images']:
            if hashlib.sha256((out / image['path']).read_bytes()).hexdigest() != image['sha256']:
                raise ValueError('Dataset hash mismatch: ' + image['id'])
        print('Existing manifest verified:', len(manifest['images']), 'images')
        return
    (out / 'images').mkdir(parents=True, exist_ok=True)
    card = requests.get('https://huggingface.co/api/datasets/' + DATASET, timeout=30)
    card.raise_for_status()
    revision = card.json()['sha']
    r = requests.get('https://datasets-server.huggingface.co/rows', params={
        'dataset': DATASET, 'config': 'default', 'split': 'validation',
        'offset': 0, 'length': 100}, timeout=45)
    r.raise_for_status()
    rows = r.json()['rows']
    # Round-robin labels so an ordered viewer slice does not select one disease.
    buckets = {}
    for row in rows:
        buckets.setdefault(row['row']['labels'], []).append(row)
    rng = random.Random(7952026)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected = []
    while len(selected) < 64:
        for label in sorted(buckets):
            if buckets[label] and len(selected) < 64:
                selected.append(buckets[label].pop())
        if not any(buckets.values()) and len(selected) < 64:
            raise RuntimeError('Insufficient rows in public dataset response')
    splits = ['warmup'] * 4 + ['calibration'] * 8 + ['evaluation'] * 40 + ['performance'] * 12
    manifest = {'dataset': DATASET, 'revision_at_download': revision,
                'dataset_url': 'https://huggingface.co/datasets/' + DATASET,
                'viewer_split': 'validation', 'license_in_metadata': card.json().get('cardData', {}).get('license'),
                'normalization': 'EXIF transpose, RGB, aspect-preserving pad to 336x336, PNG',
                'seed': 7952026, 'images': [],
                'limitation': 'One domain: bean-leaf photographs; not general image diversity.'}
    used = set()
    for row, split in zip(selected, splits):
        record = row['row']
        src = record['image']['src']
        raw = requests.get(src, timeout=30)
        raw.raise_for_status()
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw.content))).convert('RGB')
        image = ImageOps.pad(image, (336, 336), method=Image.Resampling.LANCZOS)
        ident = 'beans-val-' + str(row['row_idx'])
        path = 'images/' + ident + '.png'
        image.save(out / path, format='PNG')
        digest = hashlib.sha256((out / path).read_bytes()).hexdigest()
        if digest in used:
            raise ValueError('Duplicate normalized image; choose another subset')
        used.add(digest)
        manifest['images'].append({'id': ident, 'path': path, 'split': split,
            'row_index': row['row_idx'], 'class_label': record['labels'],
            'source_name': Path(record['image_file_path']).name,
            'source_sha256': hashlib.sha256(raw.content).hexdigest(), 'sha256': digest,
            'source_asset_revision': src.split('/--/')[1] if '/--/' in src else None})
        print(split, ident)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Prepared', len(manifest['images']), 'images')


if __name__ == '__main__':
    main()
