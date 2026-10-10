"""Convert HazyDet COCO annotations without modifying the source dataset."""

import argparse
import json
import math
import shutil
from pathlib import Path


CLASSES = {0: 'car', 1: 'truck', 2: 'bus'}


def prepare_split(root, split, image_dir):
    with (root / split / f'{split}_coco.json').open(encoding='utf-8') as handle:
        data = json.load(handle)
    categories = {item['id']: item['name'] for item in data['categories']}
    if categories != CLASSES:
        raise ValueError(f'{split}: unexpected categories: {categories}')
    records, destinations = {}, set()
    for item in data['images']:
        relative = Path(item['file_name'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'Unsafe image path: {relative}')
        source = root / split / image_dir / relative
        if not source.is_file():
            raise FileNotFoundError(f'Image missing: {source}')
        label = relative.with_suffix('.txt')
        if item['id'] in records or label in destinations:
            raise ValueError(f'{split}: duplicate image ID or label path: {relative}')
        width, height = float(item['width']), float(item['height'])
        if not all(math.isfinite(v) and v > 0 for v in (width, height)):
            raise ValueError(f'{split}: invalid image dimensions: {relative}')
        records[item['id']] = (source, relative, width, height, [])
        destinations.add(label)

    clipped = skipped = crowds = 0
    for ann in data['annotations']:
        if ann['image_id'] not in records or ann['category_id'] not in CLASSES:
            raise ValueError(f'{split}: unknown image/category in annotation {ann.get("id")}')
        if ann.get('iscrowd', 0):
            crowds += 1
            continue
        _, _, width, height, lines = records[ann['image_id']]
        x, y, w, h = map(float, ann['bbox'])
        if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
            raise ValueError(f'{split}: invalid bbox in annotation {ann.get("id")}')
        x1, y1 = max(0.0, x), max(0.0, y)
        x2, y2 = min(width, x + w), min(height, y + h)
        if x2 <= x1 or y2 <= y1:
            skipped += 1
            continue
        clipped += (x1, y1, x2, y2) != (x, y, x + w, y + h)
        box = ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
               (x2 - x1) / width, (y2 - y1) / height)
        lines.append(str(ann['category_id']) + ' ' + ' '.join(f'{v:.10f}' for v in box))
    return records, clipped, skipped, crowds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/root/autodl-tmp/HazyDet'))
    parser.add_argument('--output', type=Path, default=Path('/root/autodl-tmp/HazyDet_YOLO'))
    parser.add_argument('--image-dir', default='hazy_images', choices=['hazy_images', 'images'])
    parser.add_argument('--mode', choices=['symlink', 'copy'], default='symlink',
                        help='symlink saves disk space; copy makes an independent dataset')
    parser.add_argument('--check-only', action='store_true', help='validate without writing files')
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if not args.check_only and output.exists():
        raise FileExistsError(f'Output already exists; choose a new --output directory: {output}')

    # Validate all splits before creating any output.
    splits = {}
    for split in ('train', 'val', 'test'):
        records, clipped, skipped, crowds = prepare_split(root, split, args.image_dir)
        splits[split] = records
        boxes = sum(len(record[4]) for record in records.values())
        empty = sum(not record[4] for record in records.values())
        print(f'{split}: images={len(records)}, boxes={boxes}, empty={empty}, '
              f'clipped={clipped}, outside_skipped={skipped}, crowd_skipped={crowds}')
    if args.check_only:
        print('Validation passed (paths and annotations; image pixels were not decoded).')
        return

    output.mkdir(parents=True)
    for split, records in splits.items():
        (output / 'images' / split).mkdir(parents=True)
        (output / 'labels' / split).mkdir(parents=True)
        for source, relative, _, _, lines in records.values():
            image = output / 'images' / split / relative
            label = output / 'labels' / split / relative.with_suffix('.txt')
            image.parent.mkdir(parents=True, exist_ok=True)
            label.parent.mkdir(parents=True, exist_ok=True)
            if args.mode == 'symlink':
                image.symlink_to(source)
            else:
                shutil.copy2(source, image)
            label.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
    yaml = ('path: ' + json.dumps(output.as_posix(), ensure_ascii=False) + '\n'
            'train: images/train\nval: images/val\ntest: images/test\n'
            "nc: 3\nnames: ['car', 'truck', 'bus']\n")
    config = output / 'data.yaml'
    config.write_text(yaml, encoding='utf-8')
    print(f'Done. Set model.train(data={str(config)!r}, ...)')


if __name__ == '__main__':
    main()
