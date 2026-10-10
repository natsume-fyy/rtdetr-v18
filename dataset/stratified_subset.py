"""Create a reproducible, class-combination-stratified YOLO training subset.

Only the training split is sampled. Images and labels stay in their original
locations. Requires PyYAML; does not import torch or ultralytics.
"""

import argparse
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import yaml


IMAGE_SUFFIXES = {'.bmp', '.dng', '.jpeg', '.jpg', '.mpo', '.png', '.tif', '.tiff', '.webp', '.pfm'}


def absolute(path, base):
    # Do not resolve symlinks: HazyDet conversion may link images into images/train,
    # while labels live under labels/train, not beside the symlink targets.
    path = Path(path)
    return Path(os.path.abspath(path if path.is_absolute() else base / path))


def split_paths(value, root):
    values = value if isinstance(value, list) else [value]
    return [absolute(item, root) for item in values]


def collect_images(sources):
    images = set()
    for source in sources:
        if source.is_dir():
            candidates = source.rglob('*')
        elif source.is_file() and source.suffix.lower() == '.txt':
            # Match this repository's loader: ./ paths are relative to the list;
            # other relative entries are relative to the current working directory.
            candidates = [absolute(line, source.parent if line.startswith('./') else Path.cwd())
                          for raw in source.read_text(encoding='utf-8-sig').splitlines()
                          if (line := raw.strip())]
        else:
            raise FileNotFoundError(f'Expected image directory or .txt image list: {source}')
        for image in candidates:
            if image.suffix.lower() in IMAGE_SUFFIXES:
                if not image.is_file():
                    raise FileNotFoundError(f'Missing image: {image}')
                images.add(image)
    if not images:
        raise ValueError('No training images found.')
    return sorted(images, key=str)


def read_counts(image, nc, allow_missing):
    parts = list(image.parts)
    if 'images' not in parts:
        raise ValueError(f'Expected YOLO images/... and labels/... layout: {image}')
    index = len(parts) - 1 - parts[::-1].index('images')
    parts[index] = 'labels'
    label = Path(*parts).with_suffix('.txt')
    if not label.is_file():
        if allow_missing:
            return Counter(), True
        raise FileNotFoundError(f'Missing label: {label}; use --allow-missing-labels only for backgrounds.')
    counts = Counter()
    for line_number, line in enumerate(label.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        try:
            values = [float(value) for value in line.split()]
            if len(values) != 5 or not all(math.isfinite(value) for value in values):
                raise ValueError('expected five finite YOLO detection values')
            cls, x, y, w, h = values
            if not cls.is_integer() or not 0 <= cls < nc:
                raise ValueError('class ID out of range')
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise ValueError('invalid normalized bounding box')
        except ValueError as error:
            raise ValueError(f'{label}:{line_number}: {error}') from error
        counts[int(cls)] += 1
    return counts, False


def sample_strata(groups, fraction, seed):
    """Largest-remainder allocation gives an exact rounded global sample size."""
    rng = random.Random(seed)
    keys = sorted(groups)
    total = sum(len(groups[key]) for key in keys)
    target = max(1, int(total * fraction + 0.5))
    ideal = {key: len(groups[key]) * fraction for key in keys}
    quota = {key: math.floor(ideal[key]) for key in keys}
    # Randomize ties reproducibly rather than systematically favoring class IDs.
    order = keys.copy()
    rng.shuffle(order)
    order.sort(key=lambda key: ideal[key] - quota[key], reverse=True)
    for key in order[:target - sum(quota.values())]:
        quota[key] += 1
    selected = []
    for key in keys:
        selected.extend(rng.sample(groups[key], quota[key]))
    return sorted(selected), quota


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path(__file__).with_name('data.yaml'))
    parser.add_argument('--root', type=Path, help='Override dataset root (recommended for relative YAML paths).')
    parser.add_argument('--fraction', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('subset_20pct'))
    parser.add_argument('--allow-missing-labels', action='store_true')
    args = parser.parse_args()
    if not 0 < args.fraction <= 1:
        parser.error('--fraction must be in (0, 1].')
    config_path = args.data.resolve()
    config = yaml.safe_load(config_path.read_text(encoding='utf-8-sig'))
    names = config['names']
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or not names or set(names) != set(range(len(names))):
        raise ValueError('names must be a list or a mapping with contiguous integer class IDs starting at 0.')
    nc = len(names)
    if config.get('nc', nc) != nc:
        raise ValueError('nc does not match names.')
    # Relative dataset roots are explicitly interpreted relative to the input YAML.
    root = absolute(args.root, Path.cwd()) if args.root else absolute(config.get('path') or '.', config_path.parent)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new --output directory; already exists: {output}')
    images = collect_images(split_paths(config['train'], root))
    groups = defaultdict(list)
    counts_by_image = []
    missing = 0
    for index, image in enumerate(images):
        counts, absent = read_counts(image, nc, args.allow_missing_labels)
        missing += absent
        counts_by_image.append(counts)
        groups[tuple(sorted(counts))].append(index)
    selected, quota = sample_strata(groups, args.fraction, args.seed)

    def statistics(indices):
        image_counts, box_counts = Counter(), Counter()
        background = 0
        for index in indices:
            counts = counts_by_image[index]
            image_counts.update(counts.keys())
            box_counts.update(counts)
            background += not counts
        return {'images': len(indices), 'background_images': background,
                'classes': [{'id': cls, 'name': names[cls], 'images': image_counts[cls],
                             'boxes': box_counts[cls]} for cls in range(nc)]}

    report = {'source_yaml': str(config_path), 'root': str(root), 'seed': args.seed,
              'fraction': args.fraction, 'missing_labels_as_background': missing,
              'original': statistics(range(len(images))), 'subset': statistics(selected),
              'strata': [{'classes': list(key), 'original': len(groups[key]), 'subset': quota[key]}
                         for key in sorted(groups)]}
    result = dict(config)
    result['path'] = root.as_posix()
    result['train'] = (output / 'train.txt').as_posix()
    for split in ('val', 'test'):
        if config.get(split):
            paths = [path.as_posix() for path in split_paths(config[split], root)]
            result[split] = paths if isinstance(config[split], list) else paths[0]
    result.pop('download', None)
    output.mkdir(parents=True)
    (output / 'train.txt').write_text(''.join(images[index].as_posix() + '\n' for index in selected), encoding='utf-8')
    (output / 'data.yaml').write_text(yaml.safe_dump(result, allow_unicode=True, sort_keys=False), encoding='utf-8')
    (output / 'sampling_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Training images: {len(images)} -> {len(selected)} ({len(selected) / len(images):.2%}); seed={args.seed}')
    for original, subset in zip(report['original']['classes'], report['subset']['classes']):
        print(f"{original['name']}: images {original['images']} -> {subset['images']}; "
              f"boxes {original['boxes']} -> {subset['boxes']}")
        if original['images'] and not subset['images']:
            print(f"WARNING: class {original['name']} is absent from the subset; consider a larger fraction.")
    print(f"Train with: model.train(data={str(output / 'data.yaml')!r}, ...)")


if __name__ == '__main__':
    main()
