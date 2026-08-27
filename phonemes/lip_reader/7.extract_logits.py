"""Extract CTC logits for each segment in the dataset JSONs.

For each .mouth.npy (frontend+proj cache), find the matching dataset JSON,
run encoder+CTC on [start, end+5] context windows, and save per-segment
logits as .phonemes.npz next to the .mouth.npy.

The .phonemes.npz contains:
  - "logits_0", "logits_1", ... : float16 [T_padded, 39] — CTC logits per segment
  - "ends": int32 [N] — actual end frame (without +5s) relative to each logits array

Usage:
    python extract_logits.py [--ckpt PATH] [--pad 5.0]
"""

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_AVSR_ROOT = str(Path(__file__).resolve().parent.parent / 'auto_avsr')
if _AVSR_ROOT not in sys.path:
    sys.path.insert(0, _AVSR_ROOT)

from auto_avsr.espnet.nets.pytorch_backend.e2e_asr_conformer import E2E  # noqa: E402
from auto_avsr.espnet.nets.pytorch_backend.nets_utils import (
    make_non_pad_mask,  # noqa: E402
)

NUM_CLASSES = 39
FPS = 25
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT = '/home/radumicea/Desktop/PV-ASR/from_scratch_best.ckpt'
PAD_SEC = 5.0

SCRAPED_TO_DATASET = {
    'digi/scraped': 'new_feb_stuff/dataset/Digi24',
    'prima/scraped': 'new_feb_stuff/dataset/PrimaTV',
    'protv/scraped': 'new_feb_stuff/dataset/ProTV',
}

# Reverse: dataset prefix -> scraped prefix
DATASET_TO_SCRAPED = {v: k for k, v in SCRAPED_TO_DATASET.items()}


def load_model(ckpt_path):
    model = E2E(NUM_CLASSES, modality='video', ctc_weight=1.0)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt['state_dict']
    state = {
        k.removeprefix('model.'): v for k, v in state.items() if k.startswith('model.')
    }
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()
    return model


@torch.no_grad()
def get_logits(model, features):
    """features: [T, 768] -> logits: [T, 39]"""
    x = features.unsqueeze(0).to(DEVICE)
    lengths = torch.tensor([x.shape[1]], device=DEVICE)
    padding_mask = make_non_pad_mask(lengths).unsqueeze(-2).to(DEVICE)

    with torch.amp.autocast(DEVICE.type, enabled=DEVICE.type == 'cuda'):
        x = model.encoder.embed(x)
        for layer in model.encoder.encoders:
            x, padding_mask = layer(x, padding_mask)
        if isinstance(x, tuple):
            x = x[0]
        if model.encoder.normalize_before:
            x = model.encoder.after_norm(x)
        logits = model.ctc.ctc_lo(x)  # [1, T, 39]
    return logits[0].float().cpu()


def find_pairs():
    """Find all .mouth.npy files with matching dataset JSONs."""
    pairs = []
    for scrape_prefix, dataset_prefix in SCRAPED_TO_DATASET.items():
        for mouth_path in glob.glob(
            os.path.join(scrape_prefix, '**', '*.mouth.npy'), recursive=True
        ):
            p = Path(mouth_path)
            stem = p.name.replace('.mouth.npy', '')
            rel = str(p.parent.relative_to(scrape_prefix))
            json_path = os.path.join(dataset_prefix, rel, stem + '.json')
            if os.path.isfile(json_path):
                pairs.append((mouth_path, json_path))
    return pairs


def process_one(mouth_path, json_path, model, pad_sec=PAD_SEC):
    """Process one video: load features, run encoder+CTC per segment, save .phonemes.npz."""
    out_path = mouth_path.replace('.mouth.npy', '.phonemes.npz')
    if Path(out_path).exists():
        return True  # skip

    with open(json_path, encoding='utf-8') as f:
        segments = json.load(f)
    if not segments:
        return False

    mm = np.load(mouth_path, mmap_mode='r')
    total_frames = len(mm)

    save_dict = {}
    ends = []

    for i, seg in enumerate(segments):
        start_sec = seg['start']
        end_sec = seg['end']
        end_padded_sec = end_sec + pad_sec

        sf = max(0, min(round(start_sec * FPS), total_frames))
        ef_actual = max(sf, min(round(end_sec * FPS), total_frames))
        ef_padded = max(ef_actual, min(round(end_padded_sec * FPS), total_frames))

        if ef_padded <= sf:
            save_dict[f'logits_{i}'] = np.zeros((1, NUM_CLASSES), dtype=np.float16)
            ends.append(0)
            continue

        features = torch.from_numpy(np.array(mm[sf:ef_padded], dtype=np.float32))
        logits = get_logits(model, features)  # [T_padded, 39]

        save_dict[f'logits_{i}'] = logits.numpy().astype(np.float16)
        ends.append(ef_actual - sf)

    save_dict['ends'] = np.array(ends, dtype=np.int32)
    np.savez(out_path, **save_dict)
    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Extract CTC logits per segment')
    parser.add_argument('--ckpt', default=CKPT)
    parser.add_argument('--pad', type=float, default=PAD_SEC)
    args = parser.parse_args()

    os.chdir(Path(__file__).resolve().parent.parent)

    pairs = find_pairs()
    print(f'{len(pairs)} mouth+json pairs')

    model = load_model(args.ckpt)

    for mouth_path, json_path in tqdm(pairs):
        try:
            process_one(mouth_path, json_path, model, pad_sec=args.pad)
        except Exception as ex:
            tqdm.write(f'SKIP {mouth_path}: {ex}')

    torch.cuda.empty_cache()
    print('Done.')


if __name__ == '__main__':
    main()
