#!/usr/bin/env python3
"""BSL-5k I3D features of the segments in the chosen splits:
mp4s/<channel>/.../segment_<i>.mp4 -> dataset/<channel>/.../segment_<i>.bsl5k.npy,
float16, shape (N, 1024); feature k covers frames [2k, 2k + 8) of the segment.

The model is the BSL-1K I3D with the 5K-vocabulary weights; the feature is its
average pool over Mixed_5c, computed with fp16 autocast. Frames are preprocessed as in the
reference test-time code (bsl1k/datasets/videodataset.py), except for the resize:
RGB / 255, shorter side resized to 224 (bilinear), centre crop 224, (x - 0.5) / 1.0.
BOBSL uses 16-frame windows at stride 4 and resizes the shorter side to 256; the
settings here translate better (FEATURES.md).

Needs ffprobe, a CUDA GPU and the packages of requirements.txt.
Stop it and run it again to resume: finished segments are skipped."""

import argparse
import hashlib
import importlib.util
import json
import queue
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
MP4S, DATASET, LIB = ROOT / 'mp4s', ROOT / 'dataset', ROOT / 'lib' / 'bsl1k'
CHANNELS = {'prima': 'PrimaTV', 'protv': 'ProTV', 'digi24': 'Digi24'}

WINDOW, STRIDE = 8, 2  # frames
RESIZE, CROP = 224, 224  # the crop keeps the whole shorter side
BATCH = 32  # windows per forward pass (256 frames); larger batches are not faster on an RTX 3060

# downloaded on first use (the bsl1k repository has no license, so it is not copied here)
FILES = {
    'i3d.py': ('https://raw.githubusercontent.com/gulvarol/bsl1k/'
               'bfeaa3a8463152a79bcfd0fafd358c4d203b4c16/models/i3d.py',
               '293b629125be89d2e643cf0d0383784e97b40cc002246de4fbf3a7be5b315a49'),
    'bsl5k.pth.tar': ('https://www.robots.ox.ac.uk/~vgg/research/bslattend/data/bsl5k.pth.tar',
                      '6430592464a357dfdaa7f31973cb684663237655fdf23f3999608d162167fc6f'),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_model(device):
    LIB.mkdir(parents=True, exist_ok=True)
    for name, (url, want) in FILES.items():
        path = LIB / name
        if not path.exists():
            print(f'downloading {url}', flush=True)
            urllib.request.urlretrieve(url, path.with_name(name + '.part'))
            path.with_name(name + '.part').replace(path)
        if sha256(path) != want:
            sys.exit(f'{path} is not the pinned version: delete it and run again')
    spec = importlib.util.spec_from_file_location('bsl1k_i3d', LIB / 'i3d.py')
    i3d = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(i3d)
    model = i3d.InceptionI3d(num_classes=5383, num_in_frames=WINDOW, include_embds=True)
    state = torch.load(LIB / 'bsl5k.pth.tar', map_location='cpu', weights_only=True)['state_dict']
    model.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()})
    return model.eval().to(device, memory_format=torch.channels_last_3d)  # faster fp16 convolutions


def segments(splits):
    """(path relative to mp4s/ and dataset/ without extension, number of frames)"""
    for channel, folder in CHANNELS.items():
        manifest = json.loads((ROOT / 'manifests' / f'{channel}_manifest.json').read_text())
        for item in manifest['items']:
            if item['split'] in splits:
                sub = item['id'] if channel == 'prima' else item['date'].replace('-', '/')
                for segment in item['segments']:
                    yield Path(folder, sub, f"segment_{segment['index']}"), segment['end'] - segment['start']


def preprocess(bgr):
    """One frame, as the reference does at test time -> (3, 224, 224) float32."""
    rgb = bgr[:, :, ::-1].astype(np.float32) / 255
    h, w = rgb.shape[:2]
    nh, nw = (RESIZE, int(RESIZE * w / h)) if w > h else (int(RESIZE * h / w), RESIZE)
    rgb = cv2.resize(rgb, (nw, nh))
    y, x = int((nh - CROP) / 2), int((nw - CROP) / 2)
    return ((rgb[y:y + CROP, x:x + CROP] - 0.5) / 1.0).transpose(2, 0, 1)


def decoded(video):
    """The video's frames, preprocessed on another thread so that the GPU never waits."""
    frames = queue.Queue(maxsize=256)

    def work():
        cap = cv2.VideoCapture(str(video))
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                frames.put(torch.from_numpy(preprocess(bgr)).pin_memory())
            frames.put(None)
        except Exception as exc:
            frames.put(exc)
        finally:
            cap.release()

    threading.Thread(target=work, daemon=True).start()
    while (frame := frames.get()) is not None:
        if isinstance(frame, Exception):
            raise frame
        yield frame


@torch.inference_mode()
def features(model, video, device):
    """(features (N, 1024) float32, frames read)"""
    out = []

    def run(windows):  # padded to BATCH: one input shape, so cuDNN picks its algorithms once
        x = torch.stack(windows + [windows[-1]] * (BATCH - len(windows)))
        with torch.autocast('cuda', dtype=torch.float16):
            out.append(model(x)['embds'].flatten(1)[:len(windows)].float())

    count, frames, windows = 0, [], []  # frames: the next window, so far
    for frame in decoded(video):
        count += 1
        frames.append(frame.to(device, non_blocking=True))
        if len(frames) == WINDOW:
            windows.append(torch.stack(frames, dim=1))  # (3, WINDOW, 224, 224)
            del frames[:STRIDE]
            if len(windows) == BATCH:
                run(windows)
                windows = []
    if windows:
        run(windows)
    feats = torch.cat(out).cpu().numpy() if out else np.zeros((0, 1024), np.float32)
    return feats, count


def extract(model, rel, device):
    """None if done, else the error."""
    video = MP4S / f'{rel}.mp4'
    out, part = DATASET / f'{rel}.bsl5k.npy', DATASET / f'{rel}.bsl5k.part.npy'
    try:
        feats, frames = features(model, video, device)
        expected = int(subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
             'stream=nb_frames', '-of', 'csv=p=0', str(video)],
            capture_output=True, text=True, check=True).stdout)
        if frames != expected:
            raise RuntimeError(f'{frames} frames read, the video has {expected}')
        if len(feats) != max(0, (frames - WINDOW) // STRIDE + 1) or not np.isfinite(feats).all():
            raise RuntimeError(f'unexpected features: {feats.shape}')
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(part, feats.astype(np.float16))
        part.replace(out)
    except Exception as exc:
        part.unlink(missing_ok=True)
        return f'{type(exc).__name__}: {exc}'


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-s', '--split', default='train,val,test',
                        help='comma-separated splits to extract (default: all)')
    splits = set(parser.parse_args().split.split(','))
    if not splits <= {'train', 'val', 'test'}:
        parser.error('--split takes train, val and/or test')
    if not torch.cuda.is_available():
        sys.exit('no CUDA GPU')
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    model = load_model(device)

    frames = {rel: n for rel, n in segments(splits) if not (DATASET / f'{rel}.bsl5k.npy').exists()}
    missing = [rel for rel in frames if not (MP4S / f'{rel}.mp4').exists()]
    if missing:
        sys.exit(f'{len(missing)} videos are not in mp4s/ (first: {missing[0]}.mp4): run download.sh')
    todo = sorted(frames, key=frames.get, reverse=True)  # long ones first
    total, done, failed, t0 = sum(frames.values()), 0, 0, time.time()
    print(f'{len(todo)} segments to do', flush=True)
    for n, rel in enumerate(todo, 1):
        error = extract(model, rel, device)
        failed += bool(error)
        done += frames[rel]
        hours = (time.time() - t0) / 3600
        print(f'[{n}/{len(todo)}] {rel} {error or "ok"} '
              f'({hours:.1f} h, ~{hours * (total - done) / done:.1f} h left)', flush=True)
    print(f'{failed} failed' if failed else 'all done')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
