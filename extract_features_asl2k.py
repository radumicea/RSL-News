#!/usr/bin/env python3
"""ASL-2k I3D features of the segments in the chosen splits:
mp4s/<channel>/.../segment_<i>.mp4 -> dataset/<channel>/.../segment_<i>.asl2k.npy,
float16, shape (N, 1024); feature k is the model's time step for frames [2k, 2k + 8) of the
segment, computed with the frames around it in view (below).

The model is WLASL's I3D (Li et al., WACV 2020) with its WLASL2000 weights: the Kinetics I3D
fine-tuned on 2,000 ASL signs. Frames are preprocessed as in the reference code
(code/I3D/datasets/nslt_dataset.py), except for the resize: BGR (OpenCV's frames, which the
reference does not convert to RGB), shorter side resized to 224 (bilinear), centre crop 224,
x / 255 * 2 - 1. The resize is that of extract_features_bsl5k.py (FEATURES.md).

The model was trained on 64-frame clips, and on 8-frame windows alone it sees too little motion
(its own classifier is at chance there). So it runs over long stretches of the segment: Mixed_5c
has one time step per 8 frames, each computed with the frames around it in view, and each step,
averaged over space (fp16 autocast), is a feature. Four runs, starting 0, 2, 4 and 6 frames into
the segment, interleave into one feature every 2 frames: the layout of the BSL-5k features.
A stretch is CHUNK frames plus CONTEXT frames on each side, whose steps are not kept: the
model's temporal receptive field is about 99 frames, so the kept steps are those of a run over
the whole segment.

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
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
MP4S, DATASET, LIB = ROOT / 'mp4s', ROOT / 'dataset', ROOT / 'lib' / 'wlasl'
CHANNELS = {'prima': 'PrimaTV', 'protv': 'ProTV', 'digi24': 'Digi24'}

WINDOW, STRIDE = 8, 2  # frames
RESIZE, CROP = 224, 224  # the crop keeps the whole shorter side
CHUNK, CONTEXT = 256, 48  # frames kept per run, and frames of context on each side (multiples of 8)
OFFSETS = (0, 2, 4, 6)    # run starts: their steps interleave into one feature every STRIDE frames

# downloaded on first use (the WLASL repository is under C-UDA, a data license, so it is not
# copied here); the weights are the README's "WLASL pre-trained weights", a zip of four subsets
FILES = {
    'pytorch_i3d.py': ('https://raw.githubusercontent.com/dxli94/WLASL/'
                       'ac00e6be631c1a2a486621b65f202219f3964d6b/code/I3D/pytorch_i3d.py',
                       '1b35b81b4dc8ea6ef55c87063515d058761fd25caf9381bad3dcc0ed5f3c3de5'),
    'asl2000.pt': ('https://drive.usercontent.google.com/download?'
                   'id=1jALimVOB69ifYkeT0Pe297S1z4U3jC48&export=download&confirm=t',
                   '243a19e6deef3becffbfc5b7dd8adb32916c8bee482565ca243c082584732620'),
}
MEMBER = 'archived/asl2000/FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt'


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
            if name == 'asl2000.pt':  # keep only the WLASL2000 weights of the 197 MB zip
                with zipfile.ZipFile(path.with_name(name + '.part')) as archive:
                    weights = archive.read(MEMBER)
                path.with_name(name + '.part').write_bytes(weights)
            path.with_name(name + '.part').replace(path)
        if sha256(path) != want:
            sys.exit(f'{path} is not the pinned version: delete it and run again')
    spec = importlib.util.spec_from_file_location('wlasl_i3d', LIB / 'pytorch_i3d.py')
    i3d = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(i3d)
    model = i3d.InceptionI3d(400, in_channels=3)
    model.replace_logits(2000)
    model.load_state_dict(torch.load(LIB / 'asl2000.pt', map_location='cpu', weights_only=True))
    model.avg_pool = torch.nn.AdaptiveAvgPool3d((None, 1, 1))  # over space only: one feature per step
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
    """One frame, as the reference does -> (3, 224, 224) float32, channels in BGR order."""
    h, w = bgr.shape[:2]
    nh, nw = (RESIZE, int(RESIZE * w / h)) if w > h else (int(RESIZE * h / w), RESIZE)
    bgr = cv2.resize(bgr, (nw, nh))
    y, x = int((nh - CROP) / 2), int((nw - CROP) / 2)
    return (bgr[y:y + CROP, x:x + CROP].astype(np.float32) / 255 * 2 - 1).transpose(2, 0, 1)


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
    """(features (N, 1024) float32, frames read); feature k is the Mixed_5c step of frames
    [2k, 2k + 8), from the run starting 2k mod 8 frames into the segment."""
    out = {}
    buffer, first, ended = [], 0, False   # buffer[i] is frame first + i, on the GPU
    frames = decoded(video)
    start = 0                             # the stretch whose steps are kept: frames [start + o, start + o + CHUNK)
    while True:
        while not ended and first + len(buffer) < start + CHUNK + CONTEXT + OFFSETS[-1]:
            frame = next(frames, None)
            if frame is None:
                ended = True
            else:
                buffer.append(frame.to(device, non_blocking=True))
        end = first + len(buffer)         # frames read so far (all of them once ended)
        if start >= end:
            break
        for o in OFFSETS:
            lo, hi = max(o, start + o - CONTEXT), min(end, start + o + CHUNK + CONTEXT)  # lo = o mod 8
            hi = lo + (hi - lo) // 8 * 8   # the model's padding, so its grid of steps, depends on
            if hi - lo < WINDOW:           # the length's parity: keep it a multiple of 8
                continue
            x = torch.stack(buffer[lo - first:hi - first], dim=1)[None]   # (1, 3, hi - lo, 224, 224)
            with torch.autocast('cuda', dtype=torch.float16):
                steps = model.extract_features(x)[0, :, :, 0, 0].T.float()  # (ceil((hi - lo) / 8), 1024)
            for i, f in enumerate(range(lo, hi, 8)):   # step i: frames [f, f + 8)
                if start + o <= f < start + o + CHUNK and f + WINDOW <= end:
                    out[f // STRIDE] = steps[i]
        start += CHUNK
        drop = min(max(0, start - CONTEXT - first), len(buffer))
        del buffer[:drop]
        first += drop
    n = max(0, (end - WINDOW) // STRIDE + 1)
    if sorted(out) != list(range(n)):
        raise RuntimeError(f'{len(out)} features for {n} windows')
    feats = torch.stack([out[k] for k in range(n)]).cpu().numpy() if n else np.zeros((0, 1024), np.float32)
    return feats, end


def extract(model, rel, device):
    """None if done, else the error."""
    video = MP4S / f'{rel}.mp4'
    out, part = DATASET / f'{rel}.asl2k.npy', DATASET / f'{rel}.asl2k.part.npy'
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

    frames = {rel: n for rel, n in segments(splits) if not (DATASET / f'{rel}.asl2k.npy').exists()}
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
