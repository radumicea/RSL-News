#!/usr/bin/env python3
"""Landmarks for every segment, on the GPU: mp4s/<channel>/.../segment_<i>.mp4 ->
dataset/<channel>/.../segment_<i>.pose.npz (body, hands, meta) and
segment_<i>.face.npz (face mesh), one entry per video frame (arrays:
lib/mp_holistic_plus/README.md). The arrays are compressed losslessly; read
them with load() below, which needs only numpy:

    from extract_poses import load
    pose = load('dataset/ProTV/2021/03/16/segment_1.pose.npz')             # all frames
    face = load('dataset/ProTV/2021/03/16/segment_1.face.npz', 500, 650)   # frames 500..649

Extracting needs ffprobe and
    pip install mediapipe==1.0.1 ai-edge-litert==2.2.0
Stop it and run it again to resume: finished videos are skipped."""

import argparse
import dataclasses
import json
import lzma
import math
import multiprocessing
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'lib' / 'mp_holistic_plus'))

MP4S, DATASET, ASSETS = ROOT / 'mp4s', ROOT / 'dataset', ROOT / 'lib' / 'mp_holistic_plus' / 'assets'

# versioned model URLs (the library's "latest" ones can change) and their sha256
_BASE = 'https://storage.googleapis.com/mediapipe-models'
MODEL_URLS = {name: f'{_BASE}/{folder}/float16/1/{name}' for name, folder in [
    ('pose_landmarker_heavy.task', 'pose_landmarker/pose_landmarker_heavy'),
    ('face_landmarker.task', 'face_landmarker/face_landmarker'),
    ('hand_landmarker.task', 'hand_landmarker/hand_landmarker')]}
SHA256 = {
    'pose_landmarker_heavy.task': '64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b',
    'face_landmarker.task': '64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff',
    'hand_landmarker.task': 'fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1',
    'hand_landmarks_detector.tflite': '6acda74af3fbf40e68265c20c7394b2bad81a16a481dcd79ad7a081887c3d6b9',
    'hand_crop_mlp.npz': '8a89d24fbbcc16b94c112bbd62f6a3755f8ed203d3a9ec350dc5904eedf9fec8',
}

FACE = ('face_xyz', 'face_ok')  # these go to the .face.npz
CHUNK = 256  # frames compressed together; a clip only decodes the chunks it touches
CODEC = ('frames in independent chunks of `chunk` frames, chunk i an xz stream in member '
         'chunk_<i:05d>; inside it, the arrays one after another, in the order of `arrays`, '
         'each with frames moved to the last axis, values as the little-endian unsigned ints '
         'of their bit patterns, each frame minus the previous one (the first frame of a '
         'chunk minus zero) modulo 2^bits, and bytes grouped by significance')


def encode(arrays):
    """One chunk of frames of every array -> one xz stream."""
    planes = []
    for a in arrays.values():
        n = a.itemsize
        u = np.ascontiguousarray(np.moveaxis(a, 0, -1)).view(f'<u{n}')
        d = np.diff(u, axis=-1, prepend=np.zeros_like(u[..., :1]))
        planes.append(np.ascontiguousarray(d.view(np.uint8).reshape(-1, n).T).tobytes())
    return np.frombuffer(lzma.compress(b''.join(planes)), np.uint8)


def decode(blob, fmt, frames):
    data, pos, out = lzma.decompress(blob), 0, {}
    for k, (dtype, shape) in fmt['arrays'].items():
        n = np.dtype(dtype).itemsize
        size = n * frames * math.prod(shape)
        planes = np.frombuffer(data, np.uint8, size, pos).reshape(n, -1)
        pos += size
        d = np.ascontiguousarray(planes.T).view(f'<u{n}').reshape(*shape, frames)
        out[k] = np.moveaxis(np.cumsum(d, axis=-1, dtype=d.dtype).view(dtype), -1, 0)
    if pos != len(data):
        raise ValueError('chunk size does not match its format')
    return out


def save(path, arrays, meta=None):
    frames = len(next(iter(arrays.values())))
    fmt = {'codec': CODEC, 'chunk': CHUNK, 'frames': frames,
           'arrays': {k: [a.dtype.str, list(a.shape[1:])] for k, a in arrays.items()}}
    chunks = {f'chunk_{i:05d}': encode({k: a[s:s + CHUNK] for k, a in arrays.items()})
              for i, s in enumerate(range(0, frames, CHUNK))}
    extra = {} if meta is None else {'meta': np.array(json.dumps(meta))}
    np.savez(path, format=np.array(json.dumps(fmt)), **extra, **chunks)


def load(path, start=0, stop=None):
    """Frames [start, stop) of a .pose.npz or .face.npz (default: all) -> {name: array},
    plus 'meta' (a dict) from a .pose.npz. Only the chunks holding them are read."""
    with np.load(path) as z:
        fmt = json.loads(str(z['format']))
        frames, chunk = fmt['frames'], fmt['chunk']
        stop = frames if stop is None else stop
        if not 0 <= start <= stop <= frames:
            raise ValueError(f'frames [{start}, {stop}) are not within [0, {frames})')
        first = start // chunk
        parts = [decode(z[f'chunk_{i:05d}'], fmt, min(chunk, frames - i * chunk))
                 for i in range(first, -(-stop // chunk))]
        out = {k: np.concatenate([p[k] for p in parts])[start - first * chunk:stop - first * chunk]
               if parts else np.zeros((0, *shape), dtype) for k, (dtype, shape) in fmt['arrays'].items()}
        if 'meta' in z.files:
            out['meta'] = json.loads(str(z['meta']))
    return out


def outputs(video):
    base = DATASET / video.relative_to(MP4S).with_suffix('')
    return base.with_suffix('.pose.npz'), base.with_suffix('.face.npz')


def worker_init():
    os.chdir(MP4S)  # so the output's meta records the video's path relative to mp4s/
    os.dup2(os.open(os.devnull, os.O_WRONLY), 2)  # MediaPipe's start-up messages, for every video
    cv2.setNumThreads(1)  # one thread per worker: many workers, not many threads each


def extract(video):
    """None if done, else the error. Everything is written to .part files first,
    so an interrupted video is redone rather than skipped."""
    import holistic_plus as hp
    pose, face = outputs(video)
    raw, parts = pose.with_suffix('.raw.part.npz'), [p.with_suffix('.part.npz') for p in (pose, face)]
    try:
        hp._run_one((str(video.relative_to(MP4S)), str(raw), dataclasses.asdict(hp.Options(hand_threads=1)), str(ASSETS)))
        with np.load(raw) as z:
            arrays = {k: z[k] for k in z.files if k != 'meta'}
            meta = json.loads(str(z['meta']))
        expected = int(subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
             'stream=nb_frames', '-of', 'csv=p=0', str(video)],
            capture_output=True, text=True, check=True).stdout)
        if len(arrays['pose_ok']) != expected:
            raise RuntimeError(f"{len(arrays['pose_ok'])} frames extracted, the video has {expected}")
        if meta['delegate'] != 'GPU':
            raise RuntimeError('the GPU could not be used')
        save(parts[0], {k: a for k, a in arrays.items() if k not in FACE}, meta)
        save(parts[1], {k: arrays[k] for k in FACE})
        verify(parts, arrays, meta)
        parts[1].replace(face)
        parts[0].replace(pose)
    except Exception as exc:
        return f'{type(exc).__name__}: {exc}'
    finally:
        for p in (raw, *parts):
            p.unlink(missing_ok=True)


def verify(files, arrays, meta):
    """The files must decode bit for bit to the arrays: all of them, and random clips."""
    def same(ours, got):
        return got.keys() == ours.keys() and all(
            (a.dtype, a.shape, a.tobytes()) == (got[k].dtype, got[k].shape, got[k].tobytes())
            for k, a in ours.items())
    frames = len(arrays['pose_ok'])
    rng = np.random.default_rng(frames)
    clips = [(0, frames)] + [(s, int(rng.integers(s, min(s + 600, frames) + 1)))
                             for s in rng.integers(0, frames + 1, 20).tolist()]
    for start, stop in clips:
        got = {**load(files[0], start, stop), **load(files[1], start, stop)}
        if got.pop('meta') != meta or not same({k: a[start:stop] for k, a in arrays.items()}, got):
            raise RuntimeError(f'frames [{start}, {stop}) do not decode to the extracted arrays')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-w', '--workers', type=int, default=12,
                        help='videos at once, sharing the GPU (default: 12, the fastest '
                             'on an RTX 3060 with 8 CPU cores)')
    workers = parser.parse_args().workers
    if not MP4S.is_dir():
        sys.exit('no mp4s/: run download.sh first')
    import holistic_plus as hp
    hp.MODEL_URLS.update(MODEL_URLS)
    hp.ensure_models(ASSETS)
    for name, want in SHA256.items():
        if hp._sha256(ASSETS / name) != want:
            sys.exit(f'{ASSETS / name} is not the pinned version: delete it and run again')

    frames = {v: int(cv2.VideoCapture(str(v)).get(cv2.CAP_PROP_FRAME_COUNT))
              for v in MP4S.rglob('segment_*.mp4') if not all(p.exists() for p in outputs(v))}
    videos = sorted(frames, key=frames.get, reverse=True)  # long ones first
    total, done, failed, t0 = sum(frames.values()), 0, 0, time.time()
    print(f'{len(videos)} videos to do', flush=True)
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[var] = '1'  # inherited by the workers
    with ProcessPoolExecutor(workers, multiprocessing.get_context('spawn'), worker_init) as pool:
        futures = {pool.submit(extract, v): v for v in videos}
        for n, future in enumerate(as_completed(futures), 1):
            video, error = futures[future], future.result()
            failed += bool(error)
            done += frames[video]
            hours = (time.time() - t0) / 3600
            print(f'[{n}/{len(videos)}] {video.relative_to(MP4S)} {error or "ok"} '
                  f'({hours:.1f} h, ~{hours * (total - done) / done:.1f} h left)', flush=True)
    print(f'{failed} failed' if failed else 'all done')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
