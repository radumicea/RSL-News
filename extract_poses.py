#!/usr/bin/env python3
"""Landmarks of every segment: mp4s/<channel>/.../segment_<i>.mp4 ->
dataset/<channel>/.../segment_<i>.pose.npz (body, hands) and segment_<i>.face.npz
(face mesh), one entry per video frame, made by mp_holistic_plus (the arrays are
described in its README.md). Read them with load() below:

    from extract_poses import load
    pose = load('dataset/ProTV/2021/03/16/segment_1.pose.npz')             # all frames
    face = load('dataset/ProTV/2021/03/16/segment_1.face.npz', 500, 650)   # frames 500..649

Needs ffprobe, an NVIDIA GPU and the packages of requirements.txt. The first run
builds the TensorRT engines for the GPU (a few minutes, once). Stop it and run it
again to resume: finished videos are skipped."""

import argparse
import json
import math
import multiprocessing
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import zstandard
from mp_holistic_plus import common   # numpy and OpenCV only; the GPU parts load in the workers

ROOT = Path(__file__).resolve().parent
MP4S, DATASET = ROOT / 'mp4s', ROOT / 'dataset'

FACE = ('face_xyz', 'face_ok')  # these go to the .face.npz
CHUNK = 256  # frames compressed together; a clip only decodes the chunks it touches


def encode(arrays):
    """One chunk of frames of every array -> one zstd frame (level 19).

    The arrays one after another, each as: frames moved to the last axis; values taken
    as the little-endian unsigned ints of their bit patterns; each frame minus the
    previous one (the chunk's first frame minus zero), modulo 2^bits; then the bytes
    grouped by significance (all lowest bytes first). Consecutive frames are close, so
    the differences are small and their high bytes mostly zero, which zstd packs well."""
    planes = []
    for a in arrays.values():
        n = a.itemsize
        u = np.ascontiguousarray(np.moveaxis(a, 0, -1)).view(f'<u{n}')
        d = np.diff(u, axis=-1, prepend=np.zeros_like(u[..., :1]))
        planes.append(np.ascontiguousarray(d.view(np.uint8).reshape(-1, n).T).tobytes())
    return np.frombuffer(zstandard.ZstdCompressor(level=19).compress(b''.join(planes)), np.uint8)


def decode(blob, fmt, frames):
    data, pos, out = zstandard.ZstdDecompressor().decompress(blob), 0, {}
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


def save(path, arrays):
    """{name: array, frames first} -> .npz with `format`, a JSON string {chunk, frames,
    arrays: {name: [dtype, shape of one frame]}}, and chunk_<i:05d>, frames
    [i * chunk, (i + 1) * chunk) of every array, encoded."""
    frames = len(next(iter(arrays.values())))
    fmt = {'chunk': CHUNK, 'frames': frames,
           'arrays': {k: [a.dtype.str, list(a.shape[1:])] for k, a in arrays.items()}}
    chunks = {f'chunk_{i:05d}': encode({k: a[s:s + CHUNK] for k, a in arrays.items()})
              for i, s in enumerate(range(0, frames, CHUNK))}
    np.savez(path, format=np.array(json.dumps(fmt)), **chunks)


def load(path, start=0, stop=None):
    """Frames [start, stop) of a .pose.npz or .face.npz (default: all) -> {name: array}.
    Only the chunks holding them are read."""
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
    return out


def outputs(video):
    base = DATASET / video.relative_to(MP4S).with_suffix('')
    return base.with_suffix('.pose.npz'), base.with_suffix('.face.npz')


def finish(video, raw, error, results):
    """A video's raw output -> checked .pose.npz and .face.npz (written as .part files first)."""
    video = Path(video)
    pose, face = outputs(video)
    parts = [p.with_suffix('.part.npz') for p in (pose, face)]
    try:
        if error:
            raise RuntimeError(error)
        with np.load(raw) as z:
            arrays = {k: z[k] for k in z.files}
        expected = int(subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
             'stream=nb_frames', '-of', 'csv=p=0', str(video)],
            capture_output=True, text=True, check=True).stdout)
        if len(arrays['pose_ok']) != expected:
            raise RuntimeError(f"{len(arrays['pose_ok'])} frames extracted, the video has {expected}")
        save(parts[0], {k: a for k, a in arrays.items() if k not in FACE})
        save(parts[1], {k: arrays[k] for k in FACE})
        got = {**load(parts[0]), **load(parts[1])}
        if got.keys() != arrays.keys() or any(
                (a.dtype, a.shape, a.tobytes()) != (got[k].dtype, got[k].shape, got[k].tobytes())
                for k, a in arrays.items()):
            raise RuntimeError('the files do not decode to the extracted arrays')
        parts[1].replace(face)
        parts[0].replace(pose)
        error = None
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        for p in (Path(raw), *parts):
            p.unlink(missing_ok=True)
    results.put((str(video), error))


def build_engines():
    """Run once in its own process, before the workers, so that they don't all build them."""
    from mp_holistic_plus import pipeline
    pipeline.HolisticPlus(batch=1).close()


def worker(jobs, results, batch):
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[var] = '1'
    cv2.setNumThreads(1)
    os.dup2(os.open(os.devnull, os.O_WRONLY), 2)  # MediaPipe's start-up messages
    from mp_holistic_plus import pipeline
    h = pipeline.HolisticPlus(batch=batch)
    with ThreadPoolExecutor(1) as writer:   # a finished video is compressed while the others go on
        written = []
        h.run(iter(jobs.get, None),
              on_done=lambda v, raw, err, n: written.append(writer.submit(finish, v, raw, err, results)))
        for w in written:
            w.result()
    h.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-w', '--workers', type=int, default=3,
                        help='processes sharing the GPU (default: 3, the fastest on an RTX 3060)')
    parser.add_argument('-b', '--batch', type=int, default=16,
                        help='videos per process, advancing in lockstep (default: 16)')
    args = parser.parse_args()
    if not MP4S.is_dir():
        sys.exit('no mp4s/: run download.sh first')
    try:
        common.verified_models()      # the pinned MediaPipe bundles, downloaded on first use
    except RuntimeError as exc:
        sys.exit(str(exc))

    frames = {v: int(cv2.VideoCapture(str(v)).get(cv2.CAP_PROP_FRAME_COUNT))
              for v in MP4S.rglob('segment_*.mp4') if not all(p.exists() for p in outputs(v))}
    videos = sorted(frames, key=frames.get, reverse=True)  # long ones first
    print(f'{len(videos)} videos to do', flush=True)
    if not videos:
        return 0
    ctx = multiprocessing.get_context('spawn')
    b = ctx.Process(target=build_engines)
    b.start(); b.join()
    if b.exitcode != 0:
        sys.exit('could not build the TensorRT engines')
    jobs, results = ctx.Queue(), ctx.Queue()
    for v in videos:
        jobs.put((str(v), str(outputs(v)[0].with_suffix('.raw.part.npz'))))
    for _ in range(args.workers):
        jobs.put(None)
    procs = [ctx.Process(target=worker, args=(jobs, results, args.batch)) for _ in range(args.workers)]
    for p in procs:
        p.start()
    total, done, failed, t0, n = sum(frames.values()), 0, 0, time.time(), 0
    while n < len(videos):
        try:
            video, error = results.get(timeout=60)
        except Exception:
            if not any(p.is_alive() for p in procs):
                break
            continue
        n += 1
        failed += bool(error)
        done += frames[Path(video)]
        hours = (time.time() - t0) / 3600
        print(f'[{n}/{len(videos)}] {Path(video).relative_to(MP4S)} {error or "ok"} '
              f'({hours:.1f} h, ~{hours * (total - done) / max(done, 1):.1f} h left)', flush=True)
    for p in procs:
        p.join()
    missing = len(videos) - n
    if missing:
        print(f'{missing} videos not finished (a worker stopped): run again to do them')
    print(f'{failed} failed' if failed else ('all done' if not missing else ''))
    return 1 if failed or missing else 0


if __name__ == '__main__':
    sys.exit(main())
