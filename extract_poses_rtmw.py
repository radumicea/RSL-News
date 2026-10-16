#!/usr/bin/env python3
"""Whole-body keypoints of every segment, by RTMW-x: mp4s/<channel>/.../segment_<i>.mp4 ->
dataset/<channel>/.../segment_<i>.pose_rtmw.npz, one entry per video frame:

    xy     (T, 133, 2) float16  the 133 COCO-WholeBody points: body 0-16, feet 17-22, face 23-90,
                                left hand 91-111, right hand 112-132 (the person's left and right;
                                each hand in MediaPipe's 21-point order). x and y are fractions of
                                the frame's width and height; points outside it fall outside [0, 1].
    score  (T, 133) uint8       RTMW's score for each point (its SimCC maximum, about 0 to 11)
                                x 20, rounded. Points it cannot see score low.

RTMW-x runs through rtmw-torch (MMPose's model re-implemented in plain PyTorch; its README has
the measurements), with the whole frame as the person's box, in fp16 and with DARK refinement.
RTMW-x always outputs both hands: rtmw_torch.hands.hand_valid tells which are real. Read a file
with extract_poses_mp.load, which decodes only the frames asked for:

    from extract_poses_mp import load
    from rtmw_torch.hands import hand_valid
    r = load('dataset/ProTV/2021/03/16/segment_1.pose_rtmw.npz', 500, 650)   # frames 500..649
    xy, score = r['xy'] * (w, h), r['score'] / 20                         # w x h: the mp4's size
    valid = hand_valid(xy, score, w, h)                                    # (T, 2): left, right

Needs ffprobe, a CUDA GPU and the packages of requirements.txt. The first run downloads the
checkpoint (370 MB). Stop it and run it again to resume: finished videos are skipped."""

import argparse
import queue
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from rtmw_torch.model import RTMW

from extract_poses_mp import load, save

ROOT = Path(__file__).resolve().parent
MP4S, DATASET = ROOT / 'mp4s', ROOT / 'dataset'
SCORE_SCALE = 20


def outputs(video):
    return DATASET / video.relative_to(MP4S).with_suffix('.pose_rtmw.npz')


def finish(video, xy, score, results):
    """A video's points (frame pixels) -> its checked .pose_rtmw.npz (written as a .part file first)."""
    path = outputs(video)
    part = path.with_suffix('.part.npz')
    try:
        cap = cv2.VideoCapture(str(video))
        size = np.array([cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)], np.float32)
        cap.release()
        arrays = {'xy': (xy / size).astype(np.float16),
                  'score': np.clip(np.round(score * SCORE_SCALE), 0, 255).astype(np.uint8)}
        expected = int(subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
             'stream=nb_frames', '-of', 'csv=p=0', str(video)],
            capture_output=True, text=True, check=True).stdout)
        if len(xy) != expected:
            raise RuntimeError(f'{len(xy)} frames read, the video has {expected}')
        path.parent.mkdir(parents=True, exist_ok=True)
        save(part, arrays)
        got = load(part)
        if got.keys() != arrays.keys() or any(
                (a.dtype, a.shape, a.tobytes()) != (got[k].dtype, got[k].shape, got[k].tobytes())
                for k, a in arrays.items()):
            raise RuntimeError('the file does not decode to the extracted arrays')
        part.replace(path)
        error = None
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        part.unlink(missing_ok=True)
    results.put((video, error))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--flip', action='store_true',
                        help="MMPose's flip test: also run the mirrored frame and average "
                             '(about 2.5%% nearer to other estimators, half the speed)')
    parser.add_argument('--refine', action=argparse.BooleanOptionalAction, default=True,
                        help='DARK refinement of the points between half-pixel steps (default: on)')
    parser.add_argument('--precision', choices=('fp16', 'tf32', 'fp32'), default='fp16',
                        help='arithmetic of the model (default: fp16)')
    parser.add_argument('-b', '--batch', type=int, default=64, help='frames per forward pass (default: 64)')
    args = parser.parse_args()
    if not MP4S.is_dir():
        sys.exit('no mp4s/: run download.sh first')
    if not torch.cuda.is_available():
        sys.exit('needs a CUDA GPU')

    frames = {v: int(cv2.VideoCapture(str(v)).get(cv2.CAP_PROP_FRAME_COUNT))
              for v in MP4S.rglob('segment_*.mp4') if not outputs(v).exists()}
    videos = sorted(frames, key=frames.get, reverse=True)  # long ones first
    print(f'{len(videos)} videos to do', flush=True)
    if not videos:
        return 0
    model = RTMW.pretrained('cuda', args.precision)
    results = queue.Queue()
    total, done, failed, t0, n = sum(frames.values()), 0, 0, time.time(), 0

    def report():
        nonlocal n, failed, done
        while not results.empty():
            video, error = results.get()
            n += 1
            failed += bool(error)
            done += frames[video]
            hours = (time.time() - t0) / 3600
            print(f'[{n}/{len(videos)}] {video.relative_to(MP4S)} {error or "ok"} '
                  f'({hours:.1f} h, ~{hours * (total - done) / max(done, 1):.1f} h left)', flush=True)

    with ThreadPoolExecutor(1) as writer:   # a finished video is compressed while the next one runs
        for video in videos:
            try:
                xy, score = model.video(video, args.batch, args.flip, args.refine)
            except Exception as exc:
                results.put((video, f'{type(exc).__name__}: {exc}'))
            else:
                writer.submit(finish, video, xy, score, results)
            report()
    report()
    print(f'{failed} failed' if failed else 'all done')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
