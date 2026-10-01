#!/usr/bin/env python3
"""Download the interpreter segments of the PrimaTV news videos.

For every entry in manifests/prima_manifest.json this produces the listed
segment, cropped to the interpreter box, video stream only.

Produces:

    <root>/<video-id>/segment_<i>.mp4

Sources are YouTube. For each video, yt-dlp downloads the video-only stream
into a temporary file under <root>, the segment is cut from that file, and the
temporary file is removed. The segments cover most of each video, so this costs
little more than fetching the segment alone, and it avoids YouTube refusing
direct range requests to its media URLs.

YouTube requires all of the following for media formats:

  * a cookies file (--cookies), exported from a logged-in YouTube session
  * a JavaScript runtime (node or deno)
  * yt-dlp's challenge-solver script (--remote-components ejs:github)
  * a PO-token provider plugin for yt-dlp, e.g. bgutil-ytdlp-pot-provider
    (https://github.com/Brainicism/bgutil-ytdlp-pot-provider) with its
    provider server running

Without these yt-dlp either returns no media formats or YouTube answers 403.

The format is pinned to the entry's source_height. The crop coordinates are
only valid for that resolution, and a video that was 720p when the dataset was
built may offer 1080p today, so picking "best" would silently mis-crop.

Segment bounds are frame indices into the source video, converted with the
stream's own start_time taken into account.

By default one video is downloaded at a time; --workers N downloads N videos
at once (each keeps its source file under <root> until its segment is cut).

Each segment is written to a .part file and renamed to .mp4 only once it is
complete; segments whose .mp4 already exists are skipped, so the script can be
stopped and re-run. The manifest is only read.

Requires ffmpeg/ffprobe and yt-dlp on PATH (or set --yt-dlp).

Usage:
    python download_prima.py --root /path/to/output --dry-run
    python download_prima.py --root /path/to/output --cookies /path/to/cookies.txt
    (optionally with --workers N)
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / 'manifests' / 'prima_manifest.json'
LOG = ROOT / 'logs' / 'prima_download_log.json'

FPS = 25.0
CRF = '18'


def ytdlp_base(args) -> list[str]:
    cmd = [args.yt_dlp]
    if args.cookies:
        cmd += ['--cookies', str(args.cookies)]
    if args.js_runtime:
        cmd += ['--js-runtimes', args.js_runtime]
    if args.remote_components:
        cmd += ['--remote-components', 'ejs:github']
    return cmd


def download_stream(args, item: dict, workdir: Path) -> Path:
    """Download the video-only rendition matching the entry's source height."""
    height = item['source_height']
    workdir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ytdlp_base(args) + [
            '-f', f'bv[height={height}][protocol=https]/bv[height={height}]',
            '-o', str(workdir / f"{item['id']}.%(ext)s"),
            '--no-progress', '--print', 'after_move:filepath',
            item['url'],
        ],
        capture_output=True, text=True,
    )
    lines = [l for l in result.stdout.splitlines() if l.strip()]
    if result.returncode != 0 or not lines or not Path(lines[-1]).exists():
        raise RuntimeError(f'no {height}p stream: {result.stderr.strip()[-200:]}')
    return Path(lines[-1])


def stream_start_time(path: Path) -> float:
    """First frame's PTS. Ignoring it shifts every seek by start_time*fps."""
    probe = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=start_time', '-of', 'default=nw=1:nk=1', str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(probe.stdout.strip().split('\n')[0])
    except (ValueError, IndexError):
        return 0.0


def check_source(path: Path, item: dict) -> None:
    """Refuse to crop a stream that is not the resolution the crop assumes."""
    probe = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=width,height', '-of', 'csv=p=0:nk=1', str(path)],
        capture_output=True, text=True,
    )
    dims = probe.stdout.strip().split('\n')[0]
    if not dims:
        raise RuntimeError('could not probe source dimensions')
    width, height = (int(v) for v in dims.split(','))
    if height != item['source_height']:
        raise RuntimeError(
            f'source is {width}x{height}, crop assumes height '
            f"{item['source_height']}")


def grab_segment(path: Path, start_time: float, segment: dict,
                 crop: dict, dest: Path) -> int:
    """Cut one cropped segment. Returns bytes written."""
    first, last = segment['start'], segment['end']
    count = last - first + 1
    partial = dest.with_suffix('.part')

    result = subprocess.run(
        ['ffmpeg', '-v', 'error', '-y',
         '-ss', f'{start_time + first / FPS:.6f}',
         '-i', str(path),
         '-map', '0:v:0', '-frames:v', str(count),
         '-vf', f"crop={crop['w']}:{crop['h']}:{crop['x']}:{crop['y']}",
         '-an',
         '-vsync', 'vfr',
         '-c:v', 'libx264', '-crf', CRF, '-preset', 'veryfast',
         '-pix_fmt', 'yuv420p',
         '-f', 'mp4', str(partial)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f'ffmpeg failed: {result.stderr.strip()[:200]}')

    size = partial.stat().st_size
    if size == 0:
        partial.unlink(missing_ok=True)
        raise RuntimeError('empty output')

    partial.replace(dest)
    return size


def write_atomic(path: Path, text: str) -> None:
    """Write via a temporary file, so a run stopped mid-write never leaves a
    half-written manifest or log behind."""
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(text)
    tmp.replace(path)


def segment_path(item: dict, index: int, root: Path) -> Path:
    return root / item['id'] / f'segment_{index}.mp4'


def key(item: dict, index: int) -> str:
    return f"{item['id']}/segment_{index}"


def pending_segments(item: dict, root: Path) -> list[dict]:
    return [
        s for s in item['segments']
        if not segment_path(item, s['index'], root).exists()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True, help='output directory')
    parser.add_argument('--cookies', type=Path,
                        help='Netscape cookies.txt from a logged-in YouTube session')
    parser.add_argument('--yt-dlp', default='yt-dlp', help='path to yt-dlp')
    parser.add_argument('--js-runtime', default='node',
                        help='JS runtime for the challenge solver (node/deno)')
    parser.add_argument('--no-remote-components', dest='remote_components',
                        action='store_false',
                        help="don't let yt-dlp fetch its challenge-solver script")
    parser.add_argument('--workers', type=int, default=1,
                        help='videos downloaded at the same time (default: 1). '
                             'Be nice to the servers: please keep this low and raise '
                             'it only if you need to')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be at least 1')

    manifest = json.loads(MANIFEST.read_text())

    items = manifest['items']

    if args.dry_run:
        total = 0
        for item in items:
            for segment in pending_segments(item, args.root):
                print(f"{segment_path(item, segment['index'], args.root)}  "
                      f"[{segment['start']}-{segment['end']}] "
                      f"{item['source_height']}p crop "
                      f"{item['crop']['w']}x{item['crop']['h']}"
                      f"+{item['crop']['x']}+{item['crop']['y']}")
                total += 1
        n = sum(len(i['segments']) for i in items)
        print(f'\n{total} segments to download, {n - total} already downloaded')
        return 0

    if not args.cookies:
        print('warning: no --cookies given; YouTube will likely refuse media formats',
              file=sys.stderr)

    results = []
    lock = threading.Lock()

    def record(result: dict) -> None:
        with lock:
            results.append(result)
            LOG.parent.mkdir(parents=True, exist_ok=True)
            write_atomic(LOG, json.dumps(results, indent=2))

    def download_item(numbered: tuple[int, dict]) -> None:
        index, item = numbered
        todo = pending_segments(item, args.root)
        prefix = f"[{index}/{len(items)}] {item['id']}"
        # one temporary folder per video, so parallel downloads never share one
        workdir = args.root / '.download' / item['id']
        try:
            source = download_stream(args, item, workdir)
            check_source(source, item)
            start_time = stream_start_time(source)
        except Exception as exc:
            print(f'{prefix}  download failed: {exc}', flush=True)
            record({'id': item['id'], 'status': 'FAILURE', 'error': str(exc)})
            shutil.rmtree(workdir, ignore_errors=True)
            return

        for segment in todo:
            i = segment['index']
            dest = segment_path(item, i, args.root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                written = grab_segment(source, start_time, segment, item['crop'], dest)
                print(f'{prefix}  segment_{i}  {written / 1024 / 1024:.1f} MiB',
                      flush=True)
                record({'key': key(item, i), 'status': 'SUCCESS'})
            except Exception as exc:
                print(f'{prefix}  segment_{i} failed: {exc}', flush=True)
                record({'key': key(item, i), 'status': 'FAILURE', 'error': str(exc)})

        shutil.rmtree(workdir, ignore_errors=True)

    work = [(n, item) for n, item in enumerate(items, 1)
            if pending_segments(item, args.root)]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(download_item, work))
    try:
        (args.root / '.download').rmdir()
    except OSError:
        pass

    failures = [r for r in results if r['status'] == 'FAILURE']
    print(f'\n{len(results) - len(failures)} segments downloaded, '
          f'{len(failures)} failed')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())

