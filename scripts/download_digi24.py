#!/usr/bin/env python3
"""Download the interpreter segments of the Digi24 journal videos.

For every entry in manifests/digi24_manifest.json this fetches only what is
needed: each listed segment, cropped to the interpreter box, video stream only.
The source videos are ~2h at 1280x720; the output is a few 200x200 clips per
day, so this downloads a small fraction of the bytes a full fetch would.

Each entry names the rendition it is cropped from (source_height: 720); the crop
is only valid for that resolution, so a day whose 720p file is gone is reported
as failed rather than cropped from a smaller one.

Produces:

    <root>/<yyyy>/<mm>/<dd>/segment_<i>.mp4

Segment bounds [start, end) are frame indices into the source video (end
exclusive). They are converted with
the stream's own start_time taken into account -- seeking with a bare
frame/fps timestamp lands on the wrong frame whenever a stream does not start
at PTS 0.

By default downloads run one at a time; --workers N fetches N days at once.
Each segment is written to a .part file and renamed to .mp4 only once it is
complete; segments whose .mp4 already exists are skipped, so the script can be
stopped and re-run. The manifest is only read.

Requires ffmpeg/ffprobe on PATH.

Usage:
    python download_digi24.py --root /path/to/output --dry-run
    python download_digi24.py --root /path/to/output [--workers N]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / 'manifests' / 'digi24_manifest.json'
LOG = ROOT / 'logs' / 'digi24_download_log.json'

PLAYER_CFG = re.compile(r'<script[^>]+id="video-player-cfg"[^>]*>(.*?)</script>', re.S)
FPS = 25.0
CRF = '18'

USER_AGENT = (
    'Mozilla/5.0 (X11; Linux x86_64; rv:140.0) Gecko/20100101 Firefox/140.0'
)
PAGE_HEADERS = {
    'User-Agent': USER_AGENT,
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'ro-RO,ro;q=0.9,en;q=0.5',
}


def video_headers(video_src: str) -> list[str]:
    """The header block ffmpeg needs for the iw.ro CDN (none for a local file)."""
    if '://' not in video_src:
        return []
    return ['-headers', f'Referer: {video_src}\r\nUser-Agent: {USER_AGENT}\r\n']


def resolve_video_src(session: requests.Session, url: str, height: int) -> str:
    """Read the URL of the video's <height>p version off the article page."""
    response = session.get(url, headers=PAGE_HEADERS, timeout=30)
    response.raise_for_status()

    match = PLAYER_CFG.search(response.text)
    if not match:
        raise RuntimeError(f'no video-player-cfg in {url}')

    meta = json.loads(match.group(1))['new-info']['meta']
    versions = meta.get('versions') or {}

    if versions.get(f'{height}p.mp4'):
        return versions[f'{height}p.mp4']
    raise RuntimeError(f'no {height}p video in {url} (have {sorted(versions)})')


def check_source(video_src: str, height: int) -> None:
    """Refuse to crop a stream that is not the resolution the crop assumes."""
    probe = subprocess.run(
        ['ffprobe', '-v', 'error', *video_headers(video_src),
         '-select_streams', 'v:0', '-show_entries', 'stream=height',
         '-of', 'default=nw=1:nk=1', video_src],
        capture_output=True, text=True, timeout=300,
    )
    found = probe.stdout.strip().split('\n')[0]
    if found != str(height):
        raise RuntimeError(f'source is {found or "unreadable"}p, crop assumes {height}p')


def stream_start_time(video_src: str) -> float:
    """First frame's PTS. Ignoring it shifts every seek by start_time*fps."""
    probe = subprocess.run(
        ['ffprobe', '-v', 'error', *video_headers(video_src),
         '-select_streams', 'v:0', '-show_entries', 'stream=start_time',
         '-of', 'default=nw=1:nk=1', video_src],
        capture_output=True, text=True, timeout=300,
    )
    try:
        return float(probe.stdout.strip().split('\n')[0])
    except (ValueError, IndexError):
        return 0.0


def grab_segment(video_src: str, start_time: float, segment: dict,
                 crop: dict, dest: Path) -> int:
    """Fetch one cropped segment. Returns bytes written."""
    first, last = segment['start'], segment['end']
    count = last - first
    partial = dest.with_suffix('.part')

    result = subprocess.run(
        ['ffmpeg', '-v', 'error', '-y',
         *video_headers(video_src),
         '-ss', f'{start_time + first / FPS:.6f}',
         '-i', video_src,
         '-map', '0:v:0', '-frames:v', str(count),
         '-vf', f"crop={crop['w']}:{crop['h']}:{crop['x']}:{crop['y']}",
         '-an',
         '-vsync', 'passthrough',
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
    year, month, day = item['date'].split('-')
    return root / year / month / day / f'segment_{index}.mp4'


def key(item: dict, index: int) -> str:
    return f"{item['date']}/segment_{index}"


def pending_segments(item: dict, root: Path) -> list[dict]:
    """Segments still to fetch.

    A segment's 'index' is its original index from the source segmentation, not
    its position in this list. Gaps are expected: segments that never made it
    into the dataset are absent, and the surviving ones keep their old numbers.
    """
    return [
        s for s in item['segments']
        if not segment_path(item, s['index'], root).exists()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True, help='output directory')
    parser.add_argument('--workers', type=int, default=1,
                        help="days downloaded at the same time (default: 1). "
                             "These are a broadcaster's public servers: please keep "
                             "this low and raise it only if you need to")
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
                      f"[{segment['start']}, {segment['end']}) "
                      f"crop {item['crop']['w']}x{item['crop']['h']}"
                      f"+{item['crop']['x']}+{item['crop']['y']}")
                total += 1
        n = sum(len(i['segments']) for i in items)
        print(f'\n{total} segments to download, {n - total} already downloaded')
        return 0

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
        session = requests.Session()
        prefix = f"[{index}/{len(items)}] {item['date']}"
        try:
            video_src = item.get('video_src')
            try:
                if not video_src:
                    raise RuntimeError('no stored URL')
                check_source(video_src, item['source_height'])
            except Exception:
                # No stored URL, or it has expired: read the current one off the page.
                video_src = resolve_video_src(session, item['url'], item['source_height'])
                check_source(video_src, item['source_height'])
            start_time = stream_start_time(video_src)
        except Exception as exc:
            print(f'{prefix}  resolve failed: {exc}', flush=True)
            record({'date': item['date'], 'status': 'FAILURE', 'error': str(exc)})
            return

        for segment in todo:
            i = segment['index']
            dest = segment_path(item, i, args.root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                written = grab_segment(video_src, start_time, segment,
                                       item['crop'], dest)
                print(f'{prefix}  segment_{i}  {written / 1024 / 1024:.1f} MiB',
                      flush=True)
                record({'key': key(item, i), 'status': 'SUCCESS'})
            except Exception as exc:
                # A stale stored URL: re-resolve once, then give up on this segment.
                if item.get('video_src'):
                    try:
                        video_src = resolve_video_src(session, item['url'],
                                                      item['source_height'])
                        check_source(video_src, item['source_height'])
                        start_time = stream_start_time(video_src)
                        written = grab_segment(video_src, start_time,
                                               segment, item['crop'], dest)
                        print(f'{prefix}  segment_{i}  {written / 1024 / 1024:.1f} MiB'
                              f'  (refreshed)', flush=True)
                        record({'key': key(item, i), 'status': 'SUCCESS'})
                        continue
                    except Exception as exc2:
                        exc = exc2
                print(f'{prefix}  segment_{i} failed: {exc}', flush=True)
                record({'key': key(item, i), 'status': 'FAILURE', 'error': str(exc)})

    work = [(n, item) for n, item in enumerate(items, 1)
            if pending_segments(item, args.root)]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(download_item, work))

    failures = [r for r in results if r['status'] == 'FAILURE']
    print(f'\n{len(results) - len(failures)} segments downloaded, '
          f'{len(failures)} failed')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
