#!/usr/bin/env python3
"""Download the interpreter segments of the ProTV morning-news videos.

For every entry in manifests/protv_manifest.json this fetches only what is
needed: each listed segment, cropped to the interpreter box, video stream only.
The source broadcasts are ~2h. Each entry names the rendition it is cropped
from (source_height: 720 for 300x322 clips, 432 for 180x192 clips); the crop is
only valid for that resolution, so an episode without it is reported as failed
rather than cropped from another one.

Produces:

    <root>/<yyyy>/<mm>/<dd>/segment_<i>.mp4

Each manifest entry points at a protv.ro episode page. The playable stream is
behind a player embed and is signed with a short expiry, so it is resolved at
download time rather than stored in the manifest.

Segment bounds are frame indices into the source video. They are converted with
the stream's own start_time taken into account -- these streams start at PTS
0.08, so a bare frame/fps timestamp lands two frames early.

By default downloads run one at a time; --workers N fetches N entries at once.
Each segment is written to a .part file and renamed to .mp4 only once it is
complete; segments whose .mp4 already exists are skipped, so the script can be
stopped and re-run. The manifest is only read.

Requires ffmpeg/ffprobe on PATH.

Usage:
    python download_protv.py --root /path/to/output --dry-run
    python download_protv.py --root /path/to/output [--workers N]
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
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / 'manifests' / 'protv_manifest.json'
LOG = ROOT / 'logs' / 'protv_download_log.json'

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:140.0) Gecko/20100101 Firefox/140.0',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'ro-RO,ro;q=0.9,en;q=0.5',
}
SOURCE = re.compile(r'\{"src":"([^"]+)","type":"([^"]+)"\}')
FPS = 25.0
CRF = '18'


def resolve_manifest_url(session: requests.Session, episode_url: str) -> str:
    """Follow an episode page through its player embed to the DASH manifest."""
    page = session.get(episode_url, headers=HEADERS, timeout=30)
    page.raise_for_status()

    iframe = BeautifulSoup(page.text, 'html.parser').find('iframe', {'data-src': True})
    if not iframe:
        raise RuntimeError(f'no player iframe in {episode_url}')

    embed = session.get(iframe['data-src'], headers=HEADERS, timeout=30)
    embed.raise_for_status()

    sources = SOURCE.findall(embed.text)
    for url, kind in sources:
        if 'dash' in kind:
            return url
    for url, kind in sources:
        if 'mpegurl' in kind:
            return url
    raise RuntimeError(f'no playable source for {episode_url}')


def pick_video_stream(manifest_url: str, height: int) -> int:
    """Ordinal of the video stream of the given height.

    Selection is by the height ffprobe reports. yt-dlp's format IDs do not map
    reliably onto these streams, so they are not used.
    """
    probe = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v',
         '-show_entries', 'stream=height', '-of', 'json', manifest_url],
        capture_output=True, text=True, timeout=180,
    )
    if probe.returncode != 0:
        raise RuntimeError(f'ffprobe failed: {probe.stderr.strip()[:200]}')

    heights = [s.get('height') or 0 for s in json.loads(probe.stdout)['streams']]
    if not heights:
        raise RuntimeError('no video streams in manifest')

    # The crop coordinates are only valid for this rendition.
    if height not in heights:
        raise RuntimeError(f'no {height}p stream (have {sorted(set(heights))})')
    return heights.index(height)


def stream_start_time(manifest_url: str, stream: int) -> float:
    """First frame's PTS. Ignoring it shifts every seek by start_time*fps."""
    probe = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', f'v:{stream}',
         '-show_entries', 'stream=start_time', '-of', 'default=nw=1:nk=1',
         manifest_url],
        capture_output=True, text=True, timeout=180,
    )
    try:
        return float(probe.stdout.strip().split('\n')[0])
    except (ValueError, IndexError):
        return 0.0


def grab_segment(manifest_url: str, stream: int, start_time: float,
                 segment: dict, crop: dict, dest: Path) -> int:
    """Fetch one cropped segment. Returns bytes written."""
    first, last = segment['start'], segment['end']
    count = last - first + 1
    partial = dest.with_suffix('.part')

    result = subprocess.run(
        ['ffmpeg', '-v', 'error', '-y',
         '-ss', f'{start_time + first / FPS:.6f}',
         '-i', manifest_url,
         '-map', f'0:v:{stream}', '-frames:v', str(count),
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
    year, month, day = item['date'].split('-')
    return root / year / month / day / f'segment_{index}.mp4'


def key(item: dict, index: int) -> str:
    return f"{item['date']}/segment_{index}"


def pending_segments(item: dict, root: Path) -> list[dict]:
    """Segments still to fetch. A segment's 'index' is its original index from
    the source segmentation, which is what the output file is named after."""
    return [
        s for s in item['segments']
        if not segment_path(item, s['index'], root).exists()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True, help='output directory')
    parser.add_argument('--workers', type=int, default=1,
                        help="entries downloaded at the same time (default: 1). "
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
                      f"[{segment['start']}-{segment['end']}] "
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
        prefix = f"[{index}/{len(items)}] {item['date']}"
        try:
            manifest_url = resolve_manifest_url(requests.Session(), item['url'])
            stream = pick_video_stream(manifest_url, item['source_height'])
            start_time = stream_start_time(manifest_url, stream)
        except Exception as exc:
            print(f'{prefix}  resolve failed: {exc}', flush=True)
            record({'date': item['date'], 'status': 'FAILURE', 'error': str(exc)})
            return

        for segment in todo:
            i = segment['index']
            dest = segment_path(item, i, args.root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                written = grab_segment(manifest_url, stream, start_time,
                                       segment, item['crop'], dest)
                print(f'{prefix}  segment_{i}  {written / 1024 / 1024:.1f} MiB',
                      flush=True)
                record({'key': key(item, i), 'status': 'SUCCESS'})
            except Exception as exc:
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

