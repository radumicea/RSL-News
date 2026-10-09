"""Code shared by the pipeline notebooks: where things are, reading and writing them, and the
steps that several notebooks or channels use."""
import json
import os
import re
from pathlib import Path

import zstandard

PIPELINE = Path(__file__).resolve().parent
REPO = PIPELINE.parent
FROZEN = PIPELINE / 'frozen'          # what the original run produced that cannot be redone identically
DECISIONS = PIPELINE / 'decisions'    # calls made by hand
WORK = Path(os.environ.get('RSL_NEWS_WORK', PIPELINE / 'work'))   # this run: downloads, audio, outputs
CHANNELS = ('PrimaTV', 'ProTV', 'Digi24')
FPS = 25


def read(path):
    """A .json or .json.zst file."""
    path = Path(path)
    data = path.read_bytes()
    if path.suffix == '.zst':
        data = zstandard.ZstdDecompressor().decompress(data)
    return json.loads(data)


def write(path, obj, indent=1, newline=True):
    """A .json or .json.zst file, written whole or not at all."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == '.zst':
        data = zstandard.ZstdCompressor(level=19).compress(
            json.dumps(obj, ensure_ascii=False, separators=(',', ':'), sort_keys=True).encode())
    else:
        data = (json.dumps(obj, indent=indent, ensure_ascii=False) + ('\n' if newline else '')).encode()
    tmp = path.with_name(path.name + '.part')
    tmp.write_bytes(data)
    tmp.replace(path)


def box(x, y, w, h):
    return {'x': x, 'y': y, 'w': w, 'h': h}


# The interpreter box of the released videos, per channel and source height (Digi24: per era).
# These are the boxes the original cuts actually have: ffmpeg's crop filter rounds an odd x, w or h
# down to even and moves a box that overflows the frame back inside it (PrimaTV, ProTV); Digi24's
# cuts were cropped by the h264_cuvid decoder, whose box was measured on the cuts.
CROP = {
    'PrimaTV': {720: box(1014, 446, 254, 274), 1080: box(1522, 670, 382, 410)},
    'ProTV': {720: box(952, 346, 300, 322), 432: box(570, 208, 180, 192)},
    'Digi24': {'until 2023-11': box(1019, 475, 192, 206), 'from 2023-12': box(968, 287, 248, 224)},
}


def digi24_era(date):
    """Digi24 moved the interpreter box in December 2023."""
    return 'from 2023-12' if date >= '2023-12' else 'until 2023-11'


def label_path(channel, key, index):
    """Where a segment's sentences go in the dataset: PrimaTV/<id>/segment_<i>.json,
    ProTV/<yyyy>/<mm>/<dd>/segment_<i>.json, Digi24/<yyyy>/<mm>/<dd>/segment_<i>.json."""
    folder = key if channel == 'PrimaTV' else key.replace('-', '/')
    return f'{channel}/{folder}/segment_{index}.json'


# ---------------------------------------------------------------- transcripts

THANKS = re.compile(r'mul[țt]um.+vizionare', re.IGNORECASE)   # "mulțumim pentru vizionare", Whisper's sign-off


def is_hallucination(text, min_repeats=3, dominance=0.5, max_phrase_len=64):
    """Whisper's repetition loops: a phrase of 1 to 64 words repeated 3 or more times in a row
    that makes up at least half of the sentence's words."""
    words = re.findall(r'\b\w+\b', text.lower())
    n = len(words)
    for size in range(1, max_phrase_len + 1):
        i = 0
        while i <= n - size * min_repeats:
            phrase, count = words[i:i + size], 1
            while words[i + count * size:i + (count + 1) * size] == phrase:
                count += 1
            if count >= min_repeats:
                if count * size / n >= dominance:
                    return True
                i += count * size
            else:
                i += 1
    return False


def keep_sentences(rows):
    """Whisper's aligned sentences [start, end, text, words] -> the ones kept, as
    {start, end, text}: at least 2 s long, 0.1 to 2 s per word, not a repetition loop, not the
    sign-off, with the old cedilla letters ş ţ replaced by ș ț."""
    out = []
    for start, end, text, words in rows:
        if not words:
            continue
        text = text.strip().replace('ţ', 'ț').replace('ş', 'ș').replace('Ţ', 'Ț').replace('Ş', 'Ș')
        duration = end - start
        if duration < 2 or not 0.1 <= duration / words <= 2 or is_hallucination(text):
            continue
        if text and not THANKS.search(text):
            out.append({'start': start, 'end': end, 'text': text})
    return out


def place(sentences, frames, clock=None):
    """Sentences timed on the transcribed audio -> times on the segment's video (frame 0 = 0 s),
    dropping what does not fall on the video: sentences that start before the first frame or at/after
    the last one, and ends more than 0.1 s past the last frame (up to 0.1 s are clipped to it).

    clock (the original run only, see 4_labels): its audio started `offset` s before the first frame,
    and audio holes changed that offset at `steps`; a sentence boundary inside a located hole of a
    frame or more is dropped, as its frame is not known."""
    v0, offset, steps, shift = 0.0, 0.0, [], 0.0
    if clock:
        v0, offset, steps, shift = clock['v0'], clock['offset'], clock['steps'], clock.get('shift', 0.0)
    lag0, duration, out = v0 - offset, frames / FPS, []

    def lag(t):
        value = lag0
        for _, to, _, after in steps:
            if t >= to:
                value = after
        return value

    for s in sentences:
        a, b = s['start'] + lag(s['start']) - v0, s['end'] + lag(s['end']) - v0
        if b <= 0 or a < 0 or a >= duration or b > duration + 0.1:
            continue
        if any(lo <= t <= hi and abs(after - before) >= 1 / FPS for lo, hi, before, after in steps
               for t in (s['start'], s['end'])):
            continue
        out.append({**s, 'start': round(a, 3), 'end': round(min(b, duration), 3)})
    if shift:
        out = [{**s, 'start': round(s['start'] + shift, 3), 'end': round(s['end'] + shift, 3)} for s in out]
    return out


def cut(sentences, first, frames):
    """Sentences of a segment whose first `first` frames were cut and which has `frames` frames left."""
    duration, out = frames / FPS, []
    for s in sentences:
        a, b = round(s['start'] - first / FPS, 3), round(s['end'] - first / FPS, 3)
        if b <= 0 or a < 0 or a >= duration or b > duration + 0.1:
            continue
        out.append({**s, 'start': a, 'end': round(min(b, duration), 3)})
    return out


# ---------------------------------------------------------------- video

class Video:
    """Random access to the frames of a video file (as OpenCV decodes it)."""

    def __init__(self, path):
        import cv2
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise IOError(f'cannot open {path}')
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __getitem__(self, index):
        """Frame `index`, RGB."""
        import cv2
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        if not ok:
            raise IOError(f'cannot read frame {index}')
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def read(self, batch=1024):
        """All frames from the first, BGR, in lists of `batch`."""
        while True:
            frames = []
            for _ in range(batch):
                ok, frame = self.cap.read()
                if not ok:
                    break
                frames.append(frame)
            if frames:
                yield frames
            if len(frames) < batch:
                return


def first_true(lo, hi, predicate):
    """The first index in [lo, hi] where predicate holds, for a predicate that is false up to some
    index and true from there on (hi if it never holds)."""
    found = hi
    while lo <= hi:
        mid = (lo + hi) // 2
        if predicate(mid):
            found, hi = mid, mid - 1
        else:
            lo = mid + 1
    return found


def probe(path, *entries, stream='v:0'):
    """ffprobe values of a file's stream, e.g. probe(f, 'start_time') -> {'start_time': '0.08'}."""
    import subprocess
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', stream, '-show_entries',
                          'stream=' + ','.join(entries), '-of', 'json', str(path)],
                         capture_output=True, text=True, check=True).stdout
    streams = json.loads(out)['streams']
    return streams[0] if streams else {}


def save_audio(video, start, end, dest):
    """The audio of frames [start, end) of `video`, as 16 kHz mono FLAC, on the video's clock:
    its first sample plays with frame `start`, gaps in the audio are filled with silence."""
    import subprocess
    first_frame = float(probe(video, 'start_time').get('start_time', 0))
    file_start = float(json.loads(subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=start_time', '-of', 'json', str(video)],
        capture_output=True, text=True, check=True).stdout)['format'].get('start_time', 0))
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + '.part.flac')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', f'{first_frame - file_start + start / FPS:.6f}',
                    '-i', str(video), '-t', f'{(end - start) / FPS:.6f}', '-map', '0:a:0',
                    '-af', 'aresample=async=1:first_pts=0', '-ac', '1', '-ar', '16000', str(part)], check=True)
    part.replace(dest)


# ---------------------------------------------------------------- recognizing the interpreter

KNOWN_PEOPLE = PIPELINE / 'known_people'


class Faces:
    """Face recognition against reference pictures of the interpreters (known_people/<name>.png)."""

    def __init__(self, names):
        import face_recognition
        self.names = list(names)
        self.known = [face_recognition.face_encodings(face_recognition.load_image_file(KNOWN_PEOPLE / f'{n}.png'))[0]
                      for n in self.names]

    def who(self, image):
        """The first interpreter recognized in an RGB image, or None."""
        import face_recognition
        import numpy as np
        for encoding in face_recognition.face_encodings(np.ascontiguousarray(image[:, :, :3], dtype=np.uint8)):
            matches = face_recognition.compare_faces(self.known, encoding)
            if any(matches):
                return self.names[int(np.argmax(matches))]
        return None


class Classifier:
    """MobileNetV3-Small with a two-class head (frozen/models/*.pt): is the interpreter in this crop?
    Crops smaller than `pad_to` (width, height) are padded with black around them."""

    def __init__(self, weights, pad_to=None):
        import torch
        from torchvision import models, transforms as T
        from torchvision.transforms import functional as F
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = models.mobilenet_v3_small()
        self.model.classifier[3] = torch.nn.Linear(self.model.classifier[3].in_features, 2)
        self.model.load_state_dict(torch.load(weights, map_location=self.device))
        self.model.eval().to(self.device)

        def pad(img):
            w, h = img.size
            left, top = (pad_to[0] - w) // 2, (pad_to[1] - h) // 2
            return F.pad(img, padding=[left, top, pad_to[0] - w - left, pad_to[1] - h - top], fill=0)
        steps = [T.Lambda(pad)] if pad_to else []
        self.transform = T.Compose(steps + [T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def __call__(self, crops_bgr):
        """One bool per crop."""
        import torch
        from PIL import Image
        with torch.inference_mode():
            batch = torch.stack([self.transform(Image.fromarray(c[..., ::-1])) for c in crops_bgr]).to(self.device)
            return (torch.softmax(self.model(batch), dim=1)[:, 1] >= 0.5).cpu().tolist()
