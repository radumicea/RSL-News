"""Code shared by the pipeline notebooks: where things are, reading and writing them, and the
steps that several notebooks or channels use."""
import json
import os
import re
from pathlib import Path

import zstandard

PIPELINE = Path(__file__).resolve().parent
REPO = PIPELINE.parent
FROZEN = PIPELINE / 'frozen'          # the automatic steps' outputs, which cannot be redone identically
DECISIONS = PIPELINE / 'decisions.json'   # calls made by hand
WORK = Path(os.environ.get('RSL_NEWS_WORK', PIPELINE / 'work'))   # this run: downloads, audio, outputs
CHANNELS = ('PrimaTV', 'ProTV', 'Digi24')
FPS = 25

# 'replay': rebuild the release from frozen/ and decisions.json; 'update': add the broadcasts aired since;
# 'live': make everything anew (see README). Set here or with RSL_NEWS_MODE.
MODE = os.environ.get('RSL_NEWS_MODE', 'replay')
assert MODE in ('replay', 'update', 'live'), MODE


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


def found():
    """Per channel and broadcast, what 2_segment found (frozen/segments.json): in an update with this run's
    work/segments.json over it (a broadcast done again replaces what was there), in a live run this run's alone."""
    out = {c: {} for c in CHANNELS}
    paths = ([] if MODE == 'live' else [FROZEN / 'segments.json']) + ([] if MODE == 'replay' else [WORK / 'segments.json'])
    for path in paths:
        for c, made in read(path).items():
            out[c].update(made)
    return out


def last_day(channel):
    """The last day of ProTV or Digi24 the release covers; an update takes the days after it. Days without an
    interpreter are not all in segments.json, so Digi24's list counts too."""
    days = [key.split('/')[0] for key in read(FROZEN / 'segments.json')[channel]]
    return max(days + list(read(FROZEN / 'broadcasts.json').get(channel, {}).get('listing', {})))


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
# Whisper's other made-up sign-offs, of YouTube videos: "nu uitați să dați like…", "nu uitați să vă abonați la canal…"
YOUTUBE = re.compile(r'nu uita[țt]i s[ăa] (da[țt]i like|v[ăa] abona[țt]i la canal)', re.IGNORECASE)


def other_script(text):
    """Whether a sentence has letters of a script other than Latin: Whisper half-transcribing speech in another
    language (Russian, Ukrainian) into its Romanian."""
    import unicodedata
    return any(ch.isalpha() and not unicodedata.name(ch, '').startswith('LATIN') for ch in text)


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
    """Random access to the frames of a video file, as OpenCV decodes it; AV1, which OpenCV's
    FFmpeg cannot decode (YouTube's newer PrimaTV streams), as PyAV decodes it."""

    def __init__(self, path):
        import cv2
        self.av1 = probe(path, 'codec_name').get('codec_name') == 'av1'
        if self.av1:
            import av
            self.container = av.open(str(path))
            self.stream = self.container.streams.video[0]
            self.start = float(self.stream.start_time * self.stream.time_base) if self.stream.start_time else 0.0
            self.frames = self.stream.frames or round(float(self.stream.duration * self.stream.time_base) * FPS)
            self.height = self.stream.height
            return
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise IOError(f'cannot open {path}')
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def _index(self, frame):
        return round((float(frame.pts * frame.time_base) - self.start) * FPS)

    def __getitem__(self, index):
        """Frame `index`, RGB."""
        import cv2
        if self.av1:
            self.container.seek(int((self.start + index / FPS) / self.stream.time_base), stream=self.stream)
            for frame in self.container.decode(self.stream):
                if self._index(frame) >= index:
                    if self._index(frame) == index:
                        return frame.to_ndarray(format='rgb24')
                    break
            raise IOError(f'cannot read frame {index}')
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        if not ok:
            raise IOError(f'cannot read frame {index}')
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def at(self, indices):
        """Frames `indices` (ascending), RGB, as self[i] gives them, read in order: seeking only across long gaps."""
        import cv2
        out, position = [], None
        for index in indices:
            if self.av1 or position is None or not 0 <= index - position <= 250:
                out.append(self[index])
                position = index + 1
                continue
            while position < index:
                self.cap.grab()
                position += 1
            ok, frame = self.cap.read()
            if not ok:
                raise IOError(f'cannot read frame {index}')
            out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            position += 1
        return out

    def read(self, batch=1024):
        """All frames from the first, BGR, in lists of `batch`."""
        if self.av1:
            self.container.seek(0, stream=self.stream)
            decoded = (f.to_ndarray(format='bgr24') for f in self.container.decode(self.stream))
        while True:
            frames = []
            for _ in range(batch):
                if self.av1:
                    frame = next(decoded, None)
                    ok = frame is not None
                else:
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


def save_audio(video, start, end, dest, audio=None):
    """The audio of frames [start, end) of `video` (or of the `audio` file downloaded with it), as
    16 kHz mono FLAC, on the video's clock: its first sample plays with frame `start`, gaps in the
    audio are filled with silence."""
    import subprocess
    source = audio or video
    first_frame = float(probe(video, 'start_time').get('start_time', 0))
    file_start = float(json.loads(subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=start_time', '-of', 'json', str(source)],
        capture_output=True, text=True, check=True).stdout)['format'].get('start_time', 0))
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + '.part.flac')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', f'{first_frame - file_start + start / FPS:.6f}',
                    '-i', str(source), '-t', f'{(end - start) / FPS:.6f}', '-map', '0:a:0',
                    '-af', 'aresample=async=1:first_pts=0', '-ac', '1', '-ar', '16000', str(part)], check=True)
    part.replace(dest)


# ---------------------------------------------------------------- recognizing the interpreter

KNOWN_PEOPLE = PIPELINE / 'known_people'
# the interpreters of each channel whose face is searched for (known_people/<name>.png)
INTERPRETERS = {'PrimaTV': ['I3', 'I4', 'I5', 'I6', 'I7'], 'ProTV': ['I1', 'I2']}   # Digi24's (I6, I7, I8): IdentitySearch


class Faces:
    """Face recognition against reference pictures of the interpreters (known_people/<name>.png)."""

    def __init__(self, names):
        import face_recognition
        self.names = list(names)
        self.known = [face_recognition.face_encodings(face_recognition.load_image_file(KNOWN_PEOPLE / f'{n}.png'))[0]
                      for n in self.names]

    def who(self, image):
        """The interpreter nearest to a face in an RGB image, if at distance 0.6 or less (dlib's own threshold), or
        None."""
        import face_recognition
        import numpy as np
        best, name = 0.6, None
        for encoding in face_recognition.face_encodings(np.ascontiguousarray(image[:, :, :3], dtype=np.uint8)):
            distances = face_recognition.face_distance(self.known, encoding)
            k = int(distances.argmin())
            if distances[k] <= best:
                best, name = float(distances[k]), self.names[k]
        return name


_search = None


def _search_init(path, names, box):
    import cv2
    global _search
    cv2.setNumThreads(1)
    _search = (Video(path), Faces(names), box)


def _search_who(indices):
    video, faces, (x, y, w, h) = _search
    return [faces.who(frame[y:y + h, x:x + w]) for frame in video.at(indices)]


class FaceSearch:
    """Face recognition of a channel's interpreters in a box (x, y, w, h) of a video's frames: who(i) for frame i,
    many(indices) for many frames at once, on `processes` CPUs (all by default)."""

    def __init__(self, path, names, box, processes=None):
        from multiprocessing import get_context
        processes = processes or os.cpu_count()
        self.pool = get_context('fork').Pool(processes, _search_init, (str(path), list(names), box)) if processes > 1 else None
        self.local = (Video(path), Faces(names), box)

    def who(self, index):
        video, faces, (x, y, w, h) = self.local
        return faces.who(video[index][y:y + h, x:x + w])

    def many(self, indices):
        if not self.pool:
            video, faces, (x, y, w, h) = self.local
            return [faces.who(frame[y:y + h, x:x + w]) for frame in video.at(indices)]
        chunks = [indices[i:i + 16] for i in range(0, len(indices), 16)]
        return [n for chunk in self.pool.map(_search_who, chunks) for n in chunk]

    def close(self):
        if self.pool:
            self.pool.terminate()


_facenet = None


def facenet_models():
    """MTCNN and InceptionResnetV1 (trained on VGGFace2), loaded once, and their device."""
    global _facenet
    if _facenet is None:
        import torch
        from facenet_pytorch import MTCNN, InceptionResnetV1
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _facenet = (MTCNN(image_size=160, margin=14, min_face_size=20, device=device, select_largest=True),
                    InceptionResnetV1(pretrained='vggface2').eval().to(device), device)
    return _facenet


def facenet_embeddings(images):
    """Per RGB image of one size: its face's FaceNet embedding, or None if no face is found."""
    import torch
    mtcnn, facenet, device = facenet_models()
    out = []
    for j in range(0, len(images), 128):
        chunk = images[j:j + 128]
        boxes, _ = mtcnn.detect(chunk)
        faces = [None if b is None else mtcnn.extract(im, b[:1], None) for im, b in zip(chunk, boxes)]
        faces = [None if f is None else (f[0] if f.dim() == 4 else f) for f in faces]
        found = [f for f in faces if f is not None]
        with torch.inference_mode():
            emb = iter(facenet(torch.stack(found).to(device)).cpu() if found else [])
        out += [None if f is None else next(emb) for f in faces]
    return out


def dlib_embedding(image):
    """The dlib encoding of the biggest face in an RGB image, or None."""
    import face_recognition
    boxes = face_recognition.face_locations(image, 1)
    if not boxes:
        return None
    biggest = max(boxes, key=lambda b: (b[2] - b[0]) * (b[1] - b[3]))
    return face_recognition.face_encodings(image, [biggest])[0]


def identity_reference(frames):
    """{interpreter: [[RGB frames of one reference segment], ...]} -> each model's reference embeddings, by interpreter."""
    import numpy as np
    reference = {'facenet': {}, 'dlib': {}}
    for interpreter, segments in frames.items():
        for segment in segments:                    # one segment at a time: MTCNN batches frames of one size
            reference['facenet'].setdefault(interpreter, []).extend(e for e in facenet_embeddings(segment) if e is not None)
            reference['dlib'].setdefault(interpreter, []).extend(e for e in map(dlib_embedding, segment) if e is not None)
    return {m: {n: np.stack([np.asarray(e) for e in es]) for n, es in r.items()} for m, r in reference.items()}


class IdentitySearch:
    """Digi24's face check of a box (x, y, w, h) of a video's frames, against reference frames of each interpreter
    (`identity_reference`): its faces are too small and blurred for dlib against one picture. FaceNet gives the distance
    to an interpreter as the mean of the 5 nearest reference frames; a frame is a hit if its nearest interpreter is
    `expected`, at distance 1.0 or less and 0.05 closer than the next; one that is not is checked again with dlib
    (0.55 or less, 0.03 closer). who(i), many(indices): `expected` on a hit, else None, as FaceSearch's;
    nearest(indices): the nearest interpreter of each frame with a face, or None."""

    def __init__(self, path, box, reference, expected=None):
        self.video, self.box, self.reference, self.expected, self.facenet = Video(path), box, reference, expected, {}

    def _distances(self, model, embedding):
        import numpy as np
        if embedding is None:
            return None
        return {n: float(np.sort(np.linalg.norm(r - np.asarray(embedding), axis=1))[:5].mean())
                for n, r in self.reference[model].items()}

    def _hit(self, d, threshold, margin):
        if d is None or min(d, key=d.get) != self.expected or d[self.expected] > threshold:
            return False
        return min(v for n, v in d.items() if n != self.expected) - d[self.expected] >= margin

    def _crop(self, frame):
        import numpy as np
        x, y, w, h = self.box
        return np.ascontiguousarray(frame[y:y + h, x:x + w])

    def _embed(self, indices):
        todo = sorted(set(i for i in indices if i not in self.facenet))
        for j in range(0, len(todo), 128):
            chunk = todo[j:j + 128]
            for i, e in zip(chunk, facenet_embeddings([self._crop(f) for f in self.video.at(chunk)])):
                self.facenet[i] = self._distances('facenet', e)

    def nearest(self, indices):
        self._embed(indices)
        return [min(d, key=d.get) if d else None for d in (self.facenet[i] for i in indices)]

    def many(self, indices):
        self._embed(indices)
        return [self.expected if self._hit(self.facenet[i], 1.0, 0.05) or
                self._hit(self._distances('dlib', dlib_embedding(self._crop(self.video[i]))), 0.55, 0.03) else None
                for i in indices]

    def who(self, index):
        return self.many([index])[0]

    def close(self):
        pass


def find_blocks(search, frames, start=0, stride=30 * FPS, shortest=10 * FPS, gap=2 * FPS):
    """The ranges of frames [start, frames) that show a known interpreter (`search`: a FaceSearch): a frame every
    `stride` frames is looked at; each run of them that show one is a range, whose first and last frame a binary search
    finds between the frames around it, refined frame by frame. A frame counts as showing an interpreter if it or the
    frames 0.5 and 1 s after it do. Ranges less than `gap` frames apart are joined, ranges under `shortest` dropped.
    -> [(first, last, interpreters)], last inclusive; interpreters: a Counter of the frames looked at that show each."""
    from collections import Counter
    who = search.who

    def seen(i):
        return next((n for n in (who(min(i + d, frames - 1)) for d in (0, FPS // 2, FPS)) if n), None)

    probes = list(range(start, frames, stride))
    hits = search.many(probes)
    for d in (FPS // 2, FPS):                       # frames that miss: the frames 0.5 and 1 s after them
        missed = [k for k, n in enumerate(hits) if n is None]
        for k, n in zip(missed, search.many([min(probes[k] + d, frames - 1) for k in missed])):
            hits[k] = n
    ranges, k = [], 0
    while k < len(probes):
        if not hits[k]:
            k += 1
            continue
        j = k
        while j + 1 < len(probes) and hits[j + 1]:
            j += 1
        first = first_true(probes[k - 1] + 1 if k else start, probes[k], lambda x: seen(x) is not None)
        first = next((f for f in range(first, min(first + FPS, frames - 1) + 1) if who(f)), first)
        hi = probes[j + 1] - 1 if j + 1 < len(probes) else frames - 1
        gone = first_true(probes[j], hi, lambda x: seen(x) is None)
        if seen(gone) is not None:                  # never gone up to hi
            last = hi
        else:
            last = next((f for f in range(min(gone + FPS, frames - 1), max(gone - FPS, first) - 1, -1) if who(f)), gone - 1)
        names = Counter(n for n in hits[k:j + 1] if n)
        if ranges and first - ranges[-1][1] - 1 < gap:  # a short miss, not a gap
            ranges[-1][1], ranges[-1][2] = last, ranges[-1][2] + names
        else:
            ranges.append([first, last, names])
        k = j + 1
    return [(first, last, names) for first, last, names in ranges if last + 1 - first >= shortest]


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
