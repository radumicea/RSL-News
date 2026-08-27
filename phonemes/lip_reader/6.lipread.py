# compute and cache mouth features after proj_encoder with halo padding so we can process chunks without loading full video in memory,
# but still get mathematically identical features

import gc
import glob
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from auto_avsr.resnet import video_resnet
from auto_avsr.video_process import VideoProcess
from tqdm import tqdm

SCRAPED_DIRS = ['../digi/scraped', '../prima/scraped', '../protv/scraped']
FACE_LM = (23, 91)  # COCO-Wholebody face landmark range
MIN_FACE_CONF = 0.5
LETTERBOX_TARGET = (256, 192)
CKPT = Path('/home/radumicea/Desktop/PV-ASR/vsr_trlrs2lrs3vox2avsp_base.pth')
FPS = 25
CORE, HALO = 225, 2
DIM = 768
BATCH = 16  # chunks per forward pass
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ATOL = 1e-3 if DEVICE.type == 'cuda' else 1e-6
RTOL = 1e-3 if DEVICE.type == 'cuda' else 1e-5
print(f'Device: {DEVICE}')

pairs = []
for d in SCRAPED_DIRS:
    for p in glob.glob(os.path.join(d, '**', '*.pose.npy'), recursive=True):
        base = p.replace('.pose.npy', '')
        for ext in ('.mp4', '.mkv'):
            if os.path.isfile(base + ext):
                pairs.append((base + ext, p))
                break
print(f'{len(pairs)} pairs')


def undo_letterbox(coords, h, w):
    th, tw = LETTERBOX_TARGET
    s = min(th / h, tw / w)
    x0, y0 = (tw - int(w * s)) // 2, (th - int(h * s)) // 2
    out = coords.copy()
    out[..., 0] = (out[..., 0] - x0) / s
    out[..., 1] = (out[..., 1] - y0) / s
    return out


def extract_mouth(vid_path, pose_path):
    cap = cv2.VideoCapture(str(vid_path))
    W, H = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(5) or FPS

    pose = np.load(pose_path).astype(np.float32)
    n_cap = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or len(pose)
    n = min(n_cap, len(pose))
    pose = pose[:n]
    pose[:, :, :2] = undo_letterbox(pose[:, :, :2], H, W)
    face = pose[:, FACE_LM[0] : FACE_LM[1]]
    lm = [
        face[i, :, :2] if face[i, :, 2].mean() >= MIN_FACE_CONF else None
        for i in range(n)
    ]
    del pose, face

    # Stream frames one at a time — never hold entire video in RAM
    def frames():
        for _ in range(n):
            ret, frame = cap.read()
            if not ret:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    vp = VideoProcess(convert_gray=True)
    mouth = vp(frames(), lm)
    cap.release()
    del lm
    if mouth is None or len(mouth) == 0:
        raise RuntimeError(f'Mouth crop failed: {vid_path}')

    bad = getattr(vp, 'bad_frames', [])
    if bad:
        # compute runs of consecutive bad frames
        runs = []
        run_start = bad[0]
        for i in range(1, len(bad)):
            if bad[i] != bad[i - 1] + 1:
                runs.append((run_start, bad[i - 1]))
                run_start = bad[i]
        runs.append((run_start, bad[-1]))
        longest = max(e - s + 1 for s, e in runs)
        print(
            f'  {vid_path}: {len(bad)}/{len(mouth)} bad frames, '
            f'{len(runs)} run(s), longest run={longest}'
        )

    return mouth, fps


def preprocess(mouth):
    x = torch.from_numpy(mouth).float() / 255.0
    x = x[:, 4:92, 4:92].unsqueeze(1)  # center-crop 88, add channel
    return ((x - 0.421) / 0.165).contiguous()


from queue import Queue
from threading import Thread

_model = None


def get_model():
    global _model
    if _model is not None:
        return _model
    st = torch.load(CKPT, map_location='cpu', weights_only=False)
    fe = video_resnet()
    fe.load_state_dict(
        {k[9:]: v for k, v in st.items() if k.startswith('frontend.')}, strict=True
    )
    proj = torch.nn.Linear(512, 768)
    proj.load_state_dict(
        {k[13:]: v for k, v in st.items() if k.startswith('proj_encoder.')}, strict=True
    )
    _model = torch.nn.Sequential(fe, proj).to(DEVICE).eval()
    return _model


def _run_single(model, x, start, end):
    """Forward one chunk with halo, return core slice [end-start, DIM]."""
    s, e = max(0, start - HALO), min(len(x), end + HALO)
    trim = start - s
    with (
        torch.inference_mode(),
        torch.amp.autocast(DEVICE.type, enabled=DEVICE.type == 'cuda'),
    ):
        out = model(x[s:e].unsqueeze(0).to(DEVICE))
    return out[0, trim : trim + end - start].float().cpu()


def _run_batched(model, x, chunk_specs):
    """Forward a batch of chunks. Returns list of core tensors."""
    full_len = CORE + 2 * HALO
    padded, trims, keeps = [], [], []
    for start, end in chunk_specs:
        s, e = max(0, start - HALO), min(len(x), end + HALO)
        chunk = x[s:e]
        pad = full_len - chunk.shape[0]
        if pad > 0:
            chunk = torch.nn.functional.pad(chunk, (0, 0, 0, 0, 0, 0, 0, pad))
        padded.append(chunk)
        trims.append(start - s)
        keeps.append(end - start)
    batch = torch.stack(padded).to(DEVICE)
    with (
        torch.inference_mode(),
        torch.amp.autocast(DEVICE.type, enabled=DEVICE.type == 'cuda'),
    ):
        out = model(batch)
    return [
        out[i, trims[i] : trims[i] + keeps[i]].float().cpu()
        for i in range(len(chunk_specs))
    ]


def cache_for(vid):
    return Path(vid).with_suffix('.mouth.npy')


def _gpu_write(vid, x):
    """Run model in batched chunks and write memmap."""
    out = cache_for(vid)
    T = len(x)
    model = get_model()
    feat = np.lib.format.open_memmap(
        str(out), mode='w+', dtype=np.float16, shape=(T, DIM)
    )
    specs = [(s, min(s + CORE, T)) for s in range(0, T, CORE)]
    for bi in range(0, len(specs), BATCH):
        batch_specs = specs[bi : bi + BATCH]
        results = _run_batched(model, x, batch_specs)
        for (s, e), r in zip(batch_specs, results):
            feat[s:e] = r.numpy().astype(np.float16)
    feat.flush()
    del x  # free preprocessed tensor immediately
    return out


def build_cache(vid, pose, overwrite=False):
    out = cache_for(vid)
    if out.exists() and not overwrite:
        return out
    mouth, _ = extract_mouth(vid, pose)
    return _gpu_write(vid, preprocess(mouth))


def build_all(pairs, prefetch=2):
    """Producer-consumer: 1 thread does CPU extraction, main thread does GPU.
    Queue holds at most `prefetch` items to bound RAM."""
    get_model()
    q = Queue(maxsize=prefetch)
    skipped = []

    def producer():
        for vid, pose in pairs:
            if cache_for(vid).exists():
                q.put((vid, None))  # signal skip
                continue
            try:
                mouth, _ = extract_mouth(vid, pose)
                x = preprocess(mouth)
                del mouth
                gc.collect()
                q.put((vid, x))
            except (Exception, MemoryError) as ex:
                gc.collect()
                q.put((vid, ex))
        q.put(None)  # sentinel

    t = Thread(target=producer, daemon=True)
    t.start()

    pbar = tqdm(total=len(pairs))
    while True:
        item = q.get()
        if item is None:
            break
        vid, result = item
        pbar.update(1)
        if result is None:
            continue
        if isinstance(result, (Exception, MemoryError)):
            skipped.append(vid)
            pbar.write(f'SKIP {vid}: {result}')
            continue
        _gpu_write(vid, result)
        gc.collect()
        torch.cuda.empty_cache()
    t.join()
    pbar.close()
    if skipped:
        print(f'{len(skipped)} videos skipped')


def get_features(vid, start_sec, end_sec, fps=FPS):
    mm = np.load(cache_for(vid), mmap_mode='r')
    s = max(0, min(round(start_sec * fps), len(mm)))
    e = max(s, min(round(end_sec * fps), len(mm)))
    return torch.from_numpy(np.array(mm[s:e], dtype=np.float32))


def validate(vid, pose, seed=0):
    build_cache(vid, pose)
    mm = np.load(cache_for(vid), mmap_mode='r')
    T = len(mm)
    rng = random.Random(seed)
    seg = min(T, max(1, rng.randint(1, min(30, max(1, T // FPS))) * FPS))
    s = rng.randint(0, T - seg)
    cached = torch.from_numpy(np.array(mm[s : s + seg], dtype=np.float32))
    x = preprocess(extract_mouth(vid, pose)[0])
    direct = _run_single(get_model(), x, s, s + seg)
    diff = (direct - cached).abs().max().item()
    ok = torch.allclose(direct, cached, atol=ATOL, rtol=RTOL)
    print(f'[{s}:{s + seg}] max_diff={diff:.6f} allclose={ok}')
    assert ok


# Validate on a sample
v, p = pairs[0]
build_cache(v, p, overwrite=True)
validate(v, p, seed=42)

# Save middle mouth frame for first video in each source
for d in SCRAPED_DIRS:
    for vid, pose in pairs:
        if vid.startswith(d):
            mouth, _ = extract_mouth(vid, pose)
            name = f'{Path(d).parent.name}_mouth.png'
            cv2.imwrite(name, mouth[len(mouth) // 2])
            print(f'Saved {name} from {vid}')
            break

# Build all caches
build_all(pairs)
