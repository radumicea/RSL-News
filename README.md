# RSL-News

A Romanian Sign Language translation dataset: interpreted news broadcasts from
three Romanian channels, cropped to the sign-language interpreter and aligned
sentence by sentence with the transcribed speech.

| Channel | Segments | Video | Labelled | Sentences |
|---|---|---|---|---|
| Digi24 | 2,802 | 516.2 h | 437.3 h | 219,768 |
| PrimaTV | 576 | 283.0 h | 233.8 h | 145,470 |
| ProTV | 928 | 415.0 h | 346.7 h | 199,347 |
| **Total** | **4,306** | **1,214.2 h** | **1,017.9 h** | **564,585** |

The footage is not redistributed; this repository rebuilds it from the public
archives.

## Reproduce

```bash
python3.10 -m venv .venv && .venv/bin/pip install -r requirements.txt
./download.sh --cookies cookies.txt          # the videos, into mp4s/  (~210 GB)
.venv/bin/python extract_poses.py            # optional: landmarks, into dataset/ (GPU)
.venv/bin/python extract_features.py         # optional: BSL-5k I3D features, into dataset/ (GPU)
```

Besides Python 3.10, this needs ffmpeg; PrimaTV also needs Node.js 20+, and git
and npm on the first run. PrimaTV comes from YouTube and needs a `cookies.txt`
exported from a logged-in YouTube session. The two extractions need an NVIDIA GPU
with a driver for CUDA 13; `extract_features.py --split val` does one split. All
three can be stopped at any time and run again to resume.

The splits are in the manifests: per channel, whole broadcasts drawn at random,
about 90/5/5 of the labelled hours for train/val/test.

The labels (and features) are on Hugging Face: _link to be added_. Unpack them
into `dataset/`; their paths mirror `mp4s/`.

## Layout

```
mp4s/<channel>/.../segment_<i>.mp4          the video (download.sh)
dataset/<channel>/.../segment_<i>.json      sentences, start/end in seconds from the
                                            segment's first frame (frame = t * 25)
dataset/<channel>/.../segment_<i>.pose.npz  body and hands, one entry per frame (extract_poses.py)
dataset/<channel>/.../segment_<i>.face.npz  the face mesh, same frames; read both with extract_poses.load
dataset/<channel>/.../segment_<i>.bsl5k.npy I3D features, float16 (N, 1024); feature k is
                                            frames [2k, 2k + 8) (extract_features.py; settings: FEATURES.md)
manifests/                                  source, crop, frame range [start, end) and split of every segment
scripts/                                    the downloaders
lib/mp_holistic_plus/                       MediaPipe-based landmarks (pose, face, hands), batched on CUDA
pipeline/                                   the code that originally built the dataset;
                                            not needed to reproduce it
docs/                                       thesis and slides
```
