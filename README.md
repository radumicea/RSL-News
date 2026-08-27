# RSL-News

Romanian Sign Language had no continuous, sentence-level corpus. Recognition of isolated signs, yes; translation, nothing. This repository is the pipeline that built one: **1,222 hours** of interpreted news broadcasts, spatially cropped to the interpreter, temporally segmented down to the parts where somebody is actually signing, and aligned sentence by sentence with automatically transcribed Romanian speech. No glosses at any point.

The data comes from a legal accident. Romanian Audiovisual Law No. 504/2002, Art. 42¹ obliges national broadcasters to interpret at least 30 minutes of news or debate programs into RSL every single day. Six years of that, across three channels, sits in public archives. RSL-News is what happens when you mine it.

On top of the corpus we trained the first RSL translation baselines, reaching **11.16 BLEU-4** on the held-out test split with a 6.6M-parameter transformer on a single consumer GPU — the highest reported score on any broadcast-domain sign language dataset so far.

📊 [Presentation slides](docs/RSL-News_presentation.pdf) &nbsp;·&nbsp; 📄 [Full documentation](docs/RSL-News_documentation.pdf) (the master's work, all the detail)

---

## What lives here

This tree is the **data side** of the project, end to end:

- scraping the three broadcast archives,
- locating and cropping the interpreter box,
- finding the frames where interpretation is actually happening,
- transcribing, aligning, filtering and tokenizing the speech,
- extracting the visual feature streams (I3D sign features, phoneme logits from the interpreter's lips).

Model code — our patched `SignFormer` fork, the décalage machinery, the fusion module — is not part of this repository. What it does and why is summarized [below](#the-model-side-summary) and written up properly in the documentation.

## The dataset

| Channel | Program | Source | Episodes | Segments | Sentences | Duration | Vocab |
|---|---|---|---|---|---|---|---|
| Digi24 | 11:00 News Journal | website archive | 1,200 | 2,813 | 222,082 | 459h 02m | 92,965 |
| ProTV | Morning News | website archive | 1,117 | 1,241 | 281,060 | 513h 51m | 114,893 |
| PrimaTV | FOCUS | YouTube | 578 | 578 | 145,292 | 249h 45m | 82,505 |
| **Total** | | | **2,895** | **4,632** | **648,434** | **1222h 39m** | **180,892** |

11.16M words, 6.79s and 17.2 words per sentence on average, and a 51.7% singleton rate that is mostly Romanian morphology rather than genuinely rare words (the 180,892 surface forms collapse to 111,974 lemmas). Coverage runs November 2019 to November 2025.

Nothing is redistributed here. The repo holds code; the footage stays where it came from.

## Pipeline

Notebooks are numbered and meant to run in order, per channel. Everything is idempotent — re-running skips whatever is already on disk, which matters when a stage takes days.

### Per-channel collection: [`digi/`](digi/), [`protv/`](protv/), [`prima/`](prima/)

**Download.** Each archive fights back differently. ProTV encodes the episode date in the page title, so URLs are reconstructed from it, with a binary search over a ±2-year window when the title date and the upload date disagree ([`protv/1.scrape.ipynb`](protv/1.scrape.ipynb), plus [`1.1.scrape_early.ipynb`](protv/1.1.scrape_early.ipynb) for the older layout). PrimaTV comes off YouTube playlists via `yt-dlp`, filtered by title and verified by eyeballing thumbnails ([`prima/1.scrape.ipynb`](prima/1.scrape.ipynb)). Digi24's player is JavaScript-only and had to be driven with Selenium to get at the source URL; that scraper isn't in this tree, which is why `digi/` starts at step 2.

**Interpreter ROI.** The interpreter always sits in a fixed rectangle in the bottom-right corner, but the rectangle moved whenever a channel refreshed its graphics, and resolutions vary per episode. Crop coordinates are hardcoded per channel, per era and per frame height — Digi24 alone needs two sets, split at the late-2023 format change.

**Temporal segmentation.** A three-hour broadcast contains maybe 30 minutes of interpretation. Digi24 is the hard case: it scatters interpretation across the program in bursts, the median one under 7 minutes. Three stages handle it:

1. *Pose heuristic* ([`digi/2.create_interpreter_binary_dataset.ipynb`](digi/2.create_interpreter_binary_dataset.ipynb)) — 40 frames per episode through `rtmlib` Wholebody; a frame counts as interpreted iff exactly one person is detected, facing the camera (shoulder-y asymmetry under 20% of the span), both elbows and wrists above 0.5 confidence, framed waist-up. Fast, noisy, good enough to bootstrap labels.
2. *Binary classifier* — a frozen `MobileNetV3-Small` with a trained head on those labels, stopped at 99.5% validation accuracy. Light enough to push every frame of every episode through it.
3. *State machine* ([`digi/3.create_segments.ipynb`](digi/3.create_segments.ipynb)) — stitches frame verdicts into segments with four thresholds: 2s disappearance tolerance (overlays occlude the interpreter), 90s long-absence cutoff, 10s minimum segment, 2s presence confirmation before a segment opens. Boundaries go to FFmpeg, which crops the box and pulls the audio.

ProTV and PrimaTV carry a single contiguous ~25–30 minute block, so they get the cheap treatment ([`*/2.segment_interpreter.ipynb`](protv/2.segment_interpreter.ipynb)): pre-computed face encodings for the known interpreters of each channel, a strided search at 15-minute intervals until one of them shows up, then a binary search for the exact start and end of the block. Afterwards, [`protv/3.verify_segments.ipynb`](protv/3.verify_segments.ipynb) sweeps 10 minutes either side with the MobileNet classifier to catch stray interpreted content, and rejects any block under 10 minutes as a detection error.

### Text: [`dataset/`](dataset/)

[`1.stt.ipynb`](dataset/1.stt.ipynb) runs in three passes, each loading exactly one model to keep VRAM survivable, and halving the batch size on OOM instead of dying:

- *Language filtering* — Silero-VAD chunks the 16 kHz audio at a 1s silence threshold, each chunk goes through the WhisperX `large-v3-turbo` encoder plus a language-ID head, and the result is a bit-packed Romanian/not-Romanian mask. Interviews, dubbed reports and press conferences in other languages get zeroed out before transcription.
- *Transcription* — WhisperX `large-v3`, language pinned to Romanian.
- *Alignment* — the `whisperx-align_ro` model turns 20–30 second chunks into word-level, then sentence-level timestamps, which is what we actually need to map text onto frame indices.

Then filtering, in the same notebook: cedilla → comma-below diacritics; sentences under 2s dropped; sentences whose mean word duration falls under 0.1s or over 2s dropped as broken alignments; and a hallucination detector for Whisper's repetition loops — any phrase of 1 to 64 words repeated three or more times consecutively while dominating at least half the sentence's tokens kills the sentence.

[`2.tok.ipynb`](dataset/2.tok.ipynb) trains a SentencePiece Unigram model over the whole filtered corpus (16,384 pieces, full character coverage, `<unk>`/`<pad>`/`<s>`/`</s>` at IDs 0–3) in two variants, lowercase and case-preserving, and writes the token IDs back into the per-segment JSONs. A word-level vocabulary was never on the table: 180k forms, half of them seen once.

[`3.create_dataset.ipynb`](dataset/3.create_dataset.ipynb) merges features and labels into the final layout — one `.npy` of pooled features and one `.json` of sentences per segment, each sentence carrying `id`, `start`/`end` frame indices, text and tokens in both casings. Training reads it by memory-mapping and slicing.

### Sign features: `*/4.extract_features.ipynb`

Frames are resized so the larger side is 256, center-cropped to 224×224, and streamed as overlapping windows of 8 frames at stride 2 into an `InceptionI3D` pretrained on BSL-5K. A forward hook on `Mixed_5c` grabs the 1024-d activations, which are spatially mean-pooled and stored as fp16. Streaming matters: segments are half an hour long and never fit in memory.

### Mouthing features: [`phonemes/`](phonemes/)

Interpreters mouth the words they sign, and those mouthings disambiguate manual signs that look alike. To read them we needed a Romanian lip-reader, so we built one.

- [`ro_vsr/1.ro_vsr.ipynb`](phonemes/ro_vsr/1.ro_vsr.ipynb) — clean and re-split the 100+ hour `ro_vsr` visual speech corpus.
- [`phonemes/2.phonemes.ipynb`](phonemes/phonemes/2.phonemes.ipynb) — phonemize the transcripts against the RoLEX lexicon (casing and hyphenation variants included); [`3.fill_unk.ipynb`](phonemes/phonemes/3.fill_unk.ipynb) sends whatever RoLEX misses to an LLM with 100 few-shot RoLEX lines and gets entries back in the same format. Final label set: 37 phonemes + CTC blank + a shared sos/eos, 39 classes.
- [`lip_reader/4.crop_mouths.py`](phonemes/lip_reader/4.crop_mouths.py) — MediaPipe face landmarks, mouth crops, Auto-AVSR-style preprocessing.
- [`lip_reader/5.finetune.py`](phonemes/lip_reader/5.finetune.py) — fine-tune `Auto-AVSR` video-only: ResNet-18 frontend and encoder layers 0–5 frozen from the English checkpoint, layers 6–11 plus CTC head and decoder trained from scratch. Loss is **0.9·CTC + 0.1·attention**, and that ratio is the whole trick. Push attention higher and the model hallucinates complete words, because natural speech always has them; drop attention entirely and CTC output comes out temporally scrambled. 90/10 gives 29.23% PER on natural speech and, more importantly, behaves on sporadic signer mouthings. We picked epoch 22 over the nominally better epoch 40 for exactly that reason.
- [`lip_reader/6.lipread.py`](phonemes/lip_reader/6.lipread.py) — cache frontend+projection features over the full corpus, chunked with halo padding so the results are bit-for-bit identical to a full-video forward pass.
- [`lip_reader/7.extract_logits.py`](phonemes/lip_reader/7.extract_logits.py) — per-segment CTC logits over `[start, end+5s]` windows, saved as `.phonemes.npz` (fp16, 39 classes; the sos/eos column is dropped later, leaving 38-d features).

What it produces isn't clean phonetics, and it isn't supposed to be. For *"Drumul de centură al orașului Călimănești din Vâlcea..."* the greedy decode reads `n u m i n e S t i t u n o r a S ...`: **drum** → `n u m` (/d/ and /n/ share a viseme, /r/ is swallowed by the lip rounding of /u/), **oraș** → `o r a S` (exact), **Vâlcea** → `f i gZ e` (/v/ and /f/ same viseme, *â* looks like /i/, /tS/ and /gZ/ pucker identically). Function words — *de*, *al*, *din*, *este* — are absent, which is correct: interpreters don't mouth them.

## The model side (summary)

`SignFormer`, a 6.6M-parameter gloss-free encoder-decoder over the I3D features. Three things had to happen before it worked on this data.

**Two bugs.** `PositionwiseFeedForward` added its own residual on top of the one its caller already added, computing `2x + FFN(LN(x))`; and `ResidualConnectionModule` reused the 0.5 module factor as the identity factor, computing `0.5·FFN(x) + 0.5x`. One exploded gradients, the other starved them. Together: no convergence. Fixing them was worth +3.82 BLEU.

**Décalage.** Interpreters lag the speaker by a couple of seconds, so the sentence boundaries we derive from audio are systematically wrong — the start of a clip belongs to the previous sentence and the signing runs past the end. Cross-attention maps show it plainly. Two remedies: feed extra frames past the annotated end (random 1–5s in training, fixed 4s in evaluation), and *confidence ramps*, a per-frame scalar that fades features in at the start and out through the extension so the model learns to distrust uncertain boundaries. The fade-in window is derived from the gap to the previous sentence, `clip(1.5 − 0.26·gap, 0.2, 1.5)`. Combined: +2.75 and +2.39 BLEU.

**Mouthing fusion.** The obvious sigmoid gate choosing between streams collapsed at ~360k steps — sigmoid is steep near 0, the gate saturated onto the noisy phoneme stream, and BLEU fell from 8.72 to 6.31 with no recovery. What works is keeping the sign stream whole and treating phonemes as a gated residual, `h = sgn + α · LN(proj(phn))` with `α = σ(W·sgn + b)`, `b` initialized to −2 so the model starts at ~12% phoneme influence and earns its way up. +0.78 BLEU on top of everything else.

Progression on the validation split, and the final test number:

| Configuration | BLEU-4 |
|---|---|
| Original SignFormer, both bugs | 2.74 |
| Both fixes + rand (2,4)s extra | 6.56 |
| Both fixes + flat 5s extra | 9.31 |
| Both fixes + rand (1,5)s + ramps | 9.98 |
| Phonemes + rand (1,5)s + ramps | **10.76** |
| ↳ same config, test split, tuned beam | **11.16** |

Per channel on test: Digi24 12.92, ProTV 11.16, PrimaTV 8.28 — PrimaTV suffers from less data and lower source resolution. For context, iLSU-T (202h, news domain) reports 3.43 and BOBSL (1,467h, diverse) 7.3, while narrow-domain PHOENIX-2014T sits at 26.75. Wild news translation is simply harder.

Every experiment ran on one RTX 4070 Ti, roughly 8 minutes per epoch including validation.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python ≥ 3.9, CUDA GPU, and `ffmpeg` on PATH. `face_recognition` needs dlib, which wants CMake and a compiler.

Not shipped, fetch separately:

| Artifact | Used by |
|---|---|
| WhisperX `large-v3`, `large-v3-turbo`, `whisperx-align_ro` | `dataset/1.stt.ipynb` |
| I3D `bsl5k.pth.tar` + the `bsl1k` model code (under `libs/`) | `*/4.extract_features.ipynb` |
| Auto-AVSR `vsr_trlrs2lrs3vox2avsp_base.pth` + the `auto_avsr` tree | `phonemes/lip_reader/` |
| RoLEX lexicon (`rolex.v1.txt`), `iulik-pisik/ro_vsr` | `phonemes/phonemes/`, `phonemes/ro_vsr/` |
| YouTube `cookies.txt` | `prima/1.scrape.ipynb` |
| Reference face crops in `*/known_people/` | ProTV / PrimaTV segmentation |

A fair number of paths, crop rectangles and channel-specific constants are hardcoded — this is research code that ran once over a fixed archive, not a library. Read the top cells before running anything.

## More detail

The [presentation](docs/RSL-News_presentation.pdf) is the 15-minute version. The [documentation](docs/RSL-News_documentation.pdf) is the real thing: related work, the full collection and alignment pipeline, every ablation, qualitative analysis of the translations, and the failure modes we couldn't fix.

Radu-Cătălin Micea, *RSL-News: Large-Scale Dataset and Enhanced Lightweight Transformer Model for Gloss-Free Romanian SLT*, master's work, Politehnica University of Timișoara, 2026. Scientific coordinator: Prof. Dr. Habil. Eng. Călin-Adrian Popa.
