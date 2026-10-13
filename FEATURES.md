# I3D features: settings and why

`extract_features.py` turns each segment into one 1024-d feature per **window of 8 frames, every 2 frames**: 12.5 features per second, each covering 0.32 s. Feature k covers frames [2k, 2k + 8).

- **Model:** the BSL-1K I3D with the BSL-5K weights (`bsl5k.pth.tar`). The feature is Mixed_5c averaged over space (7 × 7), computed with fp16 autocast. For an 8-frame window Mixed_5c has a single time step, so nothing is averaged over time; a 16-frame window gives two time steps, which are averaged.
- **Frames:** RGB / 255, **shorter side resized to 224** (bilinear), centre crop 224 × 224, minus 0.5, as in the model's reference test-time code.
- **Reproducible:** running the script again reproduces the files in `dataset/` up to fp16 noise: per window, the difference is about 0.1% of the feature's norm (at most 0.25%), cosine similarity above 0.99999.

BOBSL's features use 16-frame windows at stride 4 and resize the shorter side to 256. Both differences are deliberate: on RSL-News they translate better, as shown below.

## Results

Every row is Signformer trained on the val and test broadcasts only, re-split 80/10/10 by broadcast (43,547 / 5,513 / 5,596 sentences). The code, training config, annotations and test sentences are the same in every row. Each run is 100 epochs, with the checkpoint, beam size and length penalty chosen on dev. Differences are against the reference row, with 95% paired bootstrap intervals over the test sentences.

| Window / stride | Resize (then centre crop 224) | Encoder conv width / query pooling | Test BLEU-4 | Difference | chrF | ROUGE |
|---|---|---|---|---|---|---|
| **8 / 2** | **shorter side → 224** | **15 / 3** | **2.96** | +0.10 [−0.08, +0.29] | **22.56** | **14.85** |
| 8 / 2 | shorter side → 224 | 31 / 7 | 2.86 | reference | 21.93 | 14.32 |
| 8 / 2 | longer side → 256 | 31 / 7 | 2.62 | −0.24 [−0.41, −0.06] | 21.84 | 13.89 |
| 16 / 4 | shorter side → 224 | 15 / 3 | 2.22 | −0.64 [−0.81, −0.46] | 20.60 | 12.60 |
| 16 / 4 | shorter side → 224 | 31 / 7 | 1.91 | −0.94 [−1.12, −0.77] | 19.86 | 11.53 |

The longer side → 256 row uses the first features extracted for this corpus. They come from an earlier notebook, with the same model and normalisation, on the same broadcasts. On these segments their frames match exactly or within one window.

### Window and stride: the largest effect

- **8 / 2 beats 16 / 4 by 0.94 BLEU-4** (chrF +2.1).
- **Twice the rate, half the span:** at 8 / 2 the model gets twice as many features per second (12.5 instead of 6.25), and each covers half the time (0.32 s instead of 0.64 s). The BOBSL paper notes that signs in context typically last up to 13 frames, so a 16-frame window can span more than one sign.
- **Adapting the model to 16 / 4 isn't enough:** Signformer's encoder counts in windows. Its convolution spans 31 windows and its attention queries pool 7, so at 16 / 4 the convolution spans 5 s. Shrinking the settings to 15 / 3 recovers a third of the gap (+0.31 [+0.16, +0.46] over 31 / 7), but 16 / 4 still ends 0.64 below 8 / 2.

### Encoder settings for these features

- **Where the defaults come from:** Signformer's 31 / 7 were set on PHOENIX-2014T, which has one feature per frame at 25 fps, so 31 features span 1.24 s. With 8 / 2 features, 31 span 2.5 s; 15 brings the span back to 1.2 s.
- **Result:** 15 / 3 gives +0.10 BLEU-4 [−0.08, +0.29] and **+0.63 chrF [+0.44, +0.82]**, with the same training time and 8k fewer parameters. Signformer's `configs/sign_AdamW.yaml` uses 15 / 3.

### Resize and crop

The interpreter boxes (the videos in `mp4s/`) are small and nearly square:

| Channel | Box (width × height) and share of segments |
|---|---|
| Digi24 | 192 × 206 (58%), 248 × 224 (42%) |
| PrimaTV | 254 × 274 (94%), 382 × 410 (6%) |
| ProTV | 300 × 322 (73%), 180 × 192 (27%) |

- **What each resize keeps:**
  - **Shorter side → 224** keeps the whole width of a portrait box and cuts only its top and bottom: about 8 px each on a 192 × 206 box. The landscape Digi24 box (248 × 224) isn't resized at all; the crop removes 12 px on each side.
  - **Longer side → 256** zooms in about 6% more (3% on the landscape Digi24 box) and also cuts the sides.
  - **BOBSL's shorter side → 256** zooms in by 1 / 0.875 and cuts 12.5% of the width.
- **Hands partly cut by the crop:** measured with MediaPipe hand landmarks on 1,770 sampled frames (2,638 hands). A hand counts as cut if a landmark falls outside the crop.

  | | Shorter → 224 | Longer → 256 | Shorter → 256 (BOBSL) | Already outside the box |
  |---|---|---|---|---|
  | All | **4.5%** | 6.6% | 9.2% | 2.2% |
  | Digi24 | 3.5% | 4.0% | 4.8% | 2.5% |
  | PrimaTV | 1.2% | 2.0% | 5.3% | 0.6% |
  | ProTV | 7.8% | 13.1% | 18.3% | 2.8% |

  - "Already outside the box" means the broadcast's own inset cuts the hand, so no crop can recover it. These are mostly at the bottom of the picture. In Digi24's 2019 layout, 15.5% of hands are already outside the box.
  - ProTV loses the most: its interpreters' hands reach the edges of the box. In 2024, BOBSL's resize cuts 27.7% of ProTV hands, against 11.9% for shorter → 224.
- **Results:**
  - At 8 / 2, shorter → 224 beats longer → 256 by 0.24 (table above).
  - At 16 / 4 it beat BOBSL's shorter → 256 by 0.35 BLEU-4 [+0.22, +0.49]. This was an earlier pair of runs, on a slightly larger version of the same re-split and the old annotations. The crop was chosen on dev (+0.31), with the rule fixed before the result.
- **Zoom:** the BSL I3D was trained on frames scaled by 0.875–1 before the 224 crop. Shorter → 224 is the 0.875 end: the most zoomed-out view the model saw in training.

## Not tested

- **Letterboxing, or no crop:** keeping the whole box would bring cut hands from 4.5% down to 2.2%. But the model never saw either alternative in training: letterboxing makes the signer about 7% smaller, with bars, and feeding the whole box gives a non-square 224 × 241 input.
- **8 / 2 with BOBSL's resize** (shorter side → 256): at 16 / 4 it lost 0.35.
