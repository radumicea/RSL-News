# I3D features: why these settings

`extract_features.py` uses the BSL-5k I3D and the preprocessing of BOBSL, with two changes:

- **windows of 8 frames at stride 2**, instead of 16 frames at stride 4;
- **the shorter side resized to 224**, instead of 256, so the centre crop keeps the whole shorter side of the interpreter box.

Both settings translate better on RSL-News.

## Ablation

Setup:

- **Model and data:** Signformer trained on the val and test broadcasts only, re-split 80/10/10 by broadcast. That is 43.5k training sentences, about 9% of the train split.
- **Training:** 100 epochs, checkpoint and beam chosen on dev, one seed per row.
- **Score:** test BLEU-4. Every row has the same sentences, so the intervals are paired 95% bootstrap intervals.

| Window / stride | Resize (then centre crop 224) | Test BLEU-4 | Difference to the first row |
|---|---|---|---|
| **8 / 2** | **shorter side → 224** | **2.86** | |
| 8 / 2 | longer side → 256 | 2.62 | −0.24 [−0.41, −0.06] |
| 16 / 4 | shorter side → 224 | 1.91 | −0.94 [−1.12, −0.77] |
| 16 / 4, with the encoder's conv kernel and query pooling halved (31 → 15, 7 → 3) | shorter side → 224 | 2.22 | −0.64 [−0.81, −0.46] |

- **BOBSL's resize** (shorter side → 256) scored 0.35 [0.21, 0.49] below shorter side → 224 at 16 / 4. That comparison used a slightly larger re-split.
- **Hands partly cut by the crop** (MediaPipe, 1,770 sampled frames):

  | Resize | Hands partly cut |
  |---|---|
  | shorter side → 224 | 4.5% |
  | longer side → 256 | 6.6% |
  | shorter side → 256 (BOBSL) | 9.2% |

  About 2.2 points of each are already outside the broadcast's box.
- **Adapting the model to 16 / 4 recovers only part of the gap** (last row). The 8 / 2 windows sample the signing twice as often.
- **Caveat:** single seeds on a small subset. The 16 / 4 vs 8 / 2 gap is large; differences under about 0.3 are less certain.

The 8 / 2 features are twice as many per second, so they take twice the disk space of 16 / 4 (about 110 GB for all splits). The extraction takes about the same GPU time.
