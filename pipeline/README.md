# How RSL-News was made

These notebooks made `../manifests/` and the labels: they find the interpreted broadcasts, locate
the interpreter, check the result and transcribe the speech. Using the dataset needs none of this
(see the [top-level README](../README.md)).

| Notebook | Does | Writes (in `work/`) |
|---|---|---|
| `1_find` | lists the broadcasts of the three archives | `broadcasts.json` |
| `2_segment` | downloads each broadcast and finds the frames with the interpreter | `segments.json`, audio, end frames |
| `3_verify` | downloads the segments as users will, checks them, writes the manifests | `alignment.json`, `release/manifests.json` |
| `4_labels` | transcribes, filters, trains the tokenizer, writes the labels and the splits | `release/` |

`release/` has the repository's layout: `manifests/`, `dataset/` (the labels), `tokenizer/` and
`docs/excluded_broadcasts.json`.

Every step is either automatic or a call made by hand, and the notebooks run in one of two modes:

| | Replay (`REPLAY = True`, the default) | Live run (`REPLAY = False`) |
|---|---|---|
| Automatic steps | read from `frozen/`: what the original run produced | run now, on the archives |
| Calls made by hand | `decisions/` | `decisions/`, plus new ones where a check flags something |
| Result | exactly the repository, checked byte for byte | close to it: the archives have changed, and Whisper and the language model vary |
| Runs | `3_verify` and `4_labels`, a few minutes on a CPU | all four, days on a GPU |

Two flaws of the original run live on in its outputs: its audio was off the video's clock, and
today's downloads of some of its segments land a frame off. A live run avoids or measures both;
replay corrects the original outputs with measurements kept in `frozen/` (see below).

## Replay

```bash
python3.10 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/jupyter nbconvert --to notebook --execute 3_verify.ipynb 4_labels.ipynb --output-dir work/
```

The last cell of `4_labels` compares the result with the repository.

## Live run

Set `REPLAY = False` and run the notebooks in order; each resumes where it stopped. Cells that
need a human are marked; their outcome goes to `decisions/`.

- An NVIDIA GPU, ffmpeg, and about 3 GB of disk per broadcast being processed (each download is
  deleted once its segments are found). 4_labels's settings are the original run's, for a 12 GB
  GPU; with less, lower its batch sizes, or use `COMPUTE_TYPE = 'int8_float16'`, which gives
  slightly different transcripts.
- Two installs that `requirements.txt` cannot express (see its header): `facenet-pytorch` without
  its old pins, and onnxruntime's GPU build in place of the CPU build that faster-whisper pulls in.
- PrimaTV: `cookies.txt` of a logged-in YouTube session, and the PO-token server that
  `../download.sh` builds and starts.
- `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY` with a `gpt-5-mini` deployment, for the
  foreign-language check.

## How the original run differed

The notebooks do what the original run did, with these exceptions. They are why some of
`frozen/` is not that run's raw output but a measurement made later:

- **PrimaTV segment bounds.** The face search's output was lost. `frozen/segments.json` has the
  bounds recovered by matching each cut segment against the YouTube video, frame by frame and by
  its audio.
- **ProTV broadcasts.** The original run took the broadcasts up to 2024-02-29 from protv.ro, at
  720p, and those from 2024-03-01 on from vid2.stirileprotv.ro, at 432p. vid2 no longer serves its
  files; protv.ro carries 249 of those 432p broadcasts unchanged, and the dataset uses those. While
  vid2 is down, 2_segment takes the later broadcasts from protv.ro's 432p version.
- **Download offsets.** For 46 ProTV segments and the 249 broadcasts above, the archive's stream
  lands one frame (once two) away from the broadcast as it was downloaded; `frozen/alignment.json`
  has these shifts, measured as 3_verify measures them.
- **The audio clock.** The audio of the original cuts was not on the video's clock: PrimaTV and
  ProTV copied it along with the cropped video, so it started at the keyframe before the first
  frame (up to 6.4 s early), and in all three channels gaps in the audio track were closed up,
  moving what followed. The transcripts were timed on that audio. `frozen/clocks.json` has, per
  segment, where that audio started and where its gaps moved it, measured on the original cuts;
  4_labels uses it to put the original transcripts on the video's frames. A new run cuts the audio
  on the video's clock and needs no such map.
- **The NeMo transcripts** that the foreign-language check showed to the model came from an
  earlier Whisper run and were paired with the chunks by position: for 101 of its 377 questions
  they were of another stretch of audio. `frozen/llm.json` keeps that check's verdicts; a live run
  transcribes exactly the chunks it asks about.
- **Digi24 interpreter labels** came from a face-recognition run whose output was lost; they are
  the most frequent interpreter of 3_verify's face check, which agrees with that run on every day.
- **ProTV's classifier sweep** (a check of the 432p segments for gaps in the interpretation)
  changed nothing and is left out.

## Files

- `common.py`: what several notebooks use: paths, file reading and writing, the interpreter boxes,
  video frames, face recognition, the frame classifier, the sentence filters and timing.
- `frozen/`: the original run's outputs at the automatic steps that cannot be redone identically.
  - `broadcasts.json`: PrimaTV's videos kept after the thumbnail check; Digi24's journal pages by
    date and their 720p video files.
  - `segments.json`: per broadcast, the interpreter, the height of the video and the segments,
    [start, end) in frames of the broadcast.
  - `alignment.json`: per PrimaTV and ProTV broadcast, the archive page and the shift of each
    segment, or why the archive no longer has it.
  - `asr.json.zst`: Whisper's aligned sentences of every transcribed segment, including those of
    broadcasts that were later excluded (the tokenizer was trained on them too).
  - `llm.json`, `clocks.json`: see above.
  - `models/digi24_classifier.pt`: 2_segment's Digi24 frame classifier.
- `decisions/`: the calls made by hand.
  - `digi24.json`: the journal page chosen for days with several; days and segments dropped
    (from the mean-frame review); frames cut from the start or end of segments (after the face
    check); the hand-checked reference segments of the face check.
  - `primatv.json`: videos dropped.
- `known_people/`: one picture of each PrimaTV and ProTV interpreter, for the face search.
- `phonemes/`: the mouthing features of the original work (phoneme logits from a Romanian lip
  reader), kept as they were; they are not part of the release.
