# How RSL-News was made

These notebooks made `../manifests/` and the labels: they find the interpreted broadcasts, locate
the interpreter, check the result and transcribe the speech. Using the dataset needs none of this
(see the [top-level README](../README.md)).

| Notebook | Does | Writes (in `work/`) |
|---|---|---|
| `1_find` | lists the broadcasts of the three archives | `broadcasts.json` |
| `2_segment` | downloads each broadcast once, finds the frames with the interpreter, checks them against the users' downloads, keeps their audio | `segments.json`, audio, contact sheets |
| `3_verify` | shows what needs a human, writes the manifests | `release/manifests.json` |
| `4_labels` | transcribes, filters, writes the labels, trains the tokenizer, draws the splits | `release/` |

`release/` has the repository's layout: `manifests/`, `dataset/` (the labels) and `tokenizer/`.

Every step is either automatic or a call made by hand. The automatic steps' outputs are kept in
`frozen/`, the calls made by hand in `decisions.json`, and the notebooks run in one of three modes
(`MODE` in `common.py`, or the environment variable `RSL_NEWS_MODE`):

| | Replay (`'replay'`, the default) | Update (`'update'`) | Live run (`'live'`) |
|---|---|---|---|
| Automatic steps | read from `frozen/` | `frozen/`; for the broadcasts aired since, run now | run now, on the archives |
| Calls made by hand | `decisions.json` | `decisions.json`, plus new ones for the new broadcasts | `decisions.json`, plus new ones where a check flags something |
| Result | exactly the repository, checked byte for byte | the repository, plus the new broadcasts | close to the repository: the archives have changed, and Whisper and the language model vary |
| Runs | `3_verify` and `4_labels`, a few minutes on a CPU | all four, on a GPU, for the new broadcasts | all four, days on a GPU |

## Replay

```bash
python3.10 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/jupyter nbconvert --to notebook --execute 3_verify.ipynb 4_labels.ipynb --output-dir work/
```

The last cell of `4_labels` compares the result with the repository.

## Update

Set the mode to `'update'` and run the notebooks in order; each resumes where it stopped. Cells
that need a human are marked; their outcome goes to `decisions.json`. Only the broadcasts the
release has not done are added: ProTV and Digi24 days after the release's last, PrimaTV videos not
checked yet. Broadcasts of the release can be done again too (`REDO` in 2_segment, e.g. after a
better search): what the update makes of them replaces what the release had, except the transcripts
of ranges whose frames did not change. Every broadcast keeps its split; new broadcasts are drawn so
that each channel's test and validation splits stay at 5% of its labelled hours. Once the release in
`work/release/` is right, set `FREEZE = True` in 4_labels's last cell: the update's outputs join
`frozen/`, and replay rebuilds the release with them. Then copy `work/release/` to the repository
and set the mode back to `'replay'`.

A new interpreter needs a picture in `known_people/` and a place in `common.INTERPRETERS` (PrimaTV,
ProTV), or a reference segment in `decisions.json` (Digi24), before they are recognized; 3_verify
lists the broadcasts in which an interpreter was expected but none was recognized.

## Live run

Set the mode to `'live'` and run the notebooks in order, as for an update: everything is made
anew, from the archives as they are now.

## What an update or a live run needs

- An NVIDIA GPU, ffmpeg, and about 3 GB of disk per broadcast being processed (each download is
  deleted once its segments are found and checked). 4_labels's settings are for a 12 GB GPU; with
  less, lower its batch sizes, or use `COMPUTE_TYPE = 'int8_float16'`, which gives slightly
  different transcripts.
- Two installs that `requirements.txt` cannot express (see its header): `facenet-pytorch` without
  its old pins, and onnxruntime's GPU build in place of the CPU build that faster-whisper pulls in.
- PrimaTV: `cookies.txt` of a logged-in YouTube session, and the PO-token server that
  `../download.sh` builds and starts.
- `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY` with a `gpt-5-mini` deployment, for the
  foreign-language check.

## What the notebooks do not show

The release was made in stages, with earlier versions of these notebooks; `frozen/` keeps what
each made, where it cannot be redone the same today:

- **The first search of PrimaTV and Digi24** ran on broadcasts since deleted; their ranges are in
  `frozen/segments.json`. PrimaTV's were searched again in October 2026 with 2_segment's search, on
  the videos downloaded anew. The bounds of the first PrimaTV ranges had been lost and were
  recovered by matching each cut segment against the YouTube video, frame by frame and by its
  audio; a bound the second search found within 1 s of them kept their value.
- **ProTV's first search** was coarser than 2_segment's: a frame every 15 minutes until one shows
  a known interpreter, binary searches for the block's first and last frame, then the 10 minutes
  after the block probed every 5 s (if the interpreter is back, a second range up to their next
  disappearance) or, if not, the block's last 10 minutes probed backwards every 5 s, splitting the
  block around a gap where the interpreter is missing for most of 3 s. protv.ro did not serve the
  broadcasts again to search them anew: their ranges are what that search found, and a run with
  2_segment's search may find a little more at their edges.
- **The check inside the ranges** (a frame every 2 s) was added after those searches. It ran on
  the ranges as users download them (2_segment's last cell; `segments` in `frozen/segments.json`),
  and replaced two checks made by hand on Digi24. A review of each segment's mean frame had dropped
  91 segments: the check drops 86 of them, and keeps 2 in which the interpreter is idle and 3 that
  hold real interpretation after footage. Cuts had been placed by hand after an identity check, on 21
  segments: the check makes 20 of them, to the frame, and splits two of those around a gap in the
  middle; on 2024-05-02 a face in a crowd fools it, and that cut stays in `decisions.json`.
- **ProTV's sources.** The broadcasts up to 2024-02-29 come from protv.ro, at 720p; those from
  2024-03-01 were taken from vid2.stirileprotv.ro, at 432p, which no longer serves its files.
  protv.ro carries 249 of those unchanged, and 90 more as another encode of the same video: the
  same frames at the same times, but softer, often with a hearing-impaired logo burnt in, and in
  some with a few frames frozen or damaged. The release takes them from protv.ro; 2_segment does
  the same while vid2 is down.
- **Download offsets.** For some ProTV broadcasts the archive's stream lands one frame (once two)
  away from the broadcast as it was downloaded; `shift` in `frozen/segments.json` has these, as
  2_segment measures them.
- **The audio clock.** The audio of the first ranges transcribed was cut together with the video
  by stream copy, so it started at the keyframe before the first frame (up to 6.4 s early), and gaps
  in the audio track were closed up, moving what followed. Their transcripts carry, as `clock`,
  where that audio started and where its gaps moved it, measured on those cuts; 4_labels uses it to
  put them on the video's frames. 2_segment cuts the audio on the video's clock.
- **The NeMo transcripts** that the foreign-language check showed to the model for the first ranges
  transcribed are NeMo's transcription of the same audio, made over an earlier Whisper run's pieces
  and matched to each chunk by time (the audio was not kept); a live run transcribes exactly the
  chunks it asks about. The verdicts are `removed` in the transcripts.
- **Digi24's interpreters** of the first search came from a face-recognition run whose output was
  lost; they are the most frequent interpreter of the identity check, which agrees with that run
  on every day.

## Files

- `common.py`: what several notebooks use: paths, file reading and writing, the interpreter boxes,
  video frames, face recognition, the frame classifier, the sentence filters and timing.
- `frozen/`: the automatic steps' outputs, which cannot be redone identically.
  - `broadcasts.json`: what 1_find listed: Digi24's journal pages by date and their video files.
    ProTV's lists were not kept; each broadcast's protv.ro episode is its `source` in
    `segments.json`.
  - `segments.json`: per broadcast, in the order found: the interpreter, the height of the video,
    the ranges [start, end) in frames of the broadcast, the archive page, the shift of each range
    and the segments as users download them (after the check inside the ranges); or why the archive
    no longer has it, or None when no interpreter was found.
  - `transcripts.json.zst`: per range (`<broadcast>/<range>`), Whisper's aligned sentences, those
    the foreign-language check removed, and `clock` where needed (above).
  - `splits.json`: the split of each broadcast.
  - `models/digi24_classifier.pt`: 2_segment's Digi24 frame classifier.
- `decisions.json`: the calls made by hand, per channel, as frames of the broadcast as users
  download it: from 3_verify's review, `dropped_segments` (segments that are not interpretation)
  and `cuts` (the frames to keep of segments the face search was fooled on). Also, for Digi24, the
  journal page chosen for days with several, the days dropped as duplicates, and the hand-checked
  reference segments of the identity check; for PrimaTV, videos outside the release checked by
  eye, and whether they show an interpreter.
- `known_people/`: one picture of each interpreter, `I1` to `I8`; one who interprets on two channels
  keeps one ID. PrimaTV's and ProTV's are what the face search looks for (`common.INTERPRETERS`).
  Digi24's faces are too small and blurred for one picture: its identity check compares with frames
  of the reference segments instead, and `I8`'s picture is for reference only.
- `phonemes/`: the mouthing features of the original work (phoneme logits from a Romanian lip
  reader), kept as they were; they are not part of the release.
