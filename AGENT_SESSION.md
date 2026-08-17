# Working session with Claude Code

This repository was built on 17 August 2026 in one session with Claude Code
(Claude Pro, personal account). The exported session transcript is in
`transcript/`. This file records what I specified, what the agent produced, and
what I changed afterwards.

## What I specified

A single opening prompt asked for a complete five-class sleep staging pipeline on
the public Sleep-EDF Expanded sleep-cassette recordings: EDF and hypnogram
loading, 30 s epoching aligned to the annotations, Welch spectral features, a
linear baseline and a gradient boosting model, cross-validation, a pytest suite
and two figures.

Because sleep EEG is my own field, that prompt fixed the methodology instead of
leaving it open. The errors that sink this kind of pipeline are predictable, so I
constrained them from the start:

* `GroupKFold` with the subject as group, and never a random epoch split;
* class weighting on both models, because N1 is rare;
* macro F1, per-class F1 and Cohen's kappa rather than accuracy;
* a test asserting that no subject appears in both train and test folds;
* fixed seed, no absolute paths, raw data never committed.

## What the agent produced

Twelve files and 51 tests, then the environment setup, the PhysioNet download,
the full run and both figures. It also read back its own output: it inspected
the generated PNGs, noticed the confusion-matrix title was clipped, widened the
figure and rewrote the title, and re-ran the suite to confirm nothing broke.

It raised four design questions before implementing, and I confirmed its
recommendation on each:

1. **Cropping.** It flagged that cassette recordings run about 20 hours, so most
   epochs are out-of-bed wake, and proposed keeping only the sleep period plus
   30 minutes on either side. I accepted. This was the agent's observation, not
   mine.
2. One night per subject rather than both.
3. `GroupKFold` with 4 folds given 20 subjects.
4. Running the pipeline end to end within the session.
* Channels `EEG Fpz-Cz`, `EEG Pz-Oz`, `EOG horizontal` and `EMG submental` are
  kept; the respiration and temperature channels are dropped.

## Where I disagreed with it

Not during the session. Having fixed the methodology in the prompt, I had
nothing to catch, and the transcript shows exactly one instruction from me. So I
did not treat a green test suite as evidence that the pipeline was correct.

After the session I audited the suite by mutation. `scripts/mutation_check.py`
breaks the pipeline in five methodologically meaningful ways, in a scratch copy
of the tree, and checks that the suite goes red for each. Three were caught. Two
survived with every test still passing:

1. **The sleep-period crop could be removed entirely.** `sleep_period_mask` was
   well tested as a pure function, but nothing verified that `epoch_raw`
   actually calls it: the synthetic recording in the fixtures is shorter than the
   30 minute margin, so the crop was a no-op in every test. Removing it takes the
   dataset from 7,647 epochs at 16.8 % wake to 21,947 at 71 % wake, which turns
   the task into wake detection.
2. **The balanced class weights could be dropped.** No test asserted them and
   none observed the consequence: recall on the rare N1 stage falls to exactly
   0.00 while overall accuracy moves by about two points. That is precisely the
   failure the weighting exists to prevent, and the reason macro F1 rather than
   accuracy is the headline metric.

`tests/test_audit.py` closes both gaps. All five mutations are now detected.
`results/audit-before.txt` and `results/audit-after.txt` are the recorded output
of the harness on either side of that change.


```
## Second session: temporal context, EOG and EMG

The first version sat at Cohen's kappa 0.693, which is roughly where the
pre-deep-learning state of the art on hand-engineered features sat. I ran a
second session to close the gap with the feature-based literature rather than
by switching to deep learning, working from Vallat & Walker 2021
(eLife 10:e70092, the YASA algorithm): rolling temporal context, the EOG and EMG
channels the first version discarded, per-recording robust normalisation,
additional non-linear features, and 20 subjects instead of 8.

### What the agent verified before proposing anything

It measured three facts in the actual files rather than assuming them, and one
of them changed the design:

* `EOG horizontal` is a genuine 100 Hz signal and survives the 0.3-35 Hz
  band-pass with 98 % of its variance intact.
* `EMG submental` is natively **1 Hz** in Sleep-EDF, upsampled to 100 Hz by MNE.
  Its Nyquist limit is 0.5 Hz, so every band in a 0.5-30 Hz set would be pure
  interpolation artefact, and the band-pass leaves 10 % of its variance. It
  therefore gets time-domain features on an unfiltered copy and no spectral
  features at all. YASA's EMG contribution cannot be reproduced on this dataset.
* One epoch is dropped mid-sleep-period across the eight subjects, so gaps in the
  epoch lattice are rare but real, which decided that rolling windows must be
  indexed on the epoch onsets rather than on row position.

### Where I corrected it

**The scale-invariance test.** It proposed replacing
`test_relative_powers_are_amplitude_invariant` with an assertion that the log
absolute power columns shift by `log(scale)`. That is wrong: power scales as the
square of amplitude, so scaling the signal by k shifts log absolute power by
`2*log(k)`. And log standard deviation and log IQR shift by `log(k)`, because a
standard deviation is an amplitude and not a power. A single shared assertion
would have been wrong in both directions, so the two constants are asserted
separately.

**The scope of normalisation for absolute power.** It flagged, correctly, that
raw log absolute power encodes electrode impedance and risks acting as a subject
identifier. I required it normalised per recording. This is a deliberate
departure from the reference: YASA keeps a raw unnormalised copy of each feature,
but it was trained on roughly 3000 nights and can absorb inter-subject amplitude
variation. With 20 subjects that column is far more likely to be a subject
identifier than signal.

**Two unguarded invariants, and what they exposed.** The agent added four
mutations of its own and left the original five verbatim, which I verified in the
diff before trusting the harness. But two invariants it had itself named as
design constraints had no mutation guarding them: that smoothing must never cross
a subject boundary, and that normalisation must never pool across recordings.
Its `norm_pooled` mutation looks like it covers the second but does not, since
dropping `axis=0` changes per-column into one global scalar, which is a
different defect.

Asked to add them, it established something worse than "they survive": no test
executed `build_dataset` at all. The only occurrence in `tests/` was inside a
docstring, so both mutations would have survived **structurally**, not by luck.
And `test_prepare_recording_gives_each_subject_the_same_answer_alone` asserted
the per-recording wiring in its docstring while testing only the pure function.

That is the same defect as the sleep-period crop from the first session, in new
code written after it was found: the pure function tested, the wiring not. The
fix was to extract `assemble_dataset` so the per-recording assembly is reachable
without a 960 MB download, add five tests driving three synthetic recordings at
deliberately divergent gains (1.0, 25.0, 0.2) and asserting each subject's rows
equal that subject processed alone to `rtol=1e-10`, and add the two missing
mutations. The refactor was verified to be numerically a no-op by rebuilding the
full 20-subject matrix with `--force` and diffing the metrics.

### What the agent debugged on its own

Splitting the work into seven commits, it staged individual diff hunks. That
corrupted `src/data.py` in two commits: `build_dataset` had been rewritten
wholesale into `assemble_dataset`, so hunk-subset staging spliced lines from one
function into the body of another, producing files that were not merely untested
but syntactically invalid. It detected this, reset both commits, rebuilt the
intermediate states by authoring the file content explicitly, and verified every
source commit with `ast.parse`. It reported the whole episode without being
asked.

### Results

20 subjects, night 1, 5-fold grouped CV, 20,626 epochs x 117 features:

| | Logistic regression | Gradient boosting |
|---|---|---|
| Macro F1 | 0.773 (+/- 0.013) | 0.788 (+/- 0.029) |
| Cohen's kappa, pooled | 0.747 (+/- 0.010) | 0.780 (+/- 0.026) |
| Median per-subject kappa | 0.751 | 0.801 (range 0.662-0.873) |
| F1 on N1 / REM | 0.497 / 0.837 | 0.502 / 0.862 |

At constant scope, same 8 subjects and 4 folds and seed as the first version,
only the features differing: kappa 0.693 to 0.811, macro F1 0.716 to 0.825, F1
on N1 0.393 to 0.608, F1 on REM 0.731 to 0.888. The gain concentrates in N1 and
REM, the two stages diagnosed by context and by eye and muscle signals rather
than by EEG spectrum alone, which is the expected shape rather than a uniform
lift.

The 8-subject ablation scores *higher* than the 20-subject run. That is not an
anomaly: the original eight were a favourable draw, and the per-subject range on
20 subjects (0.662-0.873) shows how much between-subject variance eight subjects
hides. The 20-subject numbers are the result; the ablation only isolates the
effect of the four changes.

Median per-subject kappa 0.801 sits at YASA's reported 0.80, but this is not a
like-for-like comparison and is not claimed as one. Their figure is measured on
external cohorts with different montages and different scorers; ours is
within-cohort, held-out subjects only, same montage and same scoring standard.
Generalising across subjects of one cohort is a substantially easier test than
generalising across cohorts.

## What I am not claiming

## What I am not claiming

**The commit history is dated 17-18 August.** The repository was not under
version control during the first session, so the earliest commits cover the
audit and the corrections rather than the generation. Nothing has been
backdated. The exported transcript in `transcript/` is the record of the first
session.

**I did not contradict the agent during the first session.** Having fixed the
methodology in the opening prompt, I had nothing to catch; the transcript shows
exactly one instruction from me and four design choices confirmed. My
corrections are in the audit that followed and in the second session.

**The intermediate commits are organisational, not bisectable.** The feature
matrix goes from 12 to 39 to 117 columns across `config.py`, `features.py` and
the tests, so the suite only goes green once the tests commit lands. `HEAD` is
the verified state; `git bisect` across that range would not be meaningful.

**`load_raw` is untested**, so the decision to let EMG bypass the 0.3 Hz
high-pass rests on a measurement rather than on a guarded invariant. Closing it
needs a committed or synthesised EDF fixture, which is more machinery than this
repository needs. Named rather than hidden.

**The literature that directed the second session, and the mutation audit, were
carried out with assistance from Claude on claude.ai**, used as a reviewer
against the Claude Code output. The references are cited so the reasoning can be
checked independently. Cross-checking one agent's work with another was
deliberate.