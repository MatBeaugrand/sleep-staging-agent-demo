# Sleep staging on Sleep-EDF Expanded

A small, reproducible five-class sleep staging pipeline built on the public
[Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/1.0.0/)
sleep-cassette recordings, fetched through
`mne.datasets.sleep_physionet.age.fetch_data`.

The point of this repository is **methodological correctness, not accuracy**.
Feature set and models are deliberately simple; what is treated carefully is
the evaluation: subject-wise splits, no preprocessing fitted across folds, and
metrics chosen for a heavily imbalanced problem.

---

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate    # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m src.cli run --subjects 8 --folds 4 -v
```

The first run downloads ~350 MB of EDF into `data/raw/` (cached; `data/` is
git-ignored), extracts features into `data/derivatives/`, and writes figures to
`figures/` plus a metrics summary to `results/metrics.json`. Subsequent runs
reuse the feature cache and take seconds.

```bash
python -m pytest
```

The test suite is fully synthetic — it never downloads anything.

---

## Pipeline

### 1. Data (`src/data.py`)

* Eight subjects, night 1, sleep-cassette (`SC4ss1E0`). The subject id is
  parsed out of the filename and becomes the cross-validation group.
* Channels `EEG Fpz-Cz` and `EEG Pz-Oz` are kept; EOG, EMG and the respiration
  channels are dropped.
* Band-pass **0.3–35 Hz** (FIR, `firwin`), applied to the *continuous* signal
  before any cropping, so filter edge artefacts land in the discarded wake
  tails rather than at the edge of the sleep period.
* Epochs are cut with `mne.events_from_annotations(..., chunk_duration=30 s)`,
  so every 30 s window is aligned to the hypnogram by construction and never
  straddles two scored stages.
* Stage mapping (`config.STAGE_MAP`):

  | Annotation | Class |
  |---|---|
  | `Sleep stage W` | `W` |
  | `Sleep stage 1` | `N1` |
  | `Sleep stage 2` | `N2` |
  | `Sleep stage 3`, `Sleep stage 4` | `N3` (merged, per the AASM re-scoring of R&K) |
  | `Sleep stage R` | `REM` |
  | `Sleep stage ?`, `Movement time` | **dropped**, not assigned a class |

* **Sleep-period crop.** Cassette recordings span ~20 h, most of it wake with
  the subject out of bed. Only the interval from 30 min before the first
  non-wake epoch to 30 min after the last is kept. Without this the task
  collapses into wake detection; `--no-crop` reproduces the uncropped
  behaviour if you want to see that effect.

### 2. Features (`src/features.py`)

Per 30 s epoch, per channel — **12 features** = 2 channels × (5 bands + entropy):

| Feature | Definition |
|---|---|
| `{ch}_delta_rel` … `{ch}_beta_rel` | Share of total power in delta 0.5–4, theta 4–8, alpha 8–12, sigma 12–16, beta 16–30 Hz |
| `{ch}_spectral_entropy` | Shannon entropy of the normalised PSD over 0.5–30 Hz, divided by `log(n_bins)` → bounded [0, 1] |

Details that matter:

* **Welch PSD** with a 4 s Hann segment and 50 % overlap. At 100 Hz that is a
  0.25 Hz grid, so *every band edge falls exactly on a bin boundary* (asserted
  in `test_welch_grid_aligns_with_band_edges`) and ~14 segments are averaged
  per epoch.
* Band powers are **integrated** (`scipy.integrate.simpson`) rather than
  summed over bins, so values do not depend on the frequency resolution.
* The denominator for relative power is the sum of the five band powers. The
  bands tile 0.5–30 Hz contiguously, so **the five relative powers of a channel
  sum to exactly 1**. This is a deliberate, documented linear dependency: it
  costs one degree of freedom per channel, which is harmless for an L2-penalised
  logistic regression and irrelevant to trees, but it does mean the logistic
  coefficients are only interpretable relative to each other within a channel.
* A flat channel (dead electrode, clipped block) yields a uniform spectrum and
  entropy 1.0 rather than `NaN`.

### 3. Models (`src/model.py`)

| Name | Estimator |
|---|---|
| `logreg` | `StandardScaler` → multinomial `LogisticRegression(C=1, class_weight="balanced")` |
| `gbdt` | `HistGradientBoostingClassifier(class_weight="balanced", max_iter=200, early_stopping=False)` |

* Both carry **balanced class weights**. N1 is a few percent of epochs and is
  simply not predicted by an unweighted fit.
* Both are `Pipeline` objects, so the scaler is **fitted on the training fold
  only**. Standardising the full matrix before cross-validation would leak
  held-out subjects into training.
* `early_stopping=False` on the booster is deliberate: the built-in early stop
  carves a *random* validation slice out of the training fold, which would put
  adjacent epochs of the same subject on both sides of that split and make the
  run non-deterministic.

### 4. Validation (`src/evaluate.py`)

* `GroupKFold(n_splits=4)` with **subject as the group** — two held-out
  subjects per fold. `GroupKFold` does not shuffle, so folds are reproducible
  independently of the seed.
* Group disjointness is re-checked at runtime in `iter_group_splits`, not
  merely assumed, and is asserted directly in `tests/test_validation.py`.
* **Epochs are never split randomly.** Consecutive 30 s epochs from one night
  are near-duplicates; a random split puts a near-copy of every test epoch into
  training and inflates every metric — often by 15–20 points of kappa.
* Every reported number and both figures come from **out-of-fold predictions**:
  each epoch is predicted by a model that saw no epoch from that subject.

### 5. Metrics

Macro F1 (the headline — it weights rare N1 equally), per-class F1, Cohen's κ,
and a confusion matrix. Accuracy is reported too but is misleading here: N2
alone is roughly 40 % of epochs. Per-fold macro F1 and κ are reported as
mean ± sd alongside the pooled values, since eight subjects is a small sample
and the pooled number hides between-subject variance.

### 6. Figures (`figures/`)

* `confusion_matrix_{model}.png` — row-normalised confusion matrix, i.e. row
  *i* is the recall profile of annotated class *i*. Row support is printed on
  the axis so a row is not read as more reliable than it is.
* `hypnogram_{model}_subject{NN}.png` — annotated vs predicted hypnogram for a
  held-out subject, with a disagreement strip underneath.

---

## Results

8 subjects, night 1, 4-fold grouped CV, seed 42 — **7,647 epochs × 12 features**.

Class balance after the sleep-period crop:

| W | REM | N1 | N2 | N3 |
|---|---|---|---|---|
| 1287 (16.8 %) | 1231 (16.1 %) | 677 (8.9 %) | 3489 (45.6 %) | 963 (12.6 %) |

Out-of-fold performance:

| | Logistic regression | Gradient boosting |
|---|---|---|
| Accuracy | 0.740 | **0.780** |
| **Macro F1** | 0.690 (per fold 0.692 ± 0.030) | **0.716** (per fold 0.714 ± 0.015) |
| **Cohen's κ** | 0.653 (per fold 0.651 ± 0.020) | **0.693** (per fold 0.692 ± 0.026) |
| F1 · W | 0.833 | 0.852 |
| F1 · REM | 0.690 | 0.731 |
| F1 · N1 | 0.369 | 0.393 |
| F1 · N2 | 0.820 | 0.858 |
| F1 · N3 | 0.736 | 0.748 |

Reading the numbers:

* κ ≈ 0.65–0.69 is "substantial agreement" and is where a 12-feature, no-context
  model on two EEG derivations belongs. Published deep models on the full
  Sleep-EDF cohort reach κ ≈ 0.75–0.80; the gap is mostly temporal context and
  cohort size, not features.
* **N1 is the floor** (F1 ≈ 0.37–0.39). The confusion matrix shows why: N1 is
  spread across W and REM, which is the same confusion human scorers make —
  published inter-rater agreement on N1 sits around 45–60 %.
* Gradient boosting beats the linear baseline on every class, and its per-fold
  spread is *narrower* (± 0.015 vs ± 0.030 macro F1), so the ordering is not an
  artefact of one lucky fold.
* Per-fold spread is reported because eight subjects is a small sample; the
  pooled figure alone would hide between-subject variance.

![Gradient boosting confusion matrix](figures/confusion_matrix_gbdt.png)

![Hypnogram, subject 00](figures/hypnogram_gbdt_subject00.png)

The hypnogram figure is the honest picture of what this model does: the broad
architecture of the night — sleep onset, the N3-heavy first third, REM periods
lengthening toward morning, the wake tail — is recovered, but the trace is
visibly more fragmented than the expert scoring because each epoch is
classified in isolation.

---

## Layout

```
src/config.py      paths, seed, band definitions, stage map — no magic numbers elsewhere
src/data.py        fetch, filter, epoch, stage mapping, feature-matrix cache
src/features.py    Welch PSD → relative band powers + spectral entropy
src/model.py       the two estimator pipelines
src/evaluate.py    grouped CV, metrics, figures
src/cli.py         entry point
tests/             pytest suite (synthetic; no download required)
data/              git-ignored: raw/ downloads, derivatives/ feature cache
figures/  results/ generated output
```

## CLI

```bash
python -m src.cli run                      # features (cached) + both models + figures
python -m src.cli fetch --subjects 8       # download only
python -m src.cli features --force         # rebuild the feature cache
python -m src.cli run --models logreg      # one model
python -m src.cli run --folds 8            # leave-one-subject-out
python -m src.cli run --no-crop            # keep the full ~20 h recording
python -m src.cli run --plot-subject 3     # hypnogram figure for a chosen subject
```

## Reproducibility

* Seed 42 (`config.SEED`), applied to `random`, `numpy` and both estimators'
  `random_state`. `GroupKFold` is deterministic and needs no seed.
* No absolute paths anywhere: everything derives from `config.PROJECT_ROOT`.
  Set `SLEEP_DATA_DIR` to move the cache off the repository.
* `requirements.txt` holds lower bounds; `requirements-lock.txt` holds the
  exact resolved versions of the environment these results were produced in.
* Raw data is never committed (`data/` is git-ignored) — rerunning the fetch
  reproduces it from PhysioNet.

## Known limitations

These are scope choices, not oversights:

* **Eight subjects is small.** Sleep-EDF Expanded has 78 in the age study.
  Per-fold spread is wide; treat the pooled numbers as illustrative.
* **Epochs are classified independently.** No temporal context — no
  neighbouring-epoch features, no HMM or CRF smoothing over the hypnogram.
  This is the single largest cause of the fragmented predicted hypnogram, and
  the most obvious thing to add next.
* **N1 stays hard.** It is rare and transitional; inter-rater agreement on N1
  is poor even between human scorers, so a low N1 F1 is expected rather than a
  bug.
* **No artefact rejection.** Epochs are used as scored; there is no amplitude
  or flatline rejection beyond the graceful handling of dead channels.
* **No hyperparameter tuning.** Tuning would need a nested grouped CV to stay
  honest; defaults are used instead so that no choice is made on held-out data.

## Data citation

Kemp B, Zwinderman AH, Tuk B, Kamphuisen HAC, Oberyé JJL (2000). Analysis of a
sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the
EEG. *IEEE Transactions on Biomedical Engineering* 47(9):1185–1194.

Goldberger AL et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet.
*Circulation* 101(23):e215–e220.
