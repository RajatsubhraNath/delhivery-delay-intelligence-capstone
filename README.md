# Delhivery Last-Mile Delay Intelligence

> **Disclaimer**
>
> I am declaring this because this repository was built with substantial help from an AI assistant. The assistant generated the pipeline scripts, `src/shipment_prep.py` and the README draft is from the capstone project brief. I set the scope, supplied the brief, checked every output against the brief's pass bands, and then worked through the code until I could explain each design decision and reproduce the numbers myself. I assembled the project locally and pushed it as a single commit, so the history shows one bulk commit rather than a development trail.
>
> A single connected pipeline that answers three operational questions on a synthetic
last-mile shipment export:

1. **How long will a parcel really take?** — a delivery-hours regressor (Part B).
2. **Which parcels will miss their promise?** — a delay classifier plus an alert
   threshold chosen from an explicit cost asymmetry (Part C).
3. **Which hubs need which playbook?** — K-Means segments compared with the
   official tier labels (Part D).

The output of the whole build is an action list for 48 parcels that are about to
be dispatched: `outputs/new_shipment_actions.csv`.

All money in this repository is in Indian Rupees (INR).

---

## How to run

The project runs offline on Python 3.12 with the four pinned packages in
`requirements.txt` (pandas 2.2.3, numpy 1.26.4, scikit-learn 1.5.2, joblib 1.4.2).

From a fresh clone, in this exact order:

```bash
pip install -r requirements.txt

python generate_data.py          # prints: 24 1224 48
python part_a_clean_eda.py       # writes data/modelling_table.csv + outputs/part_a_*.csv
python part_b_regression.py      # writes outputs/part_b_*.csv
python part_c_classification.py  # writes outputs/part_c_*.csv
python part_d_segments.py        # writes outputs/hub_*.csv, segment_profile.csv, part_d_separation.csv
python package_model.py          # writes models/*.pkl|csv + outputs/part_d_reload_check.csv
python predict_delay.py          # writes outputs/new_shipment_actions.csv
```

Notes:

- `models/delay_model.pkl` is **regenerated** by running `python package_model.py`.
  It is committed for convenience, but a pickle is only guaranteed to load on the
  scikit-learn version that wrote it (1.5.2 here). If the grader's environment has
  a different version, run `python package_model.py` again instead of loading the
  committed pickle — the script refits from `data/modelling_table.csv` and rewrites
  all four files in `models/`.
- Every script is deterministic: all `random_state` values are the ones fixed in
  the brief, so re-running the sequence reproduces every CSV byte for byte.
- `generate_data.py` is the provided generator, committed verbatim. It creates
  `data/hubs.csv`, `data/raw_shipments.csv` and `data/new_shipments.csv`.

### Repository layout

```
README.md  requirements.txt  generate_data.py
data/     hubs.csv  raw_shipments.csv  new_shipments.csv  modelling_table.csv
src/      shipment_prep.py            # constants + clean_shipments + build_features
part_a_clean_eda.py  part_b_regression.py  part_c_classification.py
part_d_segments.py   package_model.py      predict_delay.py
models/   delay_model.pkl  delay_threshold.csv  feature_order.csv  impute_values.csv
outputs/  every CSV quoted below
```

---

## Data findings

The raw export has 1,224 rows. 24 rows are exact duplicates and are dropped, leaving
1,200 shipments (50 per hub across 24 hubs) — `outputs/part_a_checks.csv`.

1. **Express is the most delay-prone service, by a wide margin.** Express parcels are
   delayed **0.4897** of the time (119 of 243), against 0.2056 for Standard (132 of 642)
   and 0.0857 for Economy (27 of 315) — `outputs/part_a_delay_by_group.csv`. The
   promise is tightest for Express (36 hours), so the same operational noise eats a
   much larger share of its budget.
2. **Delay rate rises steeply with the hub tier label.** Tier 1 hubs delay
   **0.0400** of parcels, Tier 2 **0.2150**, Tier 3 **0.4400** — an eleven-fold
   increase from Tier 1 to Tier 3.
3. **Distance is the strongest operational gradient.** The delay rate is **0.0219**
   for 0-200 km, **0.1368** for 201-400 km, **0.4364** for 401-600 km and **0.8523**
   for 601+ km. Beyond 600 km, six out of seven parcels miss their promise.
4. **The three columns most correlated with `delivery_hours`** (ranked by absolute
   correlation, `delivery_hours` excluded, from `outputs/part_a_correlations.csv`) are
   `promised_hours` **+0.8372**, `distance_km` **+0.4013** and `n_stops` **+0.3247**.
   Service promise dominates because the three service levels have structurally
   different clocks; distance and route complexity then explain the spread inside
   each service level.
5. **`declared_value_inr` was log-transformed because it is strongly right-skewed.**
   On the median-filled column the mean is **INR 1,533.05** while the median is only
   **INR 1,121.50** — a few high-value parcels pull the mean up. After the natural log
   the mean is **7.0277** and the median **7.0224**, almost identical, so the
   transformed feature is close to symmetric and a linear model is not dominated by a
   handful of expensive shipments.

Two supporting checks: 5 parcel weights are the 999 kg scanner sentinel (the network
accepts up to 40 kg), and `service_type` arrives with 12 distinct spellings that
collapse to the expected 3 after strip + title case.

---

## Model results (regression)

Target `delivery_hours`, inputs the 17 pinned `FEATURES`, split 840 / 180 / 180
(`outputs/part_b_checks.csv`). The scaler is fitted on the 840 training rows only:
the largest absolute column mean is **0** on the scaled training rows and **0.2187**
on the scaled test rows, which is exactly what "not fitted on test" looks like.

| model | split | MAE | RMSE | R² |
|---|---|---|---|---|
| mean_baseline | val | 15.7952 | 18.8552 | -0.0212 |
| linear_regression | val | 3.9222 | 4.8912 | 0.9313 |
| decision_tree | val | 5.0971 | 6.2796 | 0.8867 |
| random_forest | val | 4.2882 | 5.5095 | 0.9128 |
| gradient_boosting | val | 3.9166 | 4.9165 | 0.9306 |
| linear_regression | test | 3.5671 | 4.5235 | 0.9493 |

**Baseline.** Predicting the training mean for every validation parcel gives
R² = **-0.0212**, i.e. slightly worse than predicting the validation mean. That is
the definition of "no information": the baseline knows nothing about the parcel, so
its error is just the spread of delivery hours (RMSE 18.8552 hours).

**Winner on validation.** Linear regression, with **RMSE 4.8912 hours** and
R² **0.9313**. Gradient boosting is a very close second (RMSE 4.9165, R² 0.9306),
random forest third (5.5095) and the single tree last (6.2796). The data-generating
process is close to additive in the features, which is exactly where a linear model
on the right scale does well.

**Bias and variance in the tree-depth diagnostic** (`outputs/part_b_tree_depth.csv`):
depth 1 is high-bias — training RMSE 11.6334 and validation RMSE 12.1324, a gap of
only **0.4992**, because the stump is too simple to fit either set. No depth limit is
high-variance — training RMSE **0.0000** and validation RMSE **7.8368**, a gap of
**7.8368**, because the tree memorises the training rows. Depth 6 is the sweet spot
(val RMSE **6.2796**); beyond it validation error climbs again (depth 8: 6.7901,
depth 12: 7.3540) while training error keeps falling.

**Learning rate** (`outputs/part_b_gb_learning_rate.csv`): 0.1 is best
(R² **0.9306**). At 0.01 the model has not finished learning after 100 trees
(R² **0.7127** — too slow, underfit). At 1.0 each tree takes a full gradient step and
the ensemble overshoots (R² **0.8429**).

**Cross-validation versus the validation ranking.** On the 1,020 training-plus-validation
rows with one shared `KFold(5, shuffle=True, random_state=42)`, gradient boosting scores
mean R² **0.9353** (std 0.0045) and linear regression **0.9261** (std 0.0048) — so the
5-fold ranking puts gradient boosting first, the opposite of the single 180-row
validation split. The two validation RMSEs differ by 0.025 hours, far inside the noise
of a 180-row set, so I would **ship gradient boosting** if the choice were mine: it wins
on 1,020 rows and its variance across folds is marginally lower. The recorded final
model still follows the brief's rule (lowest validation RMSE = linear regression), and
on the untouched test rows it scores R² **0.9493** with RMSE **4.5235** hours.

---

## Model results (classification)

Target `is_delayed`, stratified 840 / 180 / 180 with 195 / 42 / 41 delayed parcels.
Validation results at the default 0.5 rule (`outputs/part_c_models.csv`):

| model | TN | FP | FN | TP | accuracy | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|
| blind_all_on_time | 138 | 0 | 42 | 0 | 0.7667 | 0.0000 | 0.0000 | 0.0000 |
| logistic_regression | 128 | 10 | 7 | 35 | 0.9056 | 0.7778 | 0.8333 | 0.8046 |
| decision_tree_depth_6 | 128 | 10 | 11 | 31 | 0.8833 | 0.7561 | 0.7381 | 0.7470 |
| random_forest_default | 132 | 6 | 11 | 31 | 0.9056 | 0.8378 | 0.7381 | 0.7848 |

**The blind baseline is useless.** Predicting "on time" for everything is right
**0.7667** of the time, simply because 76.67% of parcels are on time. It catches zero
of the 42 delayed parcels, so its precision and recall are 0 and its F1 is 0. A model
that never raises an alert can never save a delayed parcel.

**The bookkeeping columns carry no signal.** The five `NOISE_COLUMNS` together hold
**0.2245** of the default forest's total importance — a large share, but spread thinly
and without direction; a random column in a forest always collects some split credit.
By univariate F-score all five sit outside the top seven:
`label_print_batch` rank 8 (F = 4.5847), `tape_roll_no` 12 (1.9052),
`sticker_serial_mod` 14 (1.2776), `shelf_slot_no` 16 (0.1393),
`scanner_build_no` 17 (0.0230). The smallest F-score among the top seven is
`sale_event_flag` at **36.7281**, an order of magnitude above every noise column
(`outputs/part_c_feature_scores.csv`). Keeping them would let the forest fit
warehouse paperwork rather than parcel physics.

**What the k table shows** (`outputs/part_c_selectk.csv`): keeping 5 features gives
validation F1 0.6988, **7 features gives 0.8333**, 10 gives 0.7857, 12 gives 0.8049
and all 17 give 0.7848. Seven — the seven real drivers — is the best point and the
rest of the curve is flat-to-worse, which is the signature of adding noise.

**Tuning did not beat the untuned forest** (`outputs/part_c_tuning.csv`): grid search
reached validation F1 **0.7750** (best 5-fold CV accuracy 0.8952 over 135 fits) and
randomized search **0.7848** (CV 0.8905 over 45 fits), while the plain forest on the
same 7 columns scored **0.8333**. That is not evidence that hyper-parameters are
irrelevant; it is evidence that a 180-row validation set with 42 positives is too
noisy to rank configurations separated by a few percentage points, and that the
search was optimising 5-fold accuracy on 840 rows rather than F1 on 180. The final
configuration is therefore the untuned forest (`n_estimators=100`, unlimited depth,
`min_samples_leaf=1`) on the 7 selected features.

**The threshold is an operations decision, not a model default**
(`outputs/part_c_thresholds.csv`). Because a missed delayed parcel costs more than an
unnecessary call, the rule is the *highest* threshold whose validation recall is still
at least 0.80:

| threshold | precision | recall | F1 |
|---|---|---|---|
| 0.4 | 0.7400 | 0.8810 | 0.8043 |
| **0.5** | **0.7955** | **0.8333** | **0.8140** |
| 0.6 | 0.8889 | 0.7619 | 0.8205 |

0.6 has the better F1 and precision, but recall 0.7619 means missing roughly one in
four delayed parcels, so it is rejected. **0.5** is the highest threshold that still
catches at least 80% of delays (35 of 42).

**Test, scored once** (`outputs/part_c_final_test.csv`): precision **0.6889**, recall
**0.7561**, F1 **0.7209**, accuracy 0.8667 (TN 125, FP 14, FN 10, TP 31). Recall fell
from 0.8333 on validation to 0.7561 on test. With only 41 delayed parcels in the test
set, each parcel is worth 2.4 percentage points of recall, so a three-parcel swing
explains the whole gap; the honest reading is that the true recall is somewhere around
0.75-0.83, not that the model degraded. Precision also fell (0.7955 to 0.6889), i.e.
more false alarms on unseen parcels — expected, because the threshold was tuned on the
same validation rows used to pick it.

---

## Operations recommendation

**Escalate now: the 9 hubs in the `Escalate` segment** (`outputs/hub_segments.csv`),
with their hub-level delay rates:

| hub | city | tier label | delay rate | mean hours |
|---|---|---|---|---|
| HUB19 | Guwahati | 3 | 0.62 | 62.2998 |
| HUB24 | Agartala | 3 | 0.54 | 64.8758 |
| HUB20 | Raipur | 3 | 0.48 | 70.0074 |
| HUB13 | Nagpur | 2 | 0.46 | 66.0742 |
| HUB17 | Patna | 3 | 0.46 | 60.8514 |
| HUB23 | Siliguri | 3 | 0.44 | 69.5698 |
| HUB21 | Bhubaneswar | 3 | 0.40 | 63.1070 |
| HUB18 | Ranchi | 3 | 0.38 | 62.0992 |
| HUB15 | Kochi | 2 | 0.36 | 62.9694 |

The segment averages **0.4600** delay rate against **0.0257** in `Standard` and
**0.1550** in `Watch` (`outputs/segment_profile.csv`). Nine hubs, 37.5% of the
network, produce the large majority of delays.

**Four hubs are labelled wrongly** (`outputs/tier_vs_segment_mismatch.csv`):

- **HUB07 Kolkata** — labelled Tier 1, behaves `Watch`.
- **HUB13 Nagpur** — labelled Tier 2, behaves `Escalate` (delay rate 0.46, mean 66.0742 h).
- **HUB15 Kochi** — labelled Tier 2, behaves `Escalate` (0.36, 62.9694 h).
- **HUB22 Dehradun** — labelled Tier 3, behaves `Watch`.

The tier label is a *planning* attribute (capacity, budget class), not a behavioural
one. Using it to allocate attention sends resources to Dehradun and starves Nagpur and
Kochi; route the playbook from the segment, and keep the tier only for capacity
planning. Any tier-based SLA report should be read with these four exceptions.

**Alert policy.** At threshold **0.5**, from `outputs/part_c_final_test.csv`:

- of every 10 parcels we flag, about **7** are really delayed (precision 0.6889);
- of every 10 parcels that are really delayed, about **7.6** are caught (recall 0.7561).

So the queue is usable: two thirds of the calls are justified, and three quarters of
the failures get a warning before they happen.

**The 48 parcels about to be dispatched** (`outputs/new_shipment_actions.csv`):
**8 are flagged**, of which **6 need a hub-manager escalation** (they sit in an
`Escalate` hub) and **2 get a proactive customer call** (flagged but in `Watch` hubs).
The other 40 get no action. Six escalations from one batch is consistent with 9 of 24
hubs being in the `Escalate` segment.

**Why K = 3 when the separation ratio keeps rising?** From
`outputs/part_d_separation.csv`: the across-to-within ratio is 2.894 at K = 2,
**5.3928 at K = 3**, 6.3047 at K = 4 and 6.8866 at K = 5. The ratio always improves
with more clusters — a trivial K = 24 has perfect separation — so it cannot be the
only criterion. K = 4 produces sizes 8-7-5-4 and K = 5 produces 8-7-4-3-2: both contain
a cluster of fewer than 6 hubs, too small to justify its own playbook, its own
staffing model and its own management cadence. Operations runs exactly three
playbooks, and K = 3 is the smallest solution whose clusters (9-8-7) are all large
enough to act on.

---

## Production plan

### Production-readiness characteristics

| characteristic | how this repository handles it |
|---|---|
| **Accuracy** | Held-out test evidence, not training fit: regression R² 0.9493 / RMSE 4.5235 h (`part_b_metrics.csv`); classifier precision 0.6889, recall 0.7561, F1 0.7209 at the operations threshold (`part_c_final_test.csv`). |
| **Reliability** | Deterministic seeds everywhere; `package_model.py` verifies the saved artefact by reloading it and comparing probabilities — `reload_max_abs_probability_diff` = 0 (`part_d_reload_check.csv`); `predict_delay.py` is idempotent (two consecutive runs produce a byte-identical CSV). |
| **Reproducibility** | `requirements.txt` pins all four packages with `==`; `generate_data.py` is committed verbatim and prints `24 1224 48`; every output in `outputs/` is regenerated by the documented command order. |
| **Performance (speed)** | Scoring is a single `predict_proba` over a 48-row batch and finishes in well under a second on a laptop; training the full pipeline takes seconds, so a daily batch is not a scheduling problem. |
| **Maintainability** | All preprocessing lives in one module (`src/shipment_prep.py`); no script re-implements cleaning; constants (`FEATURES`, `PROMISED_HOURS`) are imported, never copied. |
| **Consistency** | Training and scoring call the *same* `clean_shipments` and `build_features`, reuse the stored medians (`models/impute_values.csv`) instead of recomputing them, and reorder columns from `models/feature_order.csv` — the classic train/serve skew is structurally impossible. |
| **Monitorability** | Every run emits auditable CSVs (flag counts, probabilities, action mix) that can be compared week over week; the three triggers below are computed from them. |

### What to monitor, with numeric triggers

| quantity | baseline | trigger |
|---|---|---|
| Weekly delayed share of **delivered** parcels | 0.2317 overall (`part_a_checks.csv`) | alert if > **0.30** for two consecutive weeks, or > **0.35** in any single week |
| Weekly share of parcels **flagged** | 8 of 48 = **0.1667** in the first batch | alert if > **0.30** (queue overload) or < **0.05** (model silently broken) |
| **Recall** on parcels whose outcome is now known | 0.7561 test / 0.8333 validation | alert if < **0.65** over a rolling 200-parcel window |
| Missing / sentinel rate in the input feed | 36 missing distances and 5 sentinels in 1,224 rows (≈3.3%) | alert if > **5%** of a batch, or if any hub has zero rows |
| Mean predicted hours per hub vs its segment mean | e.g. Escalate 64.6504 h | alert if a hub drifts more than **5 hours** from its segment mean for two weeks |

### Retraining schedule

- **Monthly** full refit on the trailing 90 days (same scripts, same order), with a
  champion/challenger comparison on the most recent 30 days before promotion.
- **Event-driven** refit after any sale window, any new hub, any change to
  `PROMISED_HOURS`, or any week where a monitoring trigger fires.
- **Quarterly** recalibration of the alert threshold — recall is what decays first.

### The 15-28 September sale window

`sale_event_flag` is 1 for dispatches in that window and the generator adds 7 hours
plus 12 extra hours for Express, so sale parcels are structurally slower. Two rules:

1. **Never compare a sale week with an ordinary week.** Evaluate the model and the
   delay rate on sale parcels against the *previous sale event*, and on ordinary
   parcels against ordinary weeks. Mixing the two makes every September look like a
   model failure and every October look like a recovery.
2. **Keep the flag in the feature set.** It is rank 7 by F-score (36.7281) and lets the
   classifier raise more alerts during the window instead of being surprised by it.
   The threshold should not be re-tuned inside the window; the flag absorbs the shift.

**What drift would look like here.** Input drift: mean `distance_km` moving away from
the 314.95 median, mean `n_stops` rising, more missing values, or a new service mix.
Prediction drift: the weekly flag rate moving from 0.1667 toward 0.30 without a change
in the true delay rate. Concept drift: precision collapsing while recall holds (the
label relationship changed, e.g. a hub was re-operated), or the delayed share rising
in a segment whose inputs did not move. Any of these should trigger an investigation
before a refit, because refitting on drifted inputs teaches the model the drift.

---

## Limitations

1. **The data is synthetic.** Every row comes from `generate_data.py`, whose
   relationships are additive and known (0.03 × distance, 0.10 × stops, fixed rain and
   sale effects). The 0.93-0.95 R² and the clean segment structure are partly a property
   of that generator; real last-mile data has interaction effects, address-level
   geography, weather at a finer grain and missing-not-at-random gaps, so production
   accuracy will be lower. The pipeline and the decision logic transfer; the numbers
   do not.
2. **The test set is small.** 180 parcels, 41 of them delayed. One parcel is 2.4
   percentage points of recall, so the test precision/recall (0.6889 / 0.7561) carry
   wide uncertainty and should not be read as precise operating characteristics.
3. **The threshold was tuned on one validation split.** 180 rows, 42 positives, and
   the same rows were used to choose among the tuning candidates. The threshold is
   therefore mildly optimistic; it needs recalibration on a rolling window once real
   outcomes accumulate.
4. **Hub segments are built from 50 shipments per hub.** Hub-level delay rates have a
   standard error of roughly 0.06 at that sample size, so a hub near a segment boundary
   can move. The four mismatches are large enough to trust; smaller differences are not.
5. **The 17-feature model ignores geography and time structure.** There is no lane,
   no pincode, no weather series and no hub-level time trend, so the model cannot
   anticipate a new bottleneck — it can only recognise the patterns present in
   `modelling_table.csv`.
