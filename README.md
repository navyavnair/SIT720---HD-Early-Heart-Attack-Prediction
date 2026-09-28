# Reproducing and Extending a Stacking-Based Ensemble for Early Heart Attack Prediction

**SIT720 Machine Learning — Task 11.1HD**
**Navya V Nair (226543885)**

Reproduction study and methodological extension of:

> M. Bhagat, A. Sharma and P. Agarwal, "An efficient stacking-based ensemble
> technique for early heart attack prediction," *Multimedia Tools and
> Applications*, vol. 84, pp. 36351–36375, 2025.
> doi: [10.1007/s11042-024-19293-7](https://doi.org/10.1007/s11042-024-19293-7)

---

## Headline findings

| | Accuracy | Evaluation |
|---|---|---|
| Paper, as published | 98.53% | single 80/20 split, 1025 rows |
| Paper's method, honest evaluation | 82.63% | 10-fold × 3 repeats, 302 unique patients |
| **Proposed pipeline, honest evaluation** | **85.18%** | 10-fold × 3 repeats, 302 unique patients |

1. The dataset contains **302 distinct patients presented as 1025 rows**. Under the
   paper's random 80/20 split, **97.8% of test rows have an identical copy in the
   training set**.
2. Removing that leakage costs the paper's ensemble **15.9 percentage points**.
3. Three columns of the paper's results table (Table 11) are **mislabelled duplicates**
   of other columns, including a reported **MCC of 1.0000 that is mathematically
   impossible** given the authors' own confusion matrix (the correct value is 0.9711).
4. The proposed pipeline improves on the paper's method by **+2.54 pp** under matched
   honest evaluation (paired *t*-test *p* = 0.0066, Cohen's *d* = 0.53).
5. Under honest evaluation, **no ensemble beats a single regularised logistic
   regression** on the corrected feature representation. This is reported as a
   negative result.

---

## Repository contents

| File | Description |
|---|---|
| `01_reproduction.ipynb` | **Part 1.** Faithful reproduction of the paper's protocol, duplicate audit, and an audit of the published metrics against the paper's own confusion matrices. |
| `02_proposed_method.ipynb` | **Part 2.** The proposed leakage-free, type-aware pipeline, ablation study, and paired significance testing. |
| `heart.csv` | The dataset used by the paper (Kaggle `johnsmith88/heart-disease-dataset`), 1025 rows × 14 columns. |
| `requirements.txt` | Python dependencies with pinned minimum versions. |
| `report/` | Technical report source. |
| `results/` | CSV outputs written by the notebooks. |

---

## Reproducing these results

```bash
python3 -m pip install -r requirements.txt
jupyter notebook
```

Run `01_reproduction.ipynb` first, then `02_proposed_method.ipynb`. Both read
`heart.csv` from the repository root and write their result tables into the working
directory.

Every random process is seeded (`SEED = 42` in Part 1, `SEED = 0` in Part 2), so the
numbers quoted in the report and in this README are reproducible exactly.

---

## Dataset

Source: <https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset>, cited
directly in Section 4 of the paper.

1025 rows, 13 predictors plus a binary target. The predictors fall into three types,
which matters for the proposed method:

- **Continuous** — `age`, `trestbps`, `chol`, `thalach`, `oldpeak`
- **Binary** — `sex`, `fbs`, `exang`
- **Nominal clinical codes** — `cp`, `restecg`, `slope`, `thal`, `ca`

The paper feeds all thirteen to every classifier as raw numbers. The five nominal
codes are categories, not quantities: chest-pain type 3 ("asymptomatic") is not three
times chest-pain type 1. Correcting this is the single largest source of improvement
in the proposed pipeline.

`heart.csv` is included so the study can be reproduced without a Kaggle account.

---

## Method summary

**Part 1** follows the paper as closely as its description allows: an 80/20 stratified
split, the same six base classifiers (logistic regression, decision tree, random
forest, XGBoost, naive Bayes, KNN), and a five-fold stacking meta-classifier. The
paper does not report hyperparameters, a random seed, or the identity of the
meta-classifier; the assumptions made in their place are documented in the notebook.

**Part 2** proposes a corrected pipeline:

1. **Duplicate-aware curation** — 1025 rows reduced to 302 distinct patients.
2. **Type-aware preprocessing** — nominal codes one-hot encoded; all transforms fitted
   inside each training fold.
3. **Capacity matched to the sample** — repeated stratified cross-validation with
   variance reporting, and model complexity chosen empirically rather than assumed.

---

## Acknowledgement of Generative AI use

Generative AI was used in the preparation of this submission for planning the
experimental structure, drafting and refactoring code, and editing written text for
clarity and grammar. All experiments were executed by me, all reported numbers
were produced by the code in this repository, and the interpretation, critical
analysis and conclusions are my own. 
