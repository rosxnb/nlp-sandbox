# Assignment 02 — NLP Fundamentals: Exercises 1–11

Question paper: [`assignment 2.pdf`](./assignment%202.pdf)

Eleven exercises building from word counting to a complete mini NLP pipeline. Everything is
implemented from first principles with the standard library and NumPy — no NLTK or spaCy — because
the point of these exercises is the mechanics, not the API.

## Contents

| File | Description |
|---|---|
| `RoshanBharati_NLPBasics_01.ipynb` | **Deliverable** — all 11 exercises |
| `output/exercise07_tfidf.csv` | TF, DF, IDF and TF-IDF per (document, term) |
| `output/exercise10_pipeline_summary.csv` | Mini-pipeline vocabulary, counts, TF, smoothed probabilities |
| `output/exercise10_cooccurrence.csv` | Mini-pipeline co-occurrence matrix |
| `output/exercise11_conditional_probabilities.csv` | Every bigram with `P(w2 \| w1)` |

## Exercises

| # | Exercise | Result |
|---|---|---|
| 1 | Word Frequency Counter | 9 tokens → 6 distinct words ✓ |
| 2 | Vocabulary Builder | Vocabulary size = 6 ✓ |
| 3 | Co-occurrence Matrix | 5×5 at window 1 — ⚠ see below |
| 4 | Detect Data Sparsity | 5 of 10 pairs never co-occur (60% zero cells) |
| 5 | Laplace Smoothing | cat 0.4545 / dog 0.3636 / bird 0.1818 — ⚠ see below |
| 6 | Keyboard Prediction | All three continuations of *like* are 1/3 — ⚠ see below |
| 7 | TF-IDF | Full matrices over 3 documents |
| 8 | Cosine Similarity | 0.98 ✓ |
| 9 | Find Similar Words | `lion`, 0.9985 ✓ |
| 10 | **Mini NLP Pipeline** | Vocabulary 7 ✓, predicts NLP/AI after *love* ✓ |
| 11 | Conditional Probability | 0.50 / 0.25 / 0.25 — matches all three worked examples ✓ |

## ⚠ Three discrepancies in the question paper

Three of the printed "expected outputs" do not follow from the formulas the same PDF specifies. Each
is flagged in place in the notebook with the arithmetic worked out. **The notebook implements the
stated formula in every case** and shows why the printed number differs.

**Exercise 3** — the PDF's matrix marks `on ↔ mat` as co-occurring. In *"the cat sat on the mat"*,
`on` is at index 3 and `mat` at index 5, separated by the second `the`. At window size 1 that cell
must be **0**. The notebook shows the window-2 matrix for comparison; the printed matrix does not
match window 2 either.

**Exercise 5** — the PDF prints `cat 0.50, dog 0.40, bird 0.20`, which sum to **1.10** and therefore
cannot be a probability distribution. Applying the PDF's own formula with Total = 8, V = 3 gives
5/11, 4/11 and 2/11. The PDF hedges on this itself: *"values will depend on the exact counts and
denominator."*

**Exercise 6** — the PDF prints `0.4 / 0.3 / 0.2` (summing to 0.9). In the given corpus `like` is
followed by `NLP`, `Python` and `coffee` exactly once each, so all three are **1/3**. Exercise 11
uses a larger corpus where the printed values *are* internally consistent, and the notebook
reproduces those exactly.

## Running

```bash
../.venv/bin/jupyter lab RoshanBharati_NLPBasics_01.ipynb
```
