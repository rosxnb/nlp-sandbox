# NLP Sandbox

Assignments from the M.Tech in AI at Kathmandu University, course code **AIAC 536** (Natural Language
Processing).

## Assignments

One folder per question paper. Each folder holds its own question PDF alongside the solution.

| # | Assignment | Source | Deliverable | Status |
|---|---|---|---|---|
| 1 | [POS Extraction + NER](assignment-1-pos-and-ner) | [`assignment 1.pdf`](assignment-1-pos-and-ner/assignment%201.pdf) | 2 notebooks, 2 CSVs | ✅ |
| 2 | [NLP Fundamentals — Exercises 1–11](assignment-2-nlp-fundamentals) | [`assignment 2.pdf`](assignment-2-nlp-fundamentals/assignment%202.pdf) | 1 notebook, 4 CSVs | ✅ |
| 3 | [Word Embeddings](assignment-3-word-embeddings) | [`assignment 3.pdf`](assignment-3-word-embeddings/assignment%203.pdf) | 3 notebooks | ⚠️ code done, handwritten PDF pending |
| 4 | [RNN & BPTT](assignment-4-rnn-handwritten) | [`RNN_Assignment_BPTT.pdf`](assignment-4-rnn-handwritten/RNN_Assignment_BPTT.pdf) | Handwritten PDF + `rnn_bptt.py` | ✅ |

> **Note on the numbering.** `assignment 1.pdf` contains *two* separately-titled parts — "Assignment
> 01: POS Extraction" and "Assignment 02: NER" — on consecutive pages. Both live under
> `assignment-1-pos-and-ner/` as `part-1-pos/` and `part-2-ner/`, keeping one folder per question paper.

### Outstanding

Assignment 3 Task 2 is a **Conceptual Assignment**, which must be submitted as handwritten notes
scanned to PDF. The analysis is complete and
[`Task2_Handwritten_Notes.md`](assignment-3-word-embeddings/Task2_Handwritten_Notes.md) condenses it
for copying out by hand, but **the handwritten PDF still needs to be written, scanned and uploaded.**

## Setup

A single shared Python environment at the repository root serves every assignment.

```bash
./setup.sh
```

This creates `.venv`, installs [`requirements.txt`](requirements.txt), downloads the spaCy model and
NLTK data, and registers a Jupyter kernel named **Python (NLPSandbox)**.

Then:

```bash
.venv/bin/jupyter lab
```

Every notebook uses paths relative to its own folder, so they run from wherever they sit.

### Why the versions are pinned

This repo was set up on **Intel macOS (x86_64)**, where PyTorch stopped publishing wheels after
**2.2.2**. That constraint cascades:

- `torch==2.2.2` was built against NumPy 1.x → **`numpy<2`** (NumPy 2 raises `_ARRAY_API not found`)
- `transformers` 5.x requires `torch>=2.5` → **`transformers>=4.44,<5`**
- `gensim` 4.4 wants NumPy 2 → **`gensim==4.3.3`** with `scipy<1.14`

The environment is also pinned to **Python 3.12**, since spaCy and PyTorch have no 3.14 wheels.

On Apple Silicon or Linux these constraints can be relaxed, but the pinned set works everywhere.

## Stack

`spaCy` · `NLTK` · `PyTorch` · `Hugging Face Transformers` · `gensim` · `scikit-learn` ·
`pandas` · `matplotlib`

## Layout

```
.
├── requirements.txt
├── setup.sh
├── assignment-1-pos-and-ner/         # + assignment 1.pdf
│   ├── part-1-pos/                   # RoshanBharati_POS_01.ipynb + data/ + output/
│   └── part-2-ner/                   # RoshanBharati_NER_01.ipynb + data/ + output/
├── assignment-2-nlp-fundamentals/    # notebook + output/ + assignment 2.pdf
├── assignment-3-word-embeddings/     # 3 notebooks + handwritten notes + output/ + assignment 3.pdf
└── assignment-4-rnn-handwritten/     # scanned PDF + rnn_bptt.py
```
