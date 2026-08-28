# Assignment 03 — Word Embeddings

Question paper: [`assignment 3.pdf`](./assignment%203.pdf)

The question paper lists **five** tasks and instructs: *"you are free to pick any 3 assignments
including at least one Conceptual Assignment."*

## Tasks selected

| # | Task | Category | Status |
|---|---|---|---|
| 1 | Word Embeddings — Theory and Analysis | Conceptual | not selected |
| **2** | **Embedding Space Exploration** | **Conceptual** | ✅ |
| **3** | **Build a Word2Vec Model** | **Programming** | ✅ |
| 4 | Use Pre-trained Embeddings | Programming | not selected |
| **5** | **Contextualized vs Static Embeddings** | **Programming** | ✅ |

Three tasks, including one Conceptual (Task 2) — satisfying the instruction.

## Contents

| File | Description |
|---|---|
| `RoshanBharati_Embeddings_Task2_SpaceExploration.ipynb` | Task 2 notebook |
| `Task2_Handwritten_Notes.md` | Task 2 — condensed notes for the handwritten submission |
| `RoshanBharati_Embeddings_Task3_Word2Vec.ipynb` | Task 3 notebook |
| `RoshanBharati_Embeddings_Task5_ContextualVsStatic.ipynb` | Task 5 notebook |
| `output/` | CSVs, figures, and the trained vectors |

> ### ⚠️ Task 2 still needs a handwritten PDF
> Conceptual Assignments must be submitted as **handwritten notes, scanned and uploaded as PDF**.
> The notebook produces the analysis, numbers and figures; `Task2_Handwritten_Notes.md` condenses it
> into a structure suitable for copying out by hand. **Writing, scanning and uploading that PDF is
> still outstanding** — the notebook alone does not satisfy the requirement.

---

## Task 2 — Embedding Space Exploration

Cosine similarity, vector arithmetic, PCA/t-SNE visualisation and cluster interpretation over
**GloVe** `glove-wiki-gigaword-100` (400k words, 100d, 6B training tokens).

**Results**
- **Analogies: 6/8 correct at rank 1 (75%)** — including `rome − italy + nepal → kathmandu`
- **Cluster separation: 3.51×** (within-group similarity 0.515 vs between-group 0.147)
- **Country→capital offsets: mean cosine 0.850** to their average — a real direction in 100d space

**Three caveats the notebook demonstrates rather than asserts**
- **Antonyms score high.** `good`/`bad` beats `cat`/`dog`. Embeddings capture *relatedness*, not polarity.
- **The exclusion rule does the work.** Without excluding inputs, `king − man + woman` returns `king`.
- **Low dimensions inflate cosines.** At 5d `cat`/`dog` = 0.97 (*higher* than at 100d) while the rank
  of `queen` in the analogy collapses to 210. Pairwise similarity degrades quietly; relational
  structure does not.

---

## Task 3 — Build a Word2Vec Model

**Skip-Gram with Negative Sampling implemented from scratch** in PyTorch — only autograd and the
embedding lookup are borrowed. Trained on the NLTK **Brown corpus** (1.16M tokens).

**Implemented by hand:** tokenisation and vocabulary indexing · subsampling with
`P(keep) = √(t/f(w))` · dynamic-window pair generation · the `f(w)^0.75` noise distribution ·
the SGNS loss · the batched training loop.

**Results**
- Loss 4.16 → 2.49 over 5 epochs (~40 s on CPU)
- Related pairs mean cosine **0.85** vs unrelated **0.11**
- Coherent neighbourhoods (`school` → *college, graduate, schools, student, university*)

### Two honest findings

**A real defect, found and fixed.** The first version generated (centre, context) pairs from the
corpus flattened into one token stream, so pairs straddled sentence breaks. Restricting them to
sentence boundaries **discards ~30% of the training pairs and improves results anyway** — top-10
neighbour overlap with gensim rises from ~1.5 to ~2.1 out of 10, and `school` now shares 7 of its
top 10 neighbours with the reference model. Section 5.1 ablates this in-notebook and computes the
figures live, since both metrics move a little between runs; the direction of the gap is what
reproduces, not its exact size.

*(The ablation's analogy column moves the other way, but it is a 5-item sample and too noisy to
separate the variants — the notebook says so rather than cherry-picking it.)*

**Analogies are unreliable, and that is the data, not the code.** The reference `gensim` model misses
several of the same analogies on the same corpus. Brown is 1.16M tokens; the GloVe vectors in Task 2 used ~6B —
about 5,000× more. Nearest neighbours are a *local* property needing few examples; analogy arithmetic
is a *global* geometric property needing far more. A separate sweep confirmed the model is
data-limited rather than undertrained: 15 epochs lowers the loss (2.46 → 2.06) while leaving
neighbour overlap and analogy accuracy flat.

---

## Task 5 — Contextualized vs Static Embeddings

**GloVe** (static) against **BERT** `bert-base-uncased` (contextual), on word sense disambiguation,
sentence similarity and cost.

**Results**

| Experiment | GloVe | BERT |
|---|---|---|
| WSD — same-sense similarity | ≡ 1.0 (structurally impossible) | **0.798** |
| WSD — different-sense similarity | ≡ 1.0 | **0.462** (gap **+0.336**) |
| Word order (`dog chased cat` vs `cat chased dog`) | **exactly 1.000** | 0.996 |
| Speed (CPU) | ~670× faster | baseline |

**The key point, stated precisely:** GloVe does not perform *poorly* at word sense disambiguation —
it **cannot perform the task at all**. One word maps to one vector, so the similarity between two
occurrences of `bank` is 1.0 regardless of context. The metric is undefined, not bad.

**The counterweight:** BERT's word-order win is real but slim (0.996 vs 1.000), because mean-pooling
discards most of what self-attention computed. And BERT is ~670× slower per sentence on CPU. Static
embeddings also remain far more *interpretable* — one stable, cacheable, inspectable vector per word,
where "the BERT vector for `bank`" does not exist as an object at all.

---

## Running

```bash
../.venv/bin/jupyter lab
```

First run downloads GloVe (~128 MB), BERT (~440 MB) and the NLTK Brown corpus; all are cached.
Task 3 trains two models and takes roughly 2–3 minutes on CPU.
