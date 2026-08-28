# Assignment 03 · Task 2 — Embedding Space Exploration
## Notes for handwritten submission

**Course:** AIAC 536 — Natural Language Processing
**Author:** Roshan Bharati

> Answers follow the order the question paper asks them. Numbers are from
> `RoshanBharati_Embeddings_Task2_SpaceExploration.ipynb`.

**Setup:** GloVe `glove-wiki-gigaword-100` — 400,000 words, 100 dimensions, trained on 6B tokens.

---

## Q1. Cosine similarity computations between words

$$\cos(\mathbf{a},\mathbf{b}) = \frac{\mathbf{a}\cdot\mathbf{b}}{\|\mathbf{a}\|\,\|\mathbf{b}\|}$$

Measures the **angle** between vectors, ignoring magnitude. Range −1 to +1. Magnitude is discarded on
purpose: a word used 50 times and one used twice should be comparable if they mean the same thing.

| Pair | Cosine |
|---|---|
| `cat` / `dog` | 0.880 |
| **`good` / `bad`** | **0.770** |
| `king` / `queen` | 0.751 |
| `paris` / `france` | 0.748 |
| `king` / `monarch` | 0.698 |
| `cat` / `kitten` | 0.558 |
| `king` / `banana` | 0.161 |

Related pairs score 0.5–0.9, unrelated pairs below 0.2 — the metric separates them cleanly.

### ⚠ The antonym problem

`good`/`bad` (0.770) outscores `king`/`queen` and `paris`/`france`, despite meaning opposites.

**Why:** antonyms occur in near-identical contexts ("the food was ___"). Distributional methods define
meaning by context, so opposites become neighbours.

**Conclusion:** embeddings capture **relatedness, not similarity**, and encode no polarity. A sentiment
model built on cosine distance alone inherits this flaw.

---

## Q2. Vector arithmetic

$$\mathbf{v}_{king} - \mathbf{v}_{man} + \mathbf{v}_{woman} \approx \mathbf{v}_{queen}$$

Relations correspond to consistent **translations** in the space: strip the "male" component, add the
"female" component.

**Result: 6 of 8 analogies correct at rank 1 (75%).**

| Relation | Query | Expected | Got | Rank |
|---|---|---|---|---|
| gender | `king − man + woman` | queen | queen (0.783) | **1** |
| country → capital | `paris − france + japan` | tokyo | tokyo (0.892) | **1** |
| gendered profession | `actor − man + woman` | actress | actress (0.916) | **1** |
| low-resource capital | `rome − italy + nepal` | kathmandu | kathmandu (0.767) | **1** |
| comparative | `bigger − big + small` | smaller | *larger* | 2 |
| irregular past | `walked − walk + run` | ran | *went* | 2 |

### Two caveats

**(a) The exclusion rule does the work.** Without excluding the three input words, the nearest
neighbour of `king − man + woman` is **`king` itself**. The vector barely moves; the famous result
depends on forbidding the obvious answer by hand.

**(b) Failures are morphological, not frequency-based.** Nepal → Kathmandu succeeded despite Nepal
being rare. Both failures were inflectional, and both still ranked 2nd — the model found the right
*region* but the wrong member, because `larger`/`smaller` share contexts. The antonym problem again.

### Bias
`doctor − man + woman → nurse (0.776)`. The same geometry that gives `king → queen` reproduces
occupational stereotypes from the training corpus. This comes from the data, not the algorithm.
(Bolukbasi et al., 2016)

---

## Q3. Visualise embeddings using PCA or t-SNE

100 dimensions cannot be plotted, so project to 2. Both methods used, on 36 words in 6 groups
(Royalty, Animals, Countries, Capitals, Technology, Food).

| | PCA | t-SNE |
|---|---|---|
| Type | Linear projection | Non-linear manifold |
| Preserves | Global variance | Local neighbourhoods |
| Deterministic | Yes | No (seed-dependent) |
| Distances meaningful | Roughly | **Only locally** |

**PCA captured just 30.7% of variance** (PC1 17.1%, PC2 13.7%) — most structure is invisible in that
panel. t-SNE separates the groups far more crisply, **but its gaps between clusters are not to scale**
and must not be read as "these groups are this far apart."

*Figures to reproduce: `task2_embedding_projection.png` (6 coloured clusters, labelled),
`task2_country_capital.png` (6 near-parallel arrows).*

---

## Q4. Interpret the visualised clusters

**1. Groups separate without supervision.**

| Measure | Value |
|---|---|
| Mean within-group similarity | 0.515 |
| Mean between-group similarity | 0.147 |
| **Separation ratio** | **3.51×** |

No labels were ever supplied. The structure comes purely from co-occurrence statistics — this is the
empirical content of the **distributional hypothesis** (Firth: *"You shall know a word by the company
it keeps."*).

**2. Countries and capitals form adjacent but distinct clusters.** The `country → capital` offsets have
**mean cosine 0.850** to their average offset — they genuinely point the same way in the full 100d
space, not just in the projection. The relation is a real **direction**, which is exactly why the Q2
arithmetic works.

**3. `apple` sits in Food but is pulled toward Technology.** Wikipedia and Gigaword discuss the company
more than the fruit. **A single static vector must average over all senses of a word** — the limitation
that motivates contextual embeddings.

---

## Conclusion — three sentences

1. An embedding space has **geometry**: semantic categories occupy contiguous regions, and relations
   between words appear as consistent directions that transfer across pairs.
2. That geometry is learned **without supervision**, purely from co-occurrence.
3. Its two structural limits are that **antonyms are close** (they share contexts) and **every word gets
   exactly one vector** regardless of sense.

---

### References
- Mikolov et al. (2013), *Efficient Estimation of Word Representations in Vector Space*
- Pennington, Socher & Manning (2014), *GloVe: Global Vectors for Word Representation*
- Bolukbasi et al. (2016), *Man is to Computer Programmer as Woman is to Homemaker?*
