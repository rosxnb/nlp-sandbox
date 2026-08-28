# Assignment 01 — POS Extraction and Named Entity Recognition

Question paper: [`assignment 1.pdf`](./assignment%201.pdf)

**This question paper contains two separately-titled parts**, "Assignment 01: Part-of-Speech (POS)
Extraction" and "Assignment 02: Named Entity Recognition (NER)", on consecutive pages. Both are
answered here, one per subfolder.

| Part | Task | Notebook |
|---|---|---|
| [`part-1-pos/`](part-1-pos) | POS tagging — extract nouns and verbs | `RoshanBharati_POS_01.ipynb` |
| [`part-2-ner/`](part-2-ner) | Named entity recognition | `RoshanBharati_NER_01.ipynb` |

---

## Part 1 — POS Extraction

### Deliverables
| File | Description |
|---|---|
| `part-1-pos/RoshanBharati_POS_01.ipynb` | Solution notebook |
| `part-1-pos/data/news_article.txt` | Source article as plain text |
| `part-1-pos/output/RoshanBharati_POS_01.csv` | Extracted nouns and verbs |
| `part-1-pos/output/RoshanBharati_POS_01_summary.csv` | Frequency table by lemma |

**Article:** *The real cause of most wildfires? People* — NPR (text edition), retrieved 2026-08-28.
<https://text.npr.org/nx-s1-5944328>

**Approach:** tagging with **spaCy** (`en_core_web_sm`), cross-checked against **NLTK**'s averaged
perceptron tagger. Two filtering decisions are stated explicitly in the notebook:

- `PROPN` (proper nouns) count as nouns, distinguished by a `Category` column
- `AUX` (*is*, *was*, *have*) does **not** count as a verb — a `KEEP_AUX` flag toggles this

**Results:** 86 sentences, 1,485 tokens → 427 nouns (71.6%) and 169 verbs (28.4%);
161 unique noun lemmas, 59 unique verb lemmas.

---

## Part 2 — Named Entity Recognition

### Deliverables
| File | Description |
|---|---|
| `part-2-ner/RoshanBharati_NER_01.ipynb` | Solution notebook |
| `part-2-ner/data/news_article.txt` | Source article as plain text |
| `part-2-ner/output/RoshanBharati_NER_01.csv` | Extracted entities (`Entity`, `Entity_Type`, …) |
| `part-2-ner/output/RoshanBharati_NER_01_summary.csv` | Entity counts by category |
| `part-2-ner/output/RoshanBharati_NER_01_displacy.html` | *Bonus* — full article, entities highlighted |

**Article:** *Norway's King Harald V dies at 89 and his son becomes King Haakon VIII* — NPR
(text edition), retrieved 2026-08-28. <https://text.npr.org/nx-s1-5947778>

Chosen deliberately: a head-of-state obituary is unusually dense in `PERSON`, `GPE`, `NORP`, `ORG`
and `DATE` entities.

**Results:** 991 words → 131 entity mentions, 84 unique entities across 14 types.
Most common type `PERSON` (31 mentions); most-named entity *Harald* (17 mentions).

### Bonus work — both optional extensions completed

1. **`displacy` visualisation** — inline preview plus a full-article HTML render
2. **Two-model comparison** — spaCy `en_core_web_sm` vs `dslim/bert-base-NER`

#### The finding worth reading

The comparison surfaced a methodological trap. `aggregation_strategy="simple"` — the setting most
tutorials use for the Hugging Face NER pipeline — **silently shatters non-English names**: `Haakon`
came back as two entities (`Ha` and `##akon`), and `STAVANGER` split into `STAV` (LOC) + `##AN` (ORG).

| Aggregation | Entities | Fragments | Jaccard agreement with spaCy |
|---|---|---|---|
| `"simple"` | 31 | 7 | 39.8% |
| `"first"` | 27 | **0** | **55.9%** |

Run naively, the comparison would have "demonstrated" that BERT is bad at Norwegian person names — a
conclusion entirely produced by a decoding hyperparameter. Section 7.1 of the notebook documents this.

With that fixed, `LOC` and `MISC` counts come out **exactly equal** between the two models, and type
agreement on jointly-detected entities is **92.1%**. Of the three disagreements, BERT is correct on two
(`Ragnhild` and `Astrid` are people; spaCy tagged them `ORG` and `MISC`).

---

## Running

```bash
../.venv/bin/jupyter lab           # from either part's folder
```

The BERT model (~430 MB) downloads on first run of Part 2 and is cached thereafter.
