# Evaluation

**Audience:** hackathon judging panel and engineers locking the stack.  
**Product:** clinician / student pharmacology guide.  
**Scope of this document:** the **retrieval** bakeoff only. Generation heuristics in the clinician UI are separate; see [technical-report.md](technical-report.md). The lab’s product output is [`configs/winning.yaml`](../configs/winning.yaml).  
**Frozen at:** 2026-08-18T13:06:01Z (sequential bakeoff).  
**Product lock updated:** 2026-08-19 (title-prefix ingest).  
**Machine state:** [`artifacts/lock_winning/state.json`](../artifacts/lock_winning/state.json).  
**Sequential-freeze winner:** `7d630b335113` (hybrid BM25 + cross-encoder rerank, no title prefix).  
**Locked product run:** `4a2bf096b370` (same retriever; `section_aware` windows prefixed with `section_title`).

Related docs: [user-guide.md](user-guide.md) · [architecture.md](architecture.md) · [technical-report.md](technical-report.md).

This document is the full record of the bakeoff: protocol, gold policy, metrics, stage results, per-query evidence, and the defense of every ranking decision — in particular **why Precision@k on this set is mathematically capped**, **why the scores we did max out are the ones a medical RAG system actually needs**, and **why we locked title-prefix ingest rather than sibling-fill packing**.

---



## 0. Briefing (read this first)

We locked a retrieval stack by searching **one axis at a time** (store → embed → chunk → retrieval) on 20 StatPearls pharmacology questions, with 100 same-dump distractor articles in the index. We did **not** run the 392-combination cartesian product.

**Frozen stack**


| Knob     | Locked value                                                    |
| -------- | --------------------------------------------------------------- |
| Parser   | PyMuPDF / `ocr_fallback` (NXML uses the XML router)             |
| Chunk    | `section_aware`, 400 tokens, 12% overlap, max 520, **`prefix_section_title: true`** |
| Embed    | `BAAI/bge-small-en-v1.5` (Sentence Transformers, CUDA)          |
| Store    | Chroma (on-disk, cosine)                                        |
| Retrieve | Hybrid: dense + BM25, weights 0.7 / 0.3, RRF k=60, `fetch_k=20` |
| Rerank   | `cross-encoder/ms-marco-MiniLM-L-6-v2` on the fused top 20, CPU |
| Not locked | Sibling-fill packing; parent–child candidate expand           |


**Headline scores on the official 20-query set**


| Metric          | Sequential freeze `7d630b335113` | **Product lock `4a2bf096b370`** | What it means here                                          |
| --------------- | -------------------------------- | ------------------------------- | ----------------------------------------------------------- |
| **MRR**         | **1.000**                        | **1.000**                       | Every query’s first relevant chunk is at **rank 1**         |
| **Hit@5**       | **1.000**                        | **1.000**                       | Every query has at least one gold chunk in the top 5        |
| **nDCG@5**      | 0.870                            | **0.951**                       | Extra gold siblings of long headings rank better after prefix |
| **Precision@5** | 0.280                            | **0.350**                       | On average 1.75 gold chunks in the top 5 (ceiling **0.40**) |
| **Precision@1** | **1.000**                        | **1.000**                       | The single best number for “did we land the right section?” |


**The sentence judges should take away about P@k**

Precision@5 of 0.28 on the sequential freeze is **not** a weak ranker. On this gold set a *perfect* retriever cannot exceed **mean P@5 = 0.40**, and **11 of 20 queries cannot exceed P@5 = 0.20 at all**, because they have only one labeled chunk. After the post-freeze probes (§11) we locked **title-prefix ingest**, which raises P@5 to **0.35** (87.5% of the ceiling) and nDCG@5 to **0.951** without changing labels, k, or first-citation quality (still P@1 = MRR = Hit@5 = 1.0). Pushing P@5 toward 0.8 would still require **changing the labels or changing k**, not a new model.

**The sentence judges should take away about medical RAG**

These are the right scores for a pharmacology guide. The clinical job is “put the correct drug’s correct heading in slot 1 so a later generator cites it.” We did that on **every** query (MRR = P@1 = Hit@5 = 1.0), including after title-prefix. Web-search Precision@5 wants five relevant links; a monograph question usually has **one relevant box**. A P@5 of 0.20 on that question is a perfect first citation plus four slots gold does not fill — not an 80% miss. We did **not** chase the remaining 0.05 to the 0.40 ceiling by packing siblings after rank 1. That would look better on P@5 and would not prove the ranker found those windows. Full argument: §6 and §11.

```mermaid
flowchart LR
  q[Query] --> rank[Ranked top-k]
  gold[Gold chunk ids] --> score[Metrics]
  rank --> score
  score --> mrr[MRR / P@1 / Hit@5]
  score --> p5[P@5 capped by n_gold]
  score --> ndcg[nDCG@5]
```

---



## 1. What this evaluation is, and what it is not



### 1.1 Goal of the lab

Choose a **small, locked stack** (parser, chunker, embedder, vector store, retrieval mode) that later product code can load from `winning.yaml`. Ingest and retrieve must stay **document-agnostic** (PDF / TXT / MD / XML / NXML). Scoring is **StatPearls pharmacology only**.

### 1.2 What is scored

A run asks one question:

> Given a clinician-style drug question, did the retriever put the **labeled chunks** near the top of a ranked list?

Eval never rebuilds the index. It queries an ingest job that already exists (`artifacts/jobs/<job_id>/` with `chunks.json` + `report.json`). There is **no LLM**, no generated answer, no faithfulness / ROUGE / BLEU, and no “was the medical explanation correct?” score. Those belong to a later generation slice.

### 1.3 Two contracts

1. **Ingest / retrieve** work on any supported upload.
2. **Official eval corpus** = 20 gold StatPearls NXML named in the templates + 100 seeded distractors from the same dump.

We do not special-case StatPearls inside parsers or chunkers. The section-aware win is about **headings**, which many medical documents share.

---



## 2. Eval corpus (fixed before seeing results)


| Item              | Value                                               |
| ----------------- | --------------------------------------------------- |
| Templates         | `data/eval/statpearls_pharmacology_templates.jsonl` |
| Eval set id       | `statpearls_pharmacology`                           |
| Queries           | **20**                                              |
| Gold articles     | The **20** NXML files named in the templates        |
| Distractors       | **100** other StatPearls NXML from the same dump    |
| Sample seed       | `42` (reproducible distractor draw)                 |
| Indexed files     | **120**                                             |
| Winner index size | **2,053** chunks (`section_aware`)                  |
| k values          | 1, 3, 5, 10                                         |




### 2.1 Query design

Templates are balanced across the intents a pharmacology guide must retrieve:


| Intent                  | Count | IDs              |
| ----------------------- | ----- | ---------------- |
| Indications             | 3     | sp01, sp08, sp15 |
| Mechanism of action     | 3     | sp02, sp09, sp16 |
| Administration / dosing | 3     | sp03, sp10, sp17 |
| Adverse effects         | 3     | sp04, sp11, sp18 |
| Contraindications       | 3     | sp05, sp12, sp19 |
| Monitoring              | 3     | sp06, sp13, sp20 |
| Toxicity                | 2     | sp07, sp14       |


Each template stores `filename`, `section_title`, a verbatim `anchor` span from that section, and a clinician-style `query`. The anchor is **not** the only gold (see §3). It is the fail-closed check that the section was actually indexed.

### 2.2 Honesty limits (stated up front)

- n = 20 auto-labeled queries — enough to lock a hackathon stack, **not** a universal RAG champion.
- Templates lean on StatPearls section structure (“Indications”, “Contraindications”, …). That favors heading-aware chunking **on this corpus**.
- Distractors are same-domain StatPearls articles, not adversarial hard negatives.
- Binary relevance only (a chunk is gold or not). There is no graded “partially useful” label.
- OpenAI `text-embedding-3-small` was **not evaluated** (API key present, **429 insufficient_quota**). We do not claim it was compared.

---



## 3. Gold labeling

Chunk ids change when the chunker changes. Gold from one ingest job is therefore **never reused** on another. After every ingest, `label_from_chunks` recomputes gold from that job’s `chunks.json`.

### 3.1 Official policy (what the freeze used)

**Gold = every chunk whose** `(filename, section_title)` **matches the template, union any extra chunks that contain the anchor span.**

Recorded in freeze state as `gold_policy: "filename+section union anchor"`.

Rationale: a StatPearls “Administration” section that splits into six windows is still one clinical answer. Labeling only the window that happens to contain the anchor sentence would punish a ranker for retrieving the rest of the same section.

### 3.2 Why we changed the policy (lab history)

The first labeling pass was **anchor-substring only**. On the frozen `section_aware` index that produced **17/20 queries with a single gold chunk** (mean n_\text{gold} = 1.15). Under that policy a perfect rank-1 hit still scored

P@5 = \frac{1}{5} = 0.20

Observed P@5 sat at ~0.18–0.20 and looked like a retrieval failure. It was not. Sibling chunks from the same heading were being scored as misses.

We then switched to section-union gold (mean n_\text{gold} = 2.15) and **re-ran retrieval on the same index**. The locked stack did not change. P@5 moved 0.20 → 0.28 **because the labels grew**, not because the ranker suddenly found new articles.

Anchor-only ceiling on the winning index (computed, not hypothetical):


| k   | Max mean P@k under anchor-only gold |
| --- | ----------------------------------- |
| 1   | 1.00                                |
| 3   | 0.383                               |
| 5   | **0.230**                           |
| 10  | 0.115                               |


Section-union ceiling on the same index:


| k   | Max mean P@k under section gold |
| --- | ------------------------------- |
| 1   | 1.00                            |
| 3   | 0.583                           |
| 5   | **0.400**                       |
| 10  | 0.215                           |




### 3.3 Implementation notes

- Section match is case-insensitive on `section_title`.
- Anchor fallback: if the verbatim span is missing (window split), require every anchor token of length > 3 to appear.
- If a template still matches zero chunks, ingest/eval **fails closed** (`IngestError`). We do not silently drop queries.

---



## 4. Metrics — definitions, not slogans

Let R_q be the gold chunk-id set for query q, and \pi_q = (d_1,\ldots,d_m) the retrieved ranking. All aggregates are **macro means** over the 20 queries (unweighted). Latency is query-time p50 / p95 in milliseconds.

### 4.1 Precision@k

\mathrm{P@}k(q) = \frac{\lvert d_1,\ldots,d_k \cap R_q \rvert}{k}

(If fewer than k results are returned we divide by the actual length, matching `precision_at_k` in `eval/metrics.py`. Official runs always retrieve k \ge 10.)

**This is a set-size metric.** It does not care that the relevant hit is at rank 1 versus rank 5, except insofar as that hit occupies one of the k slots.

### 4.2 Hard ceiling (the inequality that decides P@k)

For a **single query**, the number of relevant documents you can place in the top k cannot exceed how many relevant documents exist:

\lvert d_1,\ldots,d_k \cap R_q \rvert \le \min\bigl(k,\ \lvert R_q \rvert\bigr)

Divide by k:

\mathrm{P@}k(q) \le \min\Bigl(1,\ \frac{\lvert R_q \rvert}{k}\Bigr)

Average over the evaluation set Q:

\overline{\mathrm{P@}k}
\le
\frac{1}{\lvert Q \rvert}
\sum_{q \in Q}
\min\Bigl(1,\ \frac{\lvert R_q \rvert}{k}\Bigr)

Call the right-hand side the **label ceiling**. No embedder, store, hybrid fusion, or reranker can beat it. The only ways to raise the ceiling are:

1. **Label more chunks as relevant** (change the task).
2. **Decrease k** (report P@1 or P@3 instead of P@5).
3. **Split the same section into more gold pieces** (change the chunker — this *inflates* the ceiling without proving better ranking).

A fourth non-option: “train harder.” That does not appear in the inequality.

### 4.3 Recall@k

\mathrm{R@}k(q) = \frac{\lvert d_1,\ldots,d_k \cap R_q \rvert}{\lvert R_q \rvert}
\quad (\text{0 if } R_q = \emptyset)

Recall *can* go to 1.0 even when P@k cannot: if \lvert R_q \rvert = 1 and that chunk is in the top 5, R@5 = 1 and P@5 = 0.2 simultaneously. That is not a contradiction.

### 4.4 Hit@k

\mathrm{Hit@}k(q) = \mathbf{1}\bigl[d_1,\ldots,d_k \cap R_q \ne \emptyset\bigr]

This is the “did we find the right section at all?” metric. Winner: **Hit@5 = 1.0**.

### 4.5 MRR (Mean Reciprocal Rank)

\mathrm{RR}(q) = \frac{1}{\mathrm{rank\ of\ first\ relevant}}
\quad (\text{0 if none retrieved})

\mathrm{MRR} = \frac{1}{\lvert Q \rvert}\sum_q \mathrm{RR}(q)

MRR = 1.0 means **every** query has a gold chunk at rank 1. That is the winner’s headline ranking result. It is compatible with a low P@5.

### 4.6 nDCG@k (binary relevance)

\mathrm{DCG@}k = \sum_{i=1}^{k} \frac{\mathrm{rel}_i}{\log_2(i+1)},
\qquad
\mathrm{rel}_i \in 0,1

IDCG packs \min(\lvert R_q \rvert, k) ones at the top of the list. nDCG = DCG / IDCG.

Unlike P@k, nDCG **rewards putting the relevant hit first**. For a singleton gold set, nDCG@5 = 1.0 as soon as rank-1 is correct — P@5 is still 0.2.

### 4.7 Predeclared winner rule

**Before looking at bakeoff numbers**, the sort key was locked as:

1. **MRR** (descending)
2. **nDCG@5** (descending)
3. **Precision@5** (descending)

Implemented as `score_key` in `src/clinical_rag/eval/protocol.py`. The Streamlit leaderboard uses the same key. We do not re-rank after seeing results.

**Why this order, given P@k’s ceiling**

- MRR answers the product question: “is the first citation the right section?”
- nDCG@5 still sees ranking quality among the extra gold siblings, without demanding five gold chunks that do not exist.
- P@5 is a **tie-break only**. Treating it as primary would reward chunkers that shatter sections into more labeled fragments (higher ceiling, worse first-hit). Hierarchical on this set is exactly that trap: P@5 **0.32** vs section_aware **0.27**, but MRR **0.58** vs **0.925**.

Index-axis stages (store / embed / chunk) probe with **dense retrieval only**, so hybrid and rerank cannot leak into those comparisons.

---



## 5. Why P@k was mathematically impossible to increase

This section is the defense.

### 5.1 Worked example (singleton gold)

**sp02** — *“What is the mechanism of action of Upadacitinib?”*

On the frozen `section_aware` index the whole *Mechanism of Action* heading fits in **one** chunk (`article-135218-section_aware-0005`). Gold size = **1**.

The winner places that chunk at **rank 1**. Rank 2–5 are other sections / other JAK-inhibitor articles — correctly *not* labeled gold.


| Metric | Value    | Ceiling  |
| ------ | -------- | -------- |
| P@1    | **1.00** | 1.00     |
| P@3    | 0.333    | 0.333    |
| P@5    | **0.20** | **0.20** |
| P@10   | 0.10     | 0.10     |
| R@5    | 1.00     | 1.00     |
| Hit@5  | 1.00     | 1.00     |
| MRR    | 1.00     | 1.00     |
| nDCG@5 | 1.00     | 1.00     |


There is **no remaining error** on this query except the metric’s own 1/k dilution. A “better” retriever has nothing left to put in slots 2–5 that the gold set will count.

**11 of 20 queries are this case** (gold size = 1). Each is hard-capped at P@5 = 0.20. Together they contribute **at most**

\frac{11 \times 0.20}{20} = 0.11

to the *mean* P@5. That 0.11 is locked forever unless we change labels or k.

### 5.2 Gold-size histogram on the frozen index

Recomputed from winner run `7d630b335113` (identical to `label_from_chunks` on job `cd3337fa4762`):


| \lvert R_q \rvert | Queries | Max P@5 for those queries |
| ----------------- | ------- | ------------------------- |
| 1                 | 11      | 0.20                      |
| 2                 | 3       | 0.40                      |
| 3                 | 3       | 0.60                      |
| 4                 | 1       | 0.80                      |
| 6                 | 1       | 1.00                      |
| 7                 | 1       | 1.00                      |


\min = 1,\quad
\mathrm{mean} = 2.15,\quad
\mathrm{median} = 1,\quad
\max = 7

18 / 20 queries have **fewer than 5** gold chunks, so they **cannot** reach P@5 = 1.0.

### 5.3 Ceiling arithmetic for mean P@5

\begin{aligned}
\overline{\mathrm{P@}5}_{\max}
&= \frac{
11\cdot\tfrac{1}{5}

- 3\cdot\tfrac{2}{5}
- 3\cdot\tfrac{3}{5}
- 1\cdot\tfrac{4}{5}
- 1\cdot 1
- 1\cdot 1
}{20} 
&= \frac{2.2 + 1.2 + 1.8 + 0.8 + 1.0 + 1.0}{20}
= \frac{8.0}{20}
= \mathbf{0.40}
\end{aligned}

Same identity for other k:


| k   | Label ceiling | Winner actual | Fraction of ceiling |
| --- | ------------- | ------------- | ------------------- |
| 1   | **1.000**     | **1.000**     | **100%**            |
| 3   | 0.583         | 0.450         | 77%                 |
| 5   | **0.400**     | **0.280**     | **70%**             |
| 10  | 0.215         | 0.165         | 77%                 |


Mean number of gold hits in the top 5: **1.40** (perfect packing would be **2.00**, because mean \min(\lvert R_q \rvert, 5) = 2.00).

**You cannot report P@5 ≈ 0.5, 0.6, or 0.8 on this set.** Those numbers lie above 0.40. Claiming them would require different gold.

### 5.4 Per-query winner scores vs ceiling


| ID   | n_\text{gold} | P@1 | P@5  | P@5 ceiling | At ceiling? | First relevant rank |
| ---- | ------------- | --- | ---- | ----------- | ----------- | ------------------- |
| sp01 | 2             | 1.0 | 0.40 | 0.40        | yes         | 1                   |
| sp02 | 1             | 1.0 | 0.20 | 0.20        | yes         | 1                   |
| sp03 | 6             | 1.0 | 0.60 | 1.00        | no (−3)     | 1                   |
| sp04 | 1             | 1.0 | 0.20 | 0.20        | yes         | 1                   |
| sp05 | 3             | 1.0 | 0.20 | 0.60        | no (−2)     | 1                   |
| sp06 | 1             | 1.0 | 0.20 | 0.20        | yes         | 1                   |
| sp07 | 1             | 1.0 | 0.20 | 0.20        | yes         | 1                   |
| sp08 | 1             | 1.0 | 0.20 | 0.20        | yes         | 1                   |
| sp09 | 3             | 1.0 | 0.20 | 0.60        | no (−2)     | 1                   |
| sp10 | 2             | 1.0 | 0.40 | 0.40        | yes         | 1                   |
| sp11 | 2             | 1.0 | 0.20 | 0.40        | no (−1)     | 1                   |
| sp12 | 1             | 1.0 | 0.20 | 0.20        | yes         | 1                   |
| sp13 | 1             | 1.0 | 0.20 | 0.20        | yes         | 1                   |
| sp14 | 1             | 1.0 | 0.20 | 0.20        | yes         | 1                   |
| sp15 | 7             | 1.0 | 0.40 | 1.00        | no (−5)     | 1                   |
| sp16 | 1             | 1.0 | 0.20 | 0.20        | yes         | 1                   |
| sp17 | 4             | 1.0 | 0.60 | 0.80        | no (−1)     | 1                   |
| sp18 | 3             | 1.0 | 0.40 | 0.60        | no (−1)     | 1                   |
| sp19 | 1             | 1.0 | 0.20 | 0.20        | yes         | 1                   |
| sp20 | 1             | 1.0 | 0.20 | 0.20        | yes         | 1                   |


**13 / 20 queries are already at their P@5 ceiling on the sequential freeze.** The entire remaining P@5 gap (0.28 → 0.40) lives in **seven** multi-chunk sections where sibling windows did not all enter the top 5. That gap is real, but it is **0.12 of mean P@5**, and closing it still stops at **0.40**. The 2026-08-19 probes (§11) closed most of it with title-prefix ingest (P@5 **0.35**), without packing.

The queries that still have unused gold on the freeze: sp03 (Adalimumab Administration, 6 windows), sp15 (Vitrectomy Indications, 7 windows), plus five sections with 2–4 windows. These are long headings split by the 400-token cap. Retrieving every sibling is a packing problem, not a “wrong article” problem: **rank 1 is already the right section on every query**.

### 5.5 Why “just improve the ranker” does not move mean P@5 much

Rerank’s job is to pull the first relevant hit to rank 1. It did that (MRR 0.925 → 1.0). On singleton queries, once rank-1 is correct, P@5 is frozen.

Compare dense vs hybrid vs winner on the **same** gold (stage 4, frozen index):


| Config                            | MRR       | P@5      |
| --------------------------------- | --------- | -------- |
| dense                             | 0.925     | 0.27     |
| hybrid/BM25 (no rerank)           | 0.917     | **0.29** |
| **hybrid/BM25 + rerank (winner)** | **1.000** | 0.28     |


Hybrid without rerank edges P@5 by 0.01 (one extra sibling on a couple of queries) and **loses** MRR. The predeclared key correctly keeps the reranked system. If we had optimized P@5 we would have shipped a worse first-citation ranker.

Rerank actually *dropped* P@5 on **sp05** (0.40 → 0.20) while fixing first-rank on **sp07, sp09, sp14, sp20**. That is the metric disagreeing with the product goal, and it is why P@5 is third on the key.

### 5.6 Why a different chunker is not a legitimate P@5 fix

Gold cardinality is a function of how finely the matched section is split. Same templates, different strategies:


| Strategy            | Chunks in index | Mean \lvert R_q \rvert | Queries with \lvert R_q \rvert=1 | **P@5 ceiling** | Dense P@5 (actual) | Dense MRR |
| ------------------- | --------------- | ---------------------- | -------------------------------- | --------------- | ------------------ | --------- |
| **section_aware**   | 2053            | 2.15                   | 11                               | **0.40**        | 0.27               | **0.925** |
| langchain_token     | 1570            | 1.85                   | 7                                | 0.37            | 0.22               | 0.416     |
| fixed               | 1811            | 2.70                   | 3                                | 0.52            | 0.27               | 0.568     |
| langchain_recursive | 2158            | 2.95                   | 3                                | 0.53            | 0.27               | 0.633     |
| langchain_markdown  | 2026            | 3.10                   | 2                                | 0.55            | 0.31               | 0.623     |
| hierarchical        | 2645            | 3.50                   | 2                                | **0.62**        | **0.32**           | 0.580     |
| semantic            | 2309            | 3.85                   | 7                                | 0.66            | 0.28               | 0.528     |


Hierarchical’s P@5 looks “best” because it **creates more gold pieces** (ceiling 0.62 vs 0.40), not because it ranks the right heading first. MRR exposes that. Semantic chunking has the highest ceiling of all (0.66) and still loses badly on ranking.

This is the second reason P@k cannot be chased: **inflating P@5 by shredding sections would be metric gaming**, and the protocol forbids it by sorting on MRR first.

### 5.7 What *would* raise P@k — and why we refused those levers


| Lever                             | Effect on P@5                                            | Why we did not pull it in this slice                                                         |
| --------------------------------- | -------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Report P@1                        | Already **1.0**                                          | We already report it; P@1 is the fair precision number                                       |
| Label the whole article           | Ceiling would go to 1.0 (gold articles have 8–27 chunks) | That calls “Continuing Education Activity” relevant to a MoA question. It is the wrong task. |
| Shrink k to 1                     | Mean P@1 = 1.0                                           | Already done; P@5 remains in the table because the grid asked for it                         |
| Section-sibling fill after rank-1 | Hits the 0.40 ceiling (run `5151fb0b0ee9`)               | Probed post-freeze (§11). Packs metadata after rank 1; not a ranking result. **Not locked.** |
| Prefix `section_title` on every window | 0.28 → **0.35**, nDCG@5 0.870 → **0.951**            | Probed post-freeze (§11). Makes later windows independently retrievable. **Locked.**         |
| Parent–child expand before rerank | 0.28 → 0.29                                              | Probed post-freeze (§11). Candidates enter the pool; CE still demotes heading-less leftovers. |
| Cartesian sweep of chunk size     | Might split/merge sections and *move the ceiling*        | Forbidden by sequential-freeze policy; would confound chunker identity with gold cardinality |
| New vector DB / embedder          | Cannot beat 0.40                                         | Inequality in §4.2                                                                           |




### 5.8 One-line judge rebuttal

> “Your Precision@5 is only 0.28.”  
> “P@5 ≤ |gold|/5. Eleven queries have one gold chunk, so they cap at 0.20. The set cap is 0.40. The sequential freeze hit 0.28; the product lock (title-prefix) hits **0.35**, still with P@1 = MRR = Hit@5 = 1.0. The ranker finds the right section first; Precision@5 is mostly counting empty slots that gold does not fill.”

---



## 6. Why these scores are the right scores for a medical RAG system

§5 is the inequality: P@5 *cannot* be large on this gold. This section is the product claim: **even if it could, we would not want it to be.** The numbers that sit at 1.0 are the ones a clinician / student guide is for.

### 6.1 What “good retrieval” means in this product

Med-Evidence is not a web search engine. It is a **grounded pharmacology lookup**: indication, mechanism, dose, adverse effects, contraindication, monitoring, toxicity. A later generator (not in this slice) will quote retrieved chunks and attach `document_name`, `section_title`, `page_number`, `chunk_id`.

The failure mode that matters is therefore:

> The user asks for Ampicillin **contraindications**. Slot 1 is Ampicillin **indications** (or another penicillin). The model writes a fluent answer with the wrong heading as its first citation.

That is a safety miss. The failure mode that **does not** matter in the same way:

> Slot 1 is the correct Contraindications chunk. Slots 2–5 are other sections of the same article, or a related drug’s similar heading. Precision@5 counts four “misses.”

On this eval set the winner has **zero** of the first failure and a Precision@5 that is almost entirely the second.

Map the clinical job onto the metrics we actually maxed:


| Clinical requirement                                                   | Metric that tests it | Sequential freeze     | **Product lock (title-prefix)** |
| ---------------------------------------------------------------------- | -------------------- | --------------------- | ------------------------------- |
| The first citation is the right drug’s right heading                   | **P@1**, **MRR**     | **1.000**             | **1.000**                       |
| The labeled section is inside the context we would send to a generator | **Hit@5**            | **1.000**             | **1.000**                       |
| Extra gold windows of a long heading are reasonably packed             | nDCG@5               | 0.870                 | **0.951**                       |
| A large fraction of a *five-slot* list is gold                         | P@5                  | 0.280 (ceiling 0.400) | **0.350** (ceiling 0.400)       |


P@1 / MRR / Hit@5 are **at the maximum possible value**. There is no remaining first-citation error on the official set. Calling the freeze “low quality” because P@5 is 0.28 is using the wrong proxy for the job. Title-prefix raises the packing metrics without touching those 1.0s; sibling-fill packing to 0.40 was probed and refused (§11).

### 6.2 Drug questions usually have one relevant box, not five relevant documents

TREC-style Precision@k assumes a topic with many relevant documents (news wires, web pages). You hope several of the ten blue links are on-topic.

A StatPearls pharmacology question is the opposite. “What are the contraindications for Acetazolamide?” has **one** labeled heading. On the frozen chunker that heading fits in **one** 400-token window for 11 of 20 queries. The clinically perfect retrieval is then:

1. Rank 1 = that window (citable, complete enough to answer).
2. Ranks 2–5 = whatever else the fusion returned — **not gold**, and they should not be.

That ranking scores **P@5 = 0.20** and **P@1 = MRR = Hit@5 = nDCG@5 = 1.0**. Those four 1.0s are the medical result. The 0.20 is 1/k dilution of a perfect hit. Reporting only P@5 makes a **perfect monograph lookup look like an 80% failure**.

This is not a quirk of our labels. It is how drug monographs are written: one heading per intent. A retriever that *needs* five gold chunks to look good on P@5 is a retriever that needs the corpus to be shredded or over-labeled. Neither is how a clinician reads a monograph.

### 6.3 High P@5 would be the wrong objective for grounding

Grounded generation will condition on the top few chunks. What we want in that window:

- **Slot 1 is the answer section.** Already true (P@1 = 1.0).
- **We do not promote the wrong heading of the same drug.** Indications must not outrank Contraindications on a contraindication query. Section-aware gold plus MRR = 1.0 is exactly that discipline.
- **We do not promote another drug’s matching heading as if it were gold.** On sp02 (Upadacitinib MoA), ranks 3–4 were Baricitinib MoA and Adalimumab MoA. Precision@5 correctly scores those as misses. If we had labeled “any Mechanism of Action section” as relevant, P@5 would rise and the **task would become clinically wrong** — a JAK-inhibitor neighbour is not an acceptable citation for Upadacitinib.

Chasing a large P@5 therefore pulls in two bad policies:

1. **Whole-article gold** — Continuing Education Activity, team-outcomes blurbs, and Indications all count on a Toxicity question. The metric inflates; the first citation is allowed to be the wrong box.
2. **Split the right section into five windows and retrieve all five** — more tokens, more overlap, more chance a generator mixes adjacent headings, no new clinical fact if the heading already fit in one chunk.

A medical RAG stack should look **precise at rank 1**, not **busy at rank 5**. Ours does.

### 6.4 Safety-critical intents are the ones we maxed

The eval set is not trivia. It includes contraindications, toxicity, monitoring, and dosing — the intents where a wrong first chunk is the dangerous one.

Rerank’s entire gain (MRR 0.925 → 1.0) was moving first-relevant to rank 1 on queries such as **Candesartan toxicity**, **Metoclopramide toxicity**, **Chloroprocaine MoA**, and **Amitriptyline monitoring** (§10). Those are exactly the queries where “pretty good, relevant is at rank 2” is not good enough for a generator that trusts slot 1.

After that move:

- Every toxicity / contraindication / monitoring / dosing question in the set has gold at **rank 1**.
- Hit@5 = 1.0, so none of those sections fall out of a 5-chunk context window.
- The leftover P@5 gap after the freeze is unused *sibling windows of the already-correct heading* on long Administration / Indications articles (Adalimumab dosing, Vitrectomy indications, …) — completeness of a long box, not “we retrieved the wrong drug.” Title-prefix closed most of that gap by ranking (§11); we did not pack the rest.

For a medical system, **never missing the box, always showing it first** is the definition of a retrieval pass. We have that. Filling every empty P@5 slot is not.

### 6.5 What a generator would actually see

This slice does not generate. When it does, the default context is top-k with k small (the frozen config retrieves 10; a prompt will not paste all 10). Operating point:


| If the prompt uses | Sequential freeze                                                      | Product lock (title-prefix)                                            |
| ------------------ | ---------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| Top 1              | The gold section, every query                                          | Same                                                                   |
| Top 3              | Gold in the window every query; mean P@3 = 0.45 vs ceiling 0.58        | Gold in the window every query; mean P@3 = **0.53** vs ceiling 0.58    |
| Top 5              | Gold in the window every query; mean 1.4 gold chunks of a possible 2.0 | Mean **1.75** gold chunks of a possible 2.0                            |


So the “low” P@5 is unused slots **after** the answer is already in hand. Freeze nDCG@5 = 0.87 was the ranking-quality remainder among extra gold siblings; title-prefix raised it to 0.95. That remainder is useful, not the safety metric.

### 6.6 How to read the scorecard in a demo


| If someone says           | Read it as                                                                                                                |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| “P@5 is only 0.28”        | Web-search reflex. Ask P@1 and MRR. Both are 1.0.                                                                         |
| “That’s 28% accuracy”     | False. 0.28 is gold-in-five-slots / 5, not “correct vs incorrect questions.” Hit@5 = 1.0 means **20/20 questions found.** |
| “Can you get P@5 to 80%?” | Not on this gold (§5). Also not the medical target (§6.3).                                                                |
| “Is retrieval done?”      | First-citation retrieval on this set is done. Generation, citation UX, and a harder / larger gold set are the next slice. |


---



## 7. Candidate grid and sequential freeze



### 7.1 Locked candidates (YAGNI)


| Axis      | Candidates                                                                                                                                            |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Store     | `chroma`, `qdrant` (both embedded on disk, cosine)                                                                                                    |
| Embed     | MiniLM-L6-v2, mpnet-base-v2, `BAAI/bge-small-en-v1.5`, `text-embedding-3-small`                                                                       |
| Chunk     | native `{section_aware, fixed, hierarchical}`, LangChain splitters only `{langchain_recursive, langchain_token, langchain_markdown}`, plus `semantic` |
| Retrieval | dense; sparse BM25; sparse TF-IDF; hybrid BM25; hybrid TF-IDF; hybrid BM25+rerank; hybrid TF-IDF+rerank                                               |


Held constant: `target_tokens=400`, `overlap_ratio=0.12`, `max_tokens=520`, hierarchical parents/children 800/350. Parser: PyMuPDF + `ocr_fallback`. Hybrid weights 0.7 / 0.3, RRF k=60, `fetch_k=20`, rerank top 20.

Off-grid on purpose (not “we didn’t get to them”): FAISS, Weaviate, Pinecone, Milvus, Cohere, LlamaParse, Unstructured, LangChain chains / LCEL / retrievers.

**Full cartesian:** 2 \times 4 \times 7 \times 7 = 392.  
**What we ran:** sequential freeze ≈ **20 scored probes** (plus one ingest per index candidate), not 392.

### 7.2 Why sequential, not cartesian

A 392-run product is time, GPU, and OpenAI quota we did not have, and it would still leave chunk-size × weight interactions unsearched. The protocol is: **freeze one axis, hold the rest, never go back**. Interaction risk is accepted and stated (stage 1–2 hold `section_aware` by design; stage 3 is where all seven chunkers compete).

### 7.3 Protocol

```text
Corpus (20 gold + 100 distractors, seed 42)
        │
        ▼
 Stage 1  Store: chroma vs qdrant
          hold: section_aware + bge-small · probe: dense
        │ lock winning store
        ▼
 Stage 2  Embed: MiniLM / mpnet / bge-small / OpenAI-small
          hold: winning store + section_aware · probe: dense
        │ lock winning embed
        ▼
 Stage 3  Chunk: all 7 strategies
          hold: winning store + embed · probe: dense
        │ lock winning chunk (+ that job’s index)
        ▼
 Stage 4  Retrieval: 7 configs on that one index
        │
        ▼
 configs/winning.yaml
```

Resume-safe progress: `artifacts/lock_winning/state.json`.  
Official entrypoint: `uv run python scripts/lock_winning_combo.py` (bakeoff only; re-running overwrites the title-prefix product lock).  
Product lock: `configs/winning.yaml` with `prefix_section_title: true` (2026-08-19).   
CUDA is **required** (the lock script refuses a silent CPU fallback). Hardware for the official run: **NVIDIA RTX 3050 Ti**.

The operator Streamlit app calls the same ingest, label, retrieve, and metric functions. It is not a second eval.

---



## 8. Retrieval mechanics (what stage 4 actually ran)

All of this lives in `src/clinical_rag/retrieval/`. One function (`run_retrieve`) serves CLI, UI, and freeze.

### 8.1 Dense

Query is embedded with the **same model id** stored on the collection (fail closed on mismatch). Store returns cosine top-k.

### 8.2 Sparse

In-process over `chunks.json` (not a second database):

- **BM25Okapi** with Robertson–Walker IDF, k_1 = 1.5, b = 0.75.
- **TF-IDF + cosine** (l2-normalized).

Tokenization: Unicode word characters, lowercased.

### 8.3 Hybrid

Fetch `fetch_k=20` from dense and from sparse, fuse with **weighted Reciprocal Rank Fusion**:

\mathrm{score}(d)
= \sum_i \frac{w_i}{k_{\mathrm{rrf}} + \mathrm{rank}_i(d)}

w = (0.7,\ 0.3), k_{\mathrm{rrf}} = 60. A document missing from one list contributes nothing from that list. Weights must sum to 1.0 (schema validator, fail closed).

### 8.4 Rerank

Cross-encoder over the fused top `rerank_top_n=20`. Ranking uses raw logits; displayed scores are \sigma(\mathrm{logit}). Default device is **CPU**: sharing the 4 GiB GPU with the query embedder OOMs. That CPU choice is why winner p50 is ~1 s vs dense ~25 ms — quality, not a store problem.

---



## 9. Findings (official freeze)



### Stage 1 — Store (dense probe, section_aware + bge-small)


| Store      | MRR       | nDCG@5     | P@5      | Hit@5 | p50 ms   |
| ---------- | --------- | ---------- | -------- | ----- | -------- |
| **chroma** | **0.925** | **0.8057** | **0.27** | 1.0   | **25.9** |
| qdrant     | 0.925     | 0.8057     | 0.27     | 1.0   | 159.6    |


**Winner: chroma.** Quality tied to four decimals; chroma was ~6× faster on this machine (embedded Qdrant `path=` vs Chroma persistent client). Quality-tied → latency decides. Chroma became the hold for later stages.

### Stage 2 — Embed (dense probe, chroma + section_aware)


| Model                      | MRR                                | nDCG@5     | P@5      | Hit@5 | p50 ms |
| -------------------------- | ---------------------------------- | ---------- | -------- | ----- | ------ |
| **BAAI/bge-small-en-v1.5** | **0.925**                          | **0.8057** | **0.27** | 1.0   | 23.5   |
| all-mpnet-base-v2          | 0.8083                             | 0.6816     | 0.23     | 1.0   | 24.8   |
| all-MiniLM-L6-v2           | 0.6692                             | 0.6151     | 0.23     | 0.9   | 23.5   |
| text-embedding-3-small     | *skipped — 429 insufficient_quota* |            |          |       |        |


**Winner: bge-small.** Clear MRR / nDCG margin. MiniLM is the only embed that dropped Hit@5 below 1.0. OpenAI remains unevaluated; we do not rank it.

### Stage 3 — Chunk (dense probe, chroma + bge-small)


| Strategy            | Chunks | MRR       | nDCG@5     | P@5      | Hit@5 |
| ------------------- | ------ | --------- | ---------- | -------- | ----- |
| **section_aware**   | 2053   | **0.925** | **0.8057** | 0.27     | 1.0   |
| langchain_recursive | 2158   | 0.633     | 0.481      | 0.27     | 0.9   |
| langchain_markdown  | 2026   | 0.623     | 0.487      | 0.31     | 0.95  |
| hierarchical        | 2645   | 0.580     | 0.480      | **0.32** | 1.0   |
| fixed               | 1811   | 0.568     | 0.480      | 0.27     | 0.95  |
| semantic            | 2309   | 0.528     | 0.384      | 0.28     | 0.85  |
| langchain_token     | 1570   | 0.416     | 0.383      | 0.22     | 0.7   |


**Winner: section_aware.** Large MRR / nDCG gap. Hierarchical “wins” P@5 and loses the score key — see §5.6.

Interpretation: StatPearls articles are heading-driven; gold is section-aligned. Geometry-only splitters (token / recursive / fixed) and embedding breakpoints (semantic, extra MiniLM pass at ingest, 95th-percentile cosine drop) do not preserve that structure. Semantic chunking is also the only strategy that **confounds stage 2** if it reused the job embedder; it does not — breakpoints are always MiniLM, by design.

### Stage 4 — Retrieval (frozen section_aware + bge-small + chroma index)


| Config                 | MRR     | nDCG@5     | P@5      | Hit@5 | p50 ms |
| ---------------------- | ------- | ---------- | -------- | ----- | ------ |
| **hybrid/bm25+rerank** | **1.0** | **0.8701** | 0.28     | 1.0   | ~1047  |
| hybrid/tfidf+rerank    | **1.0** | **0.8701** | 0.28     | 1.0   | ~960   |
| dense                  | 0.925   | 0.8057     | 0.27     | 1.0   | ~25    |
| hybrid/bm25            | 0.917   | 0.813      | **0.29** | 1.0   | ~47    |
| hybrid/tfidf           | 0.733   | 0.694      | **0.29** | 1.0   | ~39    |
| sparse/bm25            | 0.629   | 0.569      | 0.24     | 0.9   | ~20    |
| sparse/tfidf           | 0.315   | 0.244      | 0.11     | 0.5   | ~12    |


**Winner: hybrid/BM25 + MiniLM cross-encoder.** Tie on the score key with TF-IDF+rerank; BM25 is the freeze because it is the primary sparse axis (keyword matching on drug names / section language) and BM25-without-rerank already dominates TF-IDF-without-rerank (MRR 0.917 vs 0.733).

Rerank lifts MRR 0.925 → **1.0** and nDCG@5 0.806 → **0.870**, at ~40× the p50 of dense. Sparse alone is weak (TF-IDF Hit@5 = 0.5). Hybrid without rerank is already close to dense; the cross-encoder is what finishes first-rank.

---



## 10. What rerank actually fixed (per-query)

Dense MRR 0.925 is exactly 17 queries at rank 1 and 3 queries at rank 2:

\frac{17\cdot 1 + 3\cdot 0.5}{20} = 0.925

On the dense / hybrid-no-rerank runs, first-relevant was **not** rank 1 for a subset of: sp07 (Candesartan toxicity), sp09 (Chloroprocaine MoA), sp14 (Metoclopramide toxicity), sp20 (Amitriptyline monitoring). The cross-encoder moved **all 20** to rank 1.

That is the entire MRR story. There is no leftover first-hit error to spend more models on.

nDCG@5 of 0.870 (not 1.0) on the freeze is the sibling-packing remainder on the seven multi-gold queries in §5.4. That was the only honest remaining quality gap on this set, and it is **bounded**. Title-prefix ingest later raised nDCG@5 to 0.951 without changing first-citation scores (§11).

---



---



## 11. Post-freeze sibling-gap probes (2026-08-19)

The sequential freeze (§9) left a **0.12 mean P@5 gap** to the 0.40 ceiling. That gap is not first-citation error — every query already had gold at rank 1. It is leftover windows of seven long headings whose later 400-token slices no longer contain the heading string, so BM25 and the MiniLM cross-encoder prefer Continuing Education Activity and neighbouring headings.

We probed three ways to close that gap **without changing gold, k, or the official 7-config retrieval grid**. `ChunkConfig.prefix_section_title`, `RetrievalConfig.sibling_fill`, and `RetrievalConfig.parent_child` all default **off**, so a re-run of `scripts/lock_winning_combo.py` still measures the bakeoff, not these probes.

### 11.1 Leaderboard (same 20 queries, same score key)

All four runs keep **MRR = P@1 = Hit@5 = 1.0**.


| Run            | What changed                         | Index                                                                 | P@5      | nDCG@5    | Verdict                                      |
| -------------- | ------------------------------------ | --------------------------------------------------------------------- | -------- | --------- | -------------------------------------------- |
| `7d630b335113` | Sequential freeze winner             | job `cd3337fa4762` (no prefix)                                        | 0.28     | 0.870     | Bakeoff baseline                             |
| `28aa2262af4a` | `parent_child` expand before CE      | same frozen index                                                     | 0.29     | 0.879     | Ranking barely moved                         |
| `4a2bf096b370` | `prefix_section_title` at ingest     | job `ae69f99b47b7` (`corpus_id` `statpearls_pharmacology_titleprefix`) | **0.35** | **0.951** | **Product lock**                             |
| `5151fb0b0ee9` | `sibling_fill` after rank 1          | same frozen index                                                     | 0.40     | 1.000     | Ceiling by packing; **not locked**           |


### 11.2 Sibling fill (`RetrievalConfig.sibling_fill`)

**Mechanism.** Freeze rank 1. Pack remaining slots with other chunks that share `(document_name, section_title)`. Gold is filename+section union, so every packed sibling is labeled gold. Off unless the flag is set; not on the official 7-config grid.

**Result.** Run `5151fb0b0ee9` on the frozen index: P@5 = **0.40** (the label ceiling) and nDCG@5 = **1.0**.

**Why we did not lock it.** This is packing, not ranking. Rank 1 is already gold; the rest is metadata-driven stuffing of the same heading. That is the policy §6.3 already refused: “split the right section into five windows and retrieve all five.” A generator would see more of a long Administration box, but the leaderboard would no longer measure whether BM25 or the cross-encoder found those windows. Hitting the ceiling this way would look like a solved retrieval problem. It is not.

### 11.3 Parent–child expand (`RetrievalConfig.parent_child`)

**Mechanism.** Inject other windows of each retrieved parent into the **candidate pool before** the cross-encoder. Rank 1 is not frozen. Hierarchical jobs group by `parent_chunk_id`; `section_aware` falls back to `(document_name, section_title)`. Off unless the flag is set.

**Result.** Run `28aa2262af4a` on the frozen index: P@5 0.28 → **0.29**, nDCG@5 0.870 → **0.879**. Leftover windows enter the pool; MiniLM still demotes them relative to Continuing Education Activity because their **text** still lacks the heading.

**Why we did not lock it.** Exposure without a lexical cue is not enough. The cross-encoder continues to prefer neighbouring headings that actually contain the query’s section language.

### 11.4 Title-prefix ingest (`ChunkConfig.prefix_section_title`) — locked

**Mechanism.** At `stamp_chunk`, prepend `section_title` plus a newline when the window does not already start with that heading. The first window of a section already starts with the heading; later packed windows of the same heading usually do not. Prefix makes every sibling independently matchable by BM25 and independently scoreable by the cross-encoder. Same 2,053 chunks. Ingest used a separate `corpus_id` (`statpearls_pharmacology_titleprefix`) so Chroma did not replace the freeze collection.

**Result.** Run `4a2bf096b370`: hybrid BM25 + MiniLM CE, **no** sibling fill, **no** parent-child. P@5 = **0.35** (87.5% of the 0.40 ceiling), nDCG@5 = **0.951**, P@3 = 0.533. First-citation scores stay at 1.0.

### 11.5 Why we locked title-prefix

1. **It is a ranking fix.** Later windows contain the heading, so BM25 and the cross-encoder can score them. Sibling fill skips ranking. Parent-child only exposes candidates and leaves heading-less text for the CE to reject.
2. **It does not freeze rank 1.** Slots 2–5 still have to beat Continuing Education Activity on the fused + reranked list. That is the same discipline as the sequential freeze.
3. **It does not change gold cardinality.** Same section-union labels, same 0.40 ceiling. We did not shred sections or label whole articles.
4. **It is document-agnostic.** Any headed PDF / NXML / Markdown gets the same stamp. No StatPearls DTD special case, which keeps the ingest contract in §1.3.
5. **It leaves 0.05 of mean P@5 on the table honestly.** The remaining unused gold is still leftover windows of very long Administration / Indications sections. Closing that last 0.05 with sibling fill would be packing. We would rather a generator see a slightly incomplete long box than a leaderboard that pretends those windows were retrieved.
6. **The schema default stays false.** Stage 3 of the sequential freeze must not grow a hidden prefix confound. Product ingest reads `winning.yaml`; the operator Streamlit app does the same.

`configs/winning.yaml` is the product lock: `prefix_section_title: true`, `sibling_fill: false`, `parent_child: false`. `artifacts/lock_winning/state.json` remains the 2026-08-18 bakeoff log (winner `7d630b335113` on job `cd3337fa4762`). Do not re-run `scripts/lock_winning_combo.py` unless you intend to reset the product file to the bakeoff (prefix off).

---



## 12. Frozen stack — recommendation for product

Load `[configs/winning.yaml](../configs/winning.yaml)`. Later product code should read **only** that file. That YAML is the **2026-08-19 product lock** (title-prefix on). The sequential-freeze machine log in `artifacts/lock_winning/state.json` still points at bakeoff run `7d630b335113` (prefix off); do not treat those two as the same artifact.

### 12.1 Recommendations

1. **Ship this stack** for the clinician/student guide retrieval path until a larger labeled set says otherwise.
2. **Keep ingest generic.** Do not special-case StatPearls in parsers or chunkers.
3. **Prefer BM25 hybrid + MiniLM CE rerank** when first-citation quality matters. Offer dense-only (~25 ms, MRR 0.925) or hybrid-without-rerank (~47 ms, MRR 0.917) if a latency budget forbids ~1 s CPU rerank.
4. **Quote P@1, MRR, and Hit@k** when talking quality. Quote P@5 only **with its 0.40 ceiling**, and only after saying it is the wrong primary proxy for a monograph lookup (§6).
5. **Do not reintroduce** Weaviate / Pinecone / FAISS / Cohere / LlamaParse / Unstructured for this lab.
6. **Re-run stage 2** when OpenAI quota exists if a fair API-embed comparison is required; until then do not claim OpenAI was evaluated.
7. **Do not run the 392 cartesian** under current protocol. If chunk × embed interactions matter later, run a *small* targeted follow-up.
8. **Ship title-prefix ingest** from `winning.yaml` (`prefix_section_title: true`). Do **not** turn on sibling fill to print P@5 = 0.40; that is packing, not ranking (§11).
9. **Next product slice:** clinician query UI + grounded generation + citation UX — not another retrieval bakeoff. Optional engine work: make the frozen hybrid+rerank path faster (GPU rerank without OOM, cached sparse index).



### 12.2 Non-recommendations (this corpus)

- Switching to Qdrant for quality (tied, slower here).
- LangChain recursive / token / markdown or embedding-semantic as the default chunker.
- Sparse-only retrieval.
- Optimizing Precision@5 as the primary objective.
- Expanding chunk-size / hybrid-weight sweeps before generation quality is measurable.
- Labeling whole articles as relevant in order to publish a larger P@5.
- Sibling-fill packing after rank 1, even though it hits the 0.40 P@5 ceiling (§11).
- Parent–child candidate expand as a substitute for putting the heading on later windows.

---



## 13. Anticipated panel questions

**“Why didn’t you sweep everything?”**  
Because 392 ingest+eval runs are not a methodology; they are a brute-force substitute for one. Sequential freeze with a predeclared key is the experiment. Stage 1–2 holding `section_aware` is disclosed, not hidden: stage 3 still competed all seven chunkers.

**“Chroma over Qdrant — is that production-ready?”**  
On this 2k-chunk index they returned **identical** rankings. We locked the faster local store. A later scale-out can re-run stage 1 without reopening chunk/embed.

**“bge-small beat mpnet?”**  
Yes, on this 20-query dense probe: MRR 0.925 vs 0.808. We are not claiming a general embedding leaderboard. We are claiming a winner **on this freeze**.

**“Why is Precision@5 so low?”**  
It is not low relative to the label ceiling (§5), and it is not the medical score (§6). P@1 = MRR = Hit@5 = 1.0: every question finds the right heading, and that heading is slot 1. The product lock is P@5 = **0.35** of a 0.40 ceiling.

**“0.28 means the system is wrong 72% of the time.”**  
No. Precision@5 is not accuracy. Hit@5 = 1.0 means **20/20 questions retrieved gold inside the top 5**. Freeze 0.28 means “on average 1.4 of 5 slots are labeled gold”; product lock 0.35 means 1.75 of 5. Both are what you get when most answers are a single section window.

**“Sibling fill hit 0.40. Why lock 0.35?”**  
Because 0.40 was packing after rank 1, not retrieval. Title-prefix makes later windows contain the heading so BM25 and the cross-encoder can score them. That is the only probe that moved nDCG@5 a lot (0.87 → 0.95) without freezing rank 1. Full argument: §11.

**“Why not parent–child instead of rewriting chunk text?”**  
We tried it (`28aa2262af4a`). P@5 moved 0.28 → 0.29. The leftover windows entered the candidate pool; the cross-encoder still preferred Continuing Education Activity because the window text still lacked the heading.

**“Could you get Precision@5 to 0.8 with more training?”**  
No. 0.8 > 0.40. That number is not in the feasible set. It is also the wrong target: inflating gold to whole articles would reward citing Indications on a Contraindications query.

**“Hierarchical had higher P@5. Why discard it?”**  
Because we predeclared MRR first. Hierarchical’s P@5 is a **larger gold cardinality** (ceiling 0.62 vs 0.40) plus worse first-hit ranking (MRR 0.58). Using P@5 as primary would have selected a worse retriever.

**“Is n = 20 enough?”**  
Enough to freeze a hackathon stack with a documented ceiling. Not enough to claim a universal medical RAG champion. Stated in §2.2.

**“Did you evaluate generation / hallucinations / citations in the answer text?”**  
No. This slice has no generator. Retrieval citations (`document_name`, `section_title`, `page_number`, `chunk_id`) are stored on every chunk and returned on every hit. Grounded generation is the next slice.

**“Did OpenAI embeddings lose?”**  
They were **not run**. Quota error. Fail closed; skip with a printed reason.

**“Isn’t section-aware overfitting to StatPearls headings?”**  
The *eval set* is heading-heavy. The *chunker* is generic: split on extracted headings, pack to 400 tokens. Many guidelines and textbooks have headings. We do not encode StatPearls DTD logic into the chunker. A PDF-heavy gold set could reorder stage 3; we have not claimed otherwise.

**“Why CPU rerank at ~1 second?”**  
4 GiB GPU already holds the query embedder. CE on GPU OOMs. Quality was prioritized over p50 for the freeze. Latency is an engine problem, not a ranking problem.

---



## 14. Threats to validity


| Threat                                  | Mitigation / residual                                                                |
| --------------------------------------- | ------------------------------------------------------------------------------------ |
| Small n                                 | Predeclared protocol; per-query tables published; do not over-generalize             |
| Auto-labeled gold                       | Anchor fail-closed + section union; still not clinician-adjudicated                  |
| Gold cardinality depends on chunker     | Score key puts MRR first; §5.6 table shows the ceiling shift                         |
| Sequential freeze misses interactions   | Disclosed; targeted follow-up only if generation eval demands it                     |
| Same-domain distractors                 | Harder than empty index, easier than adversarial negatives                           |
| Template language echoes section titles | Favors BM25; that is why sparse is on the grid rather than assumed                   |
| Hardware-specific store latency         | Quality tied; we would still freeze chroma on quality, Qdrant only if it won metrics |
| OpenAI hole                             | Explicit skip, not a silent MiniLM substitution                                      |
| Binary nDCG                             | No graded relevance; IDCG assumes all gold equally useful                            |
| Title-prefix changes stored chunk text  | Same gold policy; token counts shift slightly; stage-3 default remains prefix off    |


---



## 15. How to reproduce

```bash
# Unit / integration (no CUDA freeze)
uv sync --extra dev
uv run pytest

# Official sequential freeze (needs CUDA + StatPearls dump symlink).
# Re-running this overwrites configs/winning.yaml with prefix_section_title
# off (bakeoff default). Do not run it unless you intend to reset the 2026-08-19
# product lock.
uv run python scripts/lock_winning_combo.py

# Operator UI (leaderboard sorted by the same score key)
uv run streamlit run src/clinical_rag/ui/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Corpus path: symlink `data/pharmacology_data` → local StatPearls dump (e.g. `statpearls_NBK430685/`). Gold templates are committed; NXML is not.

Artifacts (local, gitignored):

- Per-run metrics: `artifacts/evals/<run_id>/metrics.json`
- Sequential freeze winner: `artifacts/evals/7d630b335113/`
- Parent–child probe: `artifacts/evals/28aa2262af4a/`
- Title-prefix product lock: `artifacts/evals/4a2bf096b370/`
- Sibling-fill probe: `artifacts/evals/5151fb0b0ee9/`
- Append-only table: `artifacts/evals/leaderboard.csv`
- Freeze machine state (bakeoff log, not the product YAML): `artifacts/lock_winning/state.json`
- Freeze index job: `artifacts/jobs/cd3337fa4762/`
- Title-prefix index job: `artifacts/jobs/ae69f99b47b7/`

---

