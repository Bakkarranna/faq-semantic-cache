# FAQ semantic cache

[![CI](https://github.com/Bakkarrarna/faq-semantic-cache/actions/workflows/ci.yml/badge.svg)](https://github.com/Bakkarrarna/faq-semantic-cache/actions/workflows/ci.yml)
[![Pages](https://github.com/Bakkarrarna/faq-semantic-cache/actions/workflows/pages.yml/badge.svg)](https://github.com/Bakkarrarna/faq-semantic-cache/actions/workflows/pages.yml)

A semantic cache for an e-commerce support bot. A repeat question is answered
from a vector cache when the meaning of the new question matches a stored
question; a negated question is answered by the LLM with the cache skipped;
everything else goes to the LLM.

The cache answers 196 of the 1,000 test questions itself, in 81.6 ms on
average, and asks the LLM 196 fewer times. Two of the 400 questions no FAQ
covers were served from the cache. Every number in this report was measured on
one machine and written to `results/` by the scripts in `scripts/`. Nothing
here is estimated.

## What the numbers are

The FAQ set is 300 entries, English and Hinglish. The test set is 1,000
labelled questions: 600 paraphrases that one FAQ entry answers, and 400
questions no FAQ covers. The baseline asks all 1,000 questions of a local
`qwen2.5:1.5b` model through Ollama with the cache bypassed. The cache run
then asks the same 1,000 questions at five similarity thresholds.

| | no cache | cache at 0.80 |
| --- | --- | --- |
| answers from the LLM | 1,000 of 1,000 | 804 of 1,000 |
| average latency | 5270 ms | 4290 ms |
| tokens spent | 98,826 | not measured, see limits |
| slowest question | 19,163 ms | see log |

A hit is timed directly, at an average of 81.6 ms, the slowest of the 196 hits
at 264.6 ms. A miss is charged both the cache lookup and the LLM time that the
baseline measured for that same question, so a miss is never reported as
optimistic. The average latency of the cache run, 4290 ms against 5270 ms, is a
1.23x improvement built out of the 196 hits and the 196 LLM calls they removed.

The three slowest baseline questions are worth naming, because they are the
shape of the latency the cache exists to remove: `limited drop mein stock kaise
allocate hota hai` at 19,163 ms, `how do I get a tax invoice with GSTIN` at
19,044 ms, and `ek subscription delivery skip karne ka kya tarika hai` at
18,339 ms.

## Threshold sweep

Five thresholds, the same 1,000 questions at each one, with a false hit counted
whenever a question no FAQ covers was served from the cache.

| threshold | hit rate | false hits | wrong FAQ | average latency | LLM calls |
| --- | --- | --- | --- | --- | --- |
| 0.60 | 47.2% | 100 of 400 | 55 | 2955 ms | 528 |
| 0.65 | 38.0% | 55 of 400 | 40 | 3374 ms | 620 |
| 0.70 | 31.5% | 31 of 400 | 29 | 3736 ms | 685 |
| 0.75 | 25.1% | 11 of 400 | 16 | 4055 ms | 749 |
| **0.80** | **19.6%** | **2 of 400** | **5** | **4290 ms** | **804** |

0.80 is the operating point. At 400 negative questions no threshold on the grid
served none of them, so the rule that picked the point is a stated budget
rather than a measured clean cut: a threshold is eligible when it keeps a false
hit within one in a hundred negatives, and the hit rate decides between the
eligible ones. `FALSE_HIT_BUDGET = 0.01` in `scripts/run_evaluation.py`, and
0.80 is the only eligible row at 0.5%. Running `choose_threshold` over the
committed `results/sweep.json` returns 0.80, so the number in
`results/summary.json` is reproducible from the committed record rather than
typed by hand.

The grid runs from 0.60 to 0.80 because the calibration shows the two
distributions overlapping from 0.478 up to 0.871: that band is the whole range a
threshold has to decide on, and a grid that stopped short of 0.60 would only
look at the region where the hit rate was already given away.

## Where the boundary actually is

`scripts/calibrate.py` loads the embedding model on its own and measures two
similarity distributions without timing any LLM call: a paraphrase against the
FAQ it answers, and a negative question against its nearest FAQ.

| group | n | p25 | p50 | p75 | p90 | max |
| --- | --- | --- | --- | --- | --- | --- |
| paraphrase to its FAQ | 600 | 0.478 | 0.678 | 0.848 | 0.935 | 0.996 |
| negative to nearest FAQ | 400 | 0.461 | 0.529 | 0.610 | 0.687 | 0.871 |

The negative distribution reaches 0.871 and its p90 is 0.687, so the two groups
overlap across the top of the grid and no threshold separates them cleanly.
That overlap is why the operating point is a budget rather than a zero. The
paraphrase median, 0.678, sits below the operating threshold, which is why the
hit rate is a fifth of the questions rather than most of them: this embedder
separates the two groups at the top but places half of the paraphrases in the
same band as the negatives. The nearest FAQ was the correct one for 424 of the
600 paraphrases at rank one.

The two distributions, with the operating point marked:

![Similarity percentiles](docs/figures/similarity_separation.png)

![Threshold trade-off](docs/figures/threshold_tradeoff.png)

## The negation guard

A negated question asks a different thing from the FAQ it resembles. `the
payment did not go through` scores highly against the FAQ that answers what to
do when a payment fails, and serving that answer would be wrong, so a question
carrying a negation word goes to the LLM with the cache skipped entirely.

| question | word found | outcome | time |
| --- | --- | --- | --- |
| the payment did not go through | not | refused | 1.05 ms |
| refund nahi milega kya | nahi | refused | 0.32 ms |
| mujhe refund nahi chahiye | nahi | refused | 0.31 ms |
| mera parcel kabhi nahi aaya | nahi | refused | 0.34 ms |
| I do not want to return this, can I exchange instead | not | refused | 0.65 ms |

45 of the 1,000 test questions match the guard and none of them is answered
from the cache: zero negated hits out of 45, recorded in
`results/negation_examples.json`. The guard costs 0.405 ms on average across
the 1,000 questions because it needs no embedding.

32 of the 45 refused questions are paraphrases of an FAQ, and the guard refuses
every one of them. The calibration counted similarity alone, so it recorded 201
paraphrase hits and 3 false hits at 0.80 where the measured sweep recorded 194
paraphrase hits and 2 false hits: the guard removed five paraphrase hits and one
false hit before the index was consulted.

```
calibration, t=0.80:  paraphrase hits 201/600, false hits 3
sweep log,    t=0.80:  hits 196 of 1,000, refused 45
```

The one-question difference is the guard working as intended. The calibration
measured embedding similarity alone and never applied the guard, so it is the
optimistic count of the two.

## One conversation, end to end

`scripts/demo.py` asks three questions of a warm cache and prints what the user
gets. The questions were chosen by measuring embedder similarities before the
run, so the outcomes below are what the run produced rather than what it was
meant to produce.

| question | outcome | similarity | latency |
| --- | --- | --- | --- |
| item nakli lage to kya karna hota hai | miss, answered by the LLM, stored | 0.320 | 3425 ms |
| seller nakli item bhej de to kya hota hai | hit, served from the stored answer | 0.940 | 82 ms |
| the payment did not go through | refused, cache skipped, LLM answered | — | 5745 ms |

The first question scores 0.320 against its nearest FAQ, below the threshold,
so it is a real miss and it costs 102 tokens. The rewording scores 0.940
against the answer that question stored, so the second is answered from the
cache in 82 ms and the LLM is not called at all. The model was loaded before
the sequence began, so the 3425 ms of the first step is the model answering
rather than the weights loading. The third question is refused by the guard and
still gets a correct answer from the model, which is what a support desk needs.
The run writes `results/demo.json`.

## How the cache works

1. **Normalise and guard.** The question is lower-cased and scanned for negation
   words in English and in Hinglish, because the FAQ set covers both. A match
   ends the lookup and the question goes straight to the LLM.
2. **Embed.** The question is encoded with
   `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` at 384
   dimensions, which fits on a machine with 8 GB of RAM alongside the LLM.
   The weights are loaded shard by shard to keep the peak allocation within
   that budget.
3. **Search.** The store is a FAISS inner-product index over normalised vectors,
   so cosine similarity is an index search rather than a linear scan, and it
   returns the nearest stored question. Against 300 stored questions the
   lookup takes 81.0 ms on a miss.
4. **Decide.** The nearest similarity is compared with the operating threshold.
   At or above it the stored answer is served; below it the question goes to
   the LLM.
5. **Write back.** A miss the LLM answered is stored against its own embedding,
   so the next similar question can hit, unless the question looks like user
   data. A miss that matched an FAQ above the threshold refreshes that FAQ's
   answer.

Every request is logged as one JSON line with the question, the outcome, the
similarity that produced it, the latency and the answer length. No answer text
and nothing customer-specific is written to the log.

## Layout

```
data/faqs.json        300 FAQ entries, English and Hinglish
data/test_set.json    1,000 labelled questions: 600 paraphrases, 400 negatives
src/faqcache/
  config.py           threshold, TTL, models, user-data patterns
  data.py             FAQ and test-set loading
  embeddings.py       sentence-transformers wrapper, loaded shard by shard
  store.py            FAISS index and nearest-neighbour search
  negation.py         negation words, and the user-data check on write-back
  cache.py            the decision path: guard, search, threshold, write-back
  llm.py              Ollama client
  logging.py          JSONL request log
scripts/
  run_evaluation.py   baseline phase, threshold sweep, request log
  calibrate.py        similarity distributions, no LLM call
  pick_negation_examples.py
  smoke_embeddings.py
  make_figures.py     regenerates the figures from results/
  demo.py             the three-question conversation
tests/               45 unit tests, 9 integration tests
results/             measured artifacts, one JSON per phase
docs/                GitHub Pages report
```

## Running it

```bash
pip install -r requirements.txt
ollama serve                                  # or the Ollama desktop app

python scripts/run_evaluation.py --phase baseline   # 1,000 LLM calls
python scripts/run_evaluation.py --phase sweep      # five thresholds
python scripts/calibrate.py
python scripts/make_figures.py
python scripts/demo.py --json results/demo.json
```

The two phases are separate commands because a machine with 8 GB of RAM cannot
hold the embedding model and the 1.5B LLM at the same time. The baseline phase
loads no embedding model and the sweep phase makes no LLM call: a miss is
charged the LLM time the baseline measured for that same question. The split is
about memory, not about hiding a timing. At the measured average of 5270 ms per
call, the baseline phase is about 88 minutes of model time. The numbers land in
`results/baseline.json`, `results/sweep.json` and `results/summary.json`.

The measurements were taken on Python 3.13 with sentence-transformers 6.2.0,
FAISS on CPU, and `qwen2.5:1.5b` served by Ollama on 127.0.0.1:11434 at
temperature 0.

## Checks

```
pytest tests/unit -q          45 passed
pytest tests/integration -q  9 passed
ruff check .                  clean
python scripts/smoke_embeddings.py   all checks pass
```

The unit tests use a deterministic embedder so they run without the model. The
integration tests load the real model and walk the three outcomes, the guard
and the logging path, and they fail rather than skip when the embedder cannot
load. Both families run in GitHub Actions on every push to `main`; the unit
suite is the required check and the model jobs confirm the real embedder still
separates the test set.

The smoke script re-measures the whole test set against the operating point on
every push. Its bounds are the ones the measurement satisfied: paraphrases
ranked at least a quarter of the time (194 of 600), false hits within the 1%
budget of negatives (2 of 400) and wrong-FAQ hits within 5% of hits (5 of 194).
It reports the counts it measured before judging them, so a regression in the
model or the code fails the run.

## Limits worth stating

- A 19.6% hit rate is a property of this test set and this embedder as much as
  of the threshold. Half the paraphrases score below 0.80, so this cache
  catches near-identical rewordings rather than any phrasing.
- Two questions no FAQ covers were served from the cache, both price-match
  questions against FAQ-130 at 0.804 and 0.826. No threshold on the grid
  reached zero false hits at 400 negatives, so the operating point is a budget
  of one false hit in a hundred negatives, not a clean separation.
- Five hits served the wrong FAQ. That is the cost of a 19.6% hit rate on this
  embedder, and it is counted, not averaged away.
- The threshold was chosen on the same 1,000 questions it was evaluated on, so
  it describes this set rather than predicting traffic.
- The cache holds answers in memory only. A restart starts from the FAQ set
  again and nothing persists to disk.
- The sweep phase made no LLM call, so the cached run has no token figure; the
  98,826 tokens quoted are the baseline's.

The report is generated into `docs/` and the Pages workflow publishes it on every push.
As of 2026-11-07 that URL returns HTTP 404: the deploy job reports success and the
deployment artifact exists, but the API still reports the Pages site as unbuilt, so the
committed copy of the report is the record:
<https://bakkarrarna.github.io/faq-semantic-cache/>
