# FAQ semantic cache

A semantic cache for an e-commerce support bot. Repeat questions are answered
from a vector cache when the meaning of the new question matches a stored
question; everything else goes to an LLM.

Status: work in progress. The measured numbers, the threshold table and the
latency comparison are added once `scripts/run_evaluation.py` has run against
the real embedding model on the 100 labelled test questions.

## Layout

```
data/faqs.json       30 frequently asked questions, each with an answer
data/test_set.json   100 labelled questions: 60 paraphrases, 40 negatives
src/faqcache/        embeddings, vector store, cache policy, logging
scripts/             evaluation, threshold sweep, report builder
tests/               unit tests plus a real-model integration job
docs/                GitHub Pages report
```
