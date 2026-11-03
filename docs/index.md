# FAQ semantic cache: measured results

A support bot that answers a repeat question from a vector cache when the
meaning of the new question matches a stored one, and sends everything else to
an LLM. All 100 test questions were asked of a local `qwen2.5:1.5b` model
through Ollama, and every number below comes from the request log of one run
rather than an estimate.

## What the cache changes

| | no cache | cache at 0.70 |
| --- | --- | --- |
| answers from the LLM | 100 of 100 | 79 of 100 |
| average latency | 3532 ms | 2682 ms |
| tokens spent | 9919 | 79 fewer calls |

The 21 questions the cache answered itself came back in 31 to 56 ms. The three
slowest questions without a cache took 14.6 s, 13.8 s and 8.2 s. A hit rate of
21% is the honest price of a threshold that refuses to answer a question it is
not sure about: the safety property the project actually claims is zero false
hits, not a large hit rate.

## Threshold sweep

Five thresholds, 100 questions each, false hits counted against the 40
questions no FAQ covers.

| threshold | hit rate | false hits | average latency | LLM calls |
| --- | --- | --- | --- | --- |
| 0.60 | 36.0% | 8 of 40 | 2192 ms | 64 |
| 0.65 | 26.0% | 1 of 40 | 2537 ms | 74 |
| **0.70** | **21.0%** | **0 of 40** | **2682 ms** | **79** |
| 0.75 | 18.0% | 0 of 40 | 2892 ms | 82 |
| 0.80 | 12.0% | 0 of 40 | 3060 ms | 88 |

0.70 is the operating point: the highest hit rate among the thresholds that
served no negative question from the cache. Each threshold below it was
discarded for the false hit it cost, and each threshold above it cost hit rate
without buying any extra safety.

![Hit rate and latency across the sweep](figures/threshold_tradeoff.png)

## Why the hit rate is not higher

The similarity of a paraphrase to the FAQ it answers and the similarity of an
unrelated question to its nearest FAQ were measured separately, so the two
distributions could be placed on one scale.

![Similarity percentiles for paraphrases and negatives](figures/similarity_separation.png)

The negative distribution ends at 0.681, which is what makes 0.70 the strictest
safe threshold. The paraphrase distribution has its median at 0.574, below the
threshold, so roughly half of the paraphrases never reach the cache. The
paraphrases that do reach it score 0.706 to 0.960 and are all routed to the
correct FAQ entry.

## The negation guard

Five of the 100 questions carry a negation word. The cache answers none of
them, matching zero against five, because a negated question asks a different
thing from the FAQ it resembles and a cache cannot tell the two apart. One of
the five is a paraphrase of an FAQ: `the payment did not go through` is a
reworded way of asking what to do when a payment fails, and the guard refuses
it. That single question is why the sweep counted 21 hits while the calibration
counted 22 for the same threshold.

## One conversation

The demo asks three questions of a warm cache. The reworded question clears the
threshold against the answer the cold question stored, and the negated question
is answered by the model with the cache skipped.

| question | outcome | similarity | latency |
| --- | --- | --- | --- |
| do you have a loyalty programme for regular customers | miss, answered by the LLM and stored | 0.462 | 1878 ms |
| what loyalty benefits do regular customers get | hit, served from the stored answer | 0.850 | 70 ms |
| I do not want to return this, can I exchange instead | refused, cache skipped | — | 1705 ms |

Both figures regenerate from the measured artifacts with
`scripts/make_figures.py`.

The full report, the test set and the scripts that produced every number are in
the repository: [faq-semantic-cache](https://github.com/Bakkarranna/faq-semantic-cache).
