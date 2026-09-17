# Retrieval eval — 36 questions, 3 retrievers

## all (n=36)

| retriever | R@1 | R@3 | R@5 | MRR@5 |
|---|---|---|---|---|
| vector | 0.97 | 1.00 | 1.00 | 0.99 |
| fulltext | 0.83 | 0.97 | 1.00 | 0.91 |
| hybrid | 0.89 | 1.00 | 1.00 | 0.94 |

## code (n=7)

| retriever | R@1 | R@3 | R@5 | MRR@5 |
|---|---|---|---|---|
| vector | 1.00 | 1.00 | 1.00 | 1.00 |
| fulltext | 0.86 | 1.00 | 1.00 | 0.93 |
| hybrid | 0.86 | 1.00 | 1.00 | 0.93 |

## paraphrase (n=24)

| retriever | R@1 | R@3 | R@5 | MRR@5 |
|---|---|---|---|---|
| vector | 0.96 | 1.00 | 1.00 | 0.98 |
| fulltext | 0.83 | 0.96 | 1.00 | 0.90 |
| hybrid | 0.92 | 1.00 | 1.00 | 0.96 |

## policy (n=5)

| retriever | R@1 | R@3 | R@5 | MRR@5 |
|---|---|---|---|---|
| vector | 1.00 | 1.00 | 1.00 | 1.00 |
| fulltext | 0.80 | 1.00 | 1.00 | 0.90 |
| hybrid | 0.80 | 1.00 | 1.00 | 0.90 |

## Misses (gold not in top 5)

| retriever | qid | question | gold | top-3 returned |
|---|---|---|---|---|
| — | — | none | — | — |
