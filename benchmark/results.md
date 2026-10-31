| Variant | Right document in top 8 | Right page in top 8 | Right page first | Gate precision | Gate recall | Correctly refused |
|---|---|---|---|---|---|---|
| full (default) | 100% (79/79) | 99% (78/79) | 84% (66/79) | 99% (79/80) | 100% (79/79) | 95% (19/20) |
| no keyword search | 99% (78/79) | 95% (75/79) | 82% (65/79) | 99% (78/79) | 99% (78/79) | 95% (19/20) |
| no reranker | 99% (78/79) | 91% (72/79) | 59% (47/79) | 96% (75/78) | 95% (75/79) | 85% (17/20) |
| vector search only | 96% (76/79) | 76% (60/79) | 54% (43/79) | 96% (75/78) | 95% (75/79) | 85% (17/20) |
| chunks of 500 characters | 100% (79/79) | 99% (78/79) | 82% (65/79) | 99% (79/80) | 100% (79/79) | 95% (19/20) |
| chunks of 2000 characters | 100% (79/79) | 99% (78/79) | 89% (70/79) | 99% (79/80) | 100% (79/79) | 95% (19/20) |

By question type, default variant:

| Question type | Right document in top 8 | Right page in top 8 |
|---|---|---|
| heading | 100% (64/64) | 100% (64/64) |
| hand-written | 100% (15/15) | 93% (14/15) |
