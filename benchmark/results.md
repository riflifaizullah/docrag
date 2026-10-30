| Variant | Right document in top 8 | Right page in top 8 | Right page first | Gate precision | Gate recall | Correctly refused |
|---|---|---|---|---|---|---|
| full (default) | 100% (79/79) | 99% (78/79) | 84% (66/79) | 95% (79/83) | 100% (79/79) | 80% (16/20) |
| no keyword search | 99% (78/79) | 95% (75/79) | 82% (65/79) | 95% (78/82) | 99% (78/79) | 80% (16/20) |
| no reranker | 99% (78/79) | 91% (72/79) | 58% (46/79) | 96% (75/78) | 95% (75/79) | 85% (17/20) |
| vector search only | 99% (78/79) | 73% (58/79) | 53% (42/79) | 96% (74/77) | 94% (74/79) | 85% (17/20) |
| chunks of 500 characters | 100% (79/79) | 99% (78/79) | 82% (65/79) | 95% (79/83) | 100% (79/79) | 80% (16/20) |
| chunks of 2000 characters | 100% (79/79) | 99% (78/79) | 89% (70/79) | 95% (79/83) | 100% (79/79) | 80% (16/20) |

By question type, default variant:

| Question type | Right document in top 8 | Right page in top 8 |
|---|---|---|
| heading | 100% (64/64) | 100% (64/64) |
| hand-written | 100% (15/15) | 93% (14/15) |
