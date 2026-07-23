# Benchmark run: seed=0, n=30, form=story, reasoning=off, temp=1

| model | exact | correct-swapped | correct-dualized | wrong | unparseable | api-error | graded | correct% | rsn rows | med rsn toks |
|---|---|---|---|---|---|---|---|---|---|---|
| anthropic/claude-opus-4.8 | 17 | 1 | 0 | 1 | 1 | 0 | 20 | 90.0 | 0 | 0 |
| openai/gpt-5.5 | 6 | 0 | 0 | 1 | 13 | 0 | 20 | 30.0 | 0 | 0 |
| x-ai/grok-4.3 | 0 | 0 | 0 | 12 | 8 | 0 | 20 | 0.0 | 0 | 0 |
