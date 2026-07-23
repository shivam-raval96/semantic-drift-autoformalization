# Benchmark run: seed=0, n=30, form=story, reasoning=off, temp=1

| model | exact | correct-swapped | correct-dualized | wrong | unparseable | api-error | graded | correct% | rsn rows | med rsn toks |
|---|---|---|---|---|---|---|---|---|---|---|
| anthropic/claude-opus-4.8 | 14 | 1 | 0 | 2 | 3 | 0 | 20 | 75.0 | 0 | 0 |
| openai/gpt-5.5 | 6 | 0 | 0 | 3 | 11 | 0 | 20 | 30.0 | 0 | 0 |
| x-ai/grok-4.3 | 0 | 0 | 0 | 5 | 15 | 0 | 20 | 0.0 | 0 | 0 |
