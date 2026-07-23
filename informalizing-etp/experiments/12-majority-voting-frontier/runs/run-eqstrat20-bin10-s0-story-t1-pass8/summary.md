# Benchmark run: seed=0, n=30, form=story, reasoning=off, temp=1

| model | exact | correct-swapped | correct-dualized | wrong | unparseable | api-error | graded | correct% | rsn rows | med rsn toks |
|---|---|---|---|---|---|---|---|---|---|---|
| anthropic/claude-opus-4.8 | 15 | 1 | 0 | 2 | 2 | 0 | 20 | 80.0 | 0 | 0 |
| openai/gpt-5.5 | 4 | 0 | 0 | 2 | 14 | 0 | 20 | 20.0 | 0 | 0 |
| x-ai/grok-4.3 | 0 | 0 | 0 | 8 | 12 | 0 | 20 | 0.0 | 0 | 0 |
