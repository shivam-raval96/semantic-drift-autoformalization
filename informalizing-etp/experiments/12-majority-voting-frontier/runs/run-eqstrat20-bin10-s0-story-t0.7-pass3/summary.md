# Benchmark run: seed=0, n=30, form=story, reasoning=off, temp=0.7

| model | exact | correct-swapped | correct-dualized | wrong | unparseable | api-error | graded | correct% | rsn rows | med rsn toks |
|---|---|---|---|---|---|---|---|---|---|---|
| anthropic/claude-opus-4.8 | 15 | 2 | 0 | 1 | 2 | 0 | 20 | 85.0 | 0 | 0 |
| openai/gpt-5.5 | 5 | 0 | 0 | 1 | 14 | 0 | 20 | 25.0 | 0 | 0 |
| x-ai/grok-4.3 | 0 | 0 | 0 | 12 | 8 | 0 | 20 | 0.0 | 0 | 0 |
