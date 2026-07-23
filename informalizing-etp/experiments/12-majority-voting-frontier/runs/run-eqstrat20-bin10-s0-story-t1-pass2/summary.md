# Benchmark run: seed=0, n=30, form=story, reasoning=off, temp=1

| model | exact | correct-swapped | correct-dualized | wrong | unparseable | api-error | graded | correct% | rsn rows | med rsn toks |
|---|---|---|---|---|---|---|---|---|---|---|
| anthropic/claude-opus-4.8 | 15 | 0 | 0 | 1 | 4 | 0 | 20 | 75.0 | 0 | 0 |
| openai/gpt-5.5 | 5 | 0 | 0 | 3 | 12 | 0 | 20 | 25.0 | 0 | 0 |
| x-ai/grok-4.3 | 0 | 0 | 0 | 8 | 12 | 0 | 20 | 0.0 | 0 | 0 |
