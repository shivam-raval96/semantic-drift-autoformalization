# Benchmark run: seed=0, n=30, form=story, reasoning=off, temp=1

| model | exact | correct-swapped | correct-dualized | wrong | unparseable | api-error | graded | correct% | rsn rows | med rsn toks |
|---|---|---|---|---|---|---|---|---|---|---|
| anthropic/claude-opus-4.8 | 18 | 1 | 0 | 0 | 1 | 0 | 20 | 95.0 | 0 | 0 |
| openai/gpt-5.5 | 6 | 0 | 0 | 3 | 11 | 0 | 20 | 30.0 | 0 | 0 |
| x-ai/grok-4.3 | 0 | 0 | 0 | 6 | 14 | 0 | 20 | 0.0 | 0 | 0 |
