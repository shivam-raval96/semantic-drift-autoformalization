# Majority vote: k=2, temp=1.0, form=story

| model | exact | correct-swapped | correct-dualized | wrong | unparseable | api-error | graded | correct% | rsn rows | med rsn toks |
|---|---|---|---|---|---|---|---|---|---|---|
| anthropic/claude-opus-4.8 | 17 | 0 | 0 | 1 | 2 | 0 | 20 | 85.0 | 0 | 0 |
| openai/gpt-5.5 | 6 | 0 | 0 | 4 | 10 | 0 | 20 | 30.0 | 0 | 0 |
| x-ai/grok-4.3 | 0 | 0 | 0 | 13 | 7 | 0 | 20 | 0.0 | 0 | 0 |
