# Majority vote: k=4, temp=1.0, form=story

| model | exact | correct-swapped | correct-dualized | wrong | unparseable | api-error | graded | correct% | rsn rows | med rsn toks |
|---|---|---|---|---|---|---|---|---|---|---|
| anthropic/claude-opus-4.8 | 18 | 2 | 0 | 0 | 0 | 0 | 20 | 100.0 | 0 | 0 |
| openai/gpt-5.5 | 6 | 0 | 0 | 7 | 7 | 0 | 20 | 30.0 | 0 | 0 |
| x-ai/grok-4.3 | 0 | 0 | 0 | 16 | 4 | 0 | 20 | 0.0 | 0 | 0 |
