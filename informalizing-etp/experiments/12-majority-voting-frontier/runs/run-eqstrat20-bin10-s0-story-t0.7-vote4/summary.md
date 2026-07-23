# Majority vote: k=4, temp=0.7, form=story

| model | exact | correct-swapped | correct-dualized | wrong | unparseable | api-error | graded | correct% | rsn rows | med rsn toks |
|---|---|---|---|---|---|---|---|---|---|---|
| anthropic/claude-opus-4.8 | 17 | 3 | 0 | 0 | 0 | 0 | 20 | 100.0 | 0 | 0 |
| openai/gpt-5.5 | 6 | 0 | 0 | 6 | 8 | 0 | 20 | 30.0 | 0 | 0 |
| x-ai/grok-4.3 | 0 | 0 | 0 | 17 | 3 | 0 | 20 | 0.0 | 0 | 0 |
