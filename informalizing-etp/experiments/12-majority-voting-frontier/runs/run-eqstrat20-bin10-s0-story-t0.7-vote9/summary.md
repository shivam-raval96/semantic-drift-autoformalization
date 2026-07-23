# Majority vote: k=9, temp=0.7, form=story

| model | exact | correct-swapped | correct-dualized | wrong | unparseable | api-error | graded | correct% | rsn rows | med rsn toks |
|---|---|---|---|---|---|---|---|---|---|---|
| anthropic/claude-opus-4.8 | 17 | 3 | 0 | 0 | 0 | 0 | 20 | 100.0 | 0 | 0 |
| openai/gpt-5.5 | 7 | 0 | 0 | 8 | 5 | 0 | 20 | 35.0 | 0 | 0 |
| x-ai/grok-4.3 | 0 | 0 | 0 | 19 | 1 | 0 | 20 | 0.0 | 0 | 0 |
