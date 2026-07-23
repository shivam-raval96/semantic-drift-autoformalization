# Majority vote: k=3, temp=0.7, form=story

| model | exact | correct-swapped | correct-dualized | wrong | unparseable | api-error | graded | correct% | rsn rows | med rsn toks |
|---|---|---|---|---|---|---|---|---|---|---|
| anthropic/claude-opus-4.8 | 17 | 2 | 0 | 1 | 0 | 0 | 20 | 95.0 | 0 | 0 |
| openai/gpt-5.5 | 5 | 0 | 0 | 6 | 9 | 0 | 20 | 25.0 | 0 | 0 |
| x-ai/grok-4.3 | 0 | 0 | 0 | 15 | 5 | 0 | 20 | 0.0 | 0 | 0 |
