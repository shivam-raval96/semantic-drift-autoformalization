# Majority vote: k=2, temp=0.7, form=story

| model | exact | correct-swapped | correct-dualized | wrong | unparseable | api-error | graded | correct% | rsn rows | med rsn toks |
|---|---|---|---|---|---|---|---|---|---|---|
| anthropic/claude-opus-4.8 | 14 | 2 | 0 | 4 | 0 | 0 | 20 | 80.0 | 0 | 0 |
| openai/gpt-5.5 | 5 | 0 | 0 | 6 | 9 | 0 | 20 | 25.0 | 0 | 0 |
| x-ai/grok-4.3 | 0 | 0 | 0 | 13 | 7 | 0 | 20 | 0.0 | 0 | 0 |
