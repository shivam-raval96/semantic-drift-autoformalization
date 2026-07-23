# Majority vote: k=5, temp=0.7, form=story

| model | exact | correct-swapped | correct-dualized | wrong | unparseable | api-error | graded | correct% | rsn rows | med rsn toks |
|---|---|---|---|---|---|---|---|---|---|---|
| deepseek/deepseek-chat-v3.1 | 55 | 20 | 0 | 94 | 31 | 0 | 200 | 37.5 | 0 | 0 |
| qwen/qwen3-32b | 30 | 12 | 3 | 149 | 6 | 0 | 200 | 22.5 | 180 | 1 |
| meta-llama/llama-3.3-70b-instruct | 16 | 37 | 1 | 113 | 33 | 0 | 200 | 27.0 | 0 | 0 |
| mistralai/mistral-small-3.2-24b-instruct | 17 | 27 | 1 | 109 | 46 | 0 | 200 | 22.5 | 0 | 0 |
