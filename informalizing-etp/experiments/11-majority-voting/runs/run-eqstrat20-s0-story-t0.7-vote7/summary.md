# Majority vote: k=7, temp=0.7, form=story

| model | exact | correct-swapped | correct-dualized | wrong | unparseable | api-error | graded | correct% | rsn rows | med rsn toks |
|---|---|---|---|---|---|---|---|---|---|---|
| deepseek/deepseek-chat-v3.1 | 54 | 20 | 0 | 103 | 23 | 0 | 200 | 37.0 | 0 | 0 |
| qwen/qwen3-32b | 31 | 13 | 3 | 149 | 4 | 0 | 200 | 23.5 | 181 | 1 |
| meta-llama/llama-3.3-70b-instruct | 17 | 37 | 1 | 119 | 26 | 0 | 200 | 27.5 | 0 | 0 |
| mistralai/mistral-small-3.2-24b-instruct | 19 | 26 | 1 | 114 | 40 | 0 | 200 | 23.0 | 0 | 0 |
