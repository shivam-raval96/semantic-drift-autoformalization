# Benchmark run: seed=0, n=30, reasoning=on

| model | exact | correct-swapped | correct-dualized | wrong | unparseable | api-error | graded | correct% | rsn rows | med rsn toks |
|---|---|---|---|---|---|---|---|---|---|---|
| deepseek/deepseek-chat-v3.1 | 38 | 2 | 0 | 0 | 0 | 0 | 40 | 100.0 | 40 | 1185 |
| qwen/qwen3-32b | 22 | 12 | 1 | 5 | 0 | 0 | 40 | 87.5 | 40 | 1086 |
| meta-llama/llama-3.3-70b-instruct | 21 | 17 | 0 | 2 | 0 | 0 | 40 | 95.0 | 0 | 0 |
| mistralai/mistral-small-3.2-24b-instruct | 34 | 2 | 1 | 3 | 0 | 0 | 40 | 92.5 | 0 | 0 |
| openai/gpt-4o-mini | 33 | 0 | 2 | 5 | 0 | 0 | 40 | 87.5 | 0 | 0 |
| google/gemini-2.5-flash | 36 | 4 | 0 | 0 | 0 | 0 | 40 | 100.0 | 40 | 2115 |
| anthropic/claude-haiku-4.5 | 38 | 1 | 0 | 1 | 0 | 0 | 40 | 97.5 | 40 | 746 |
| openai/gpt-5-mini | 38 | 2 | 0 | 0 | 0 | 0 | 40 | 100.0 | 40 | 1024 |
