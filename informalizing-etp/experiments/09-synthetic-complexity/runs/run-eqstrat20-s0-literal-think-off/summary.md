# Benchmark run: seed=0, n=30, form=literal, reasoning=off

| model | exact | correct-swapped | correct-dualized | wrong | unparseable | api-error | graded | correct% | rsn rows | med rsn toks |
|---|---|---|---|---|---|---|---|---|---|---|
| deepseek/deepseek-chat-v3.1 | 60 | 32 | 0 | 57 | 51 | 0 | 200 | 46.0 | 0 | 0 |
| qwen/qwen3-32b | 26 | 36 | 0 | 90 | 48 | 0 | 200 | 31.0 | 170 | 1 |
| meta-llama/llama-3.3-70b-instruct | 62 | 4 | 0 | 80 | 54 | 0 | 200 | 33.0 | 0 | 0 |
| mistralai/mistral-small-3.2-24b-instruct | 13 | 50 | 0 | 76 | 61 | 0 | 200 | 31.5 | 0 | 0 |
| openai/gpt-4o-mini | 42 | 6 | 0 | 103 | 49 | 0 | 200 | 24.0 | 0 | 0 |
| google/gemini-2.5-flash | 98 | 12 | 0 | 58 | 32 | 0 | 200 | 55.0 | 0 | 0 |
| anthropic/claude-haiku-4.5 | 39 | 40 | 0 | 54 | 67 | 0 | 200 | 39.5 | 0 | 0 |
| openai/gpt-5-mini | 65 | 4 | 0 | 50 | 81 | 0 | 200 | 34.5 | 0 | 0 |
| openai/gpt-5.5 | 159 | 0 | 0 | 20 | 21 | 0 | 200 | 79.5 | 0 | 0 |
| anthropic/claude-opus-4.8 | 190 | 4 | 0 | 3 | 3 | 0 | 200 | 97.0 | 0 | 0 |
