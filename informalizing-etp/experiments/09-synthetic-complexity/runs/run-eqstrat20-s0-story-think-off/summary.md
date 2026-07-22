# Benchmark run: seed=0, n=30, form=story, reasoning=off

| model | exact | correct-swapped | correct-dualized | wrong | unparseable | api-error | graded | correct% | rsn rows | med rsn toks |
|---|---|---|---|---|---|---|---|---|---|---|
| deepseek/deepseek-chat-v3.1 | 60 | 13 | 0 | 69 | 58 | 0 | 200 | 36.5 | 0 | 0 |
| qwen/qwen3-32b | 21 | 17 | 4 | 112 | 46 | 0 | 200 | 21.0 | 179 | 1 |
| meta-llama/llama-3.3-70b-instruct | 17 | 33 | 1 | 68 | 81 | 0 | 200 | 25.5 | 0 | 0 |
| mistralai/mistral-small-3.2-24b-instruct | 21 | 22 | 0 | 74 | 83 | 0 | 200 | 21.5 | 0 | 0 |
| openai/gpt-4o-mini | 23 | 3 | 2 | 127 | 45 | 0 | 200 | 14.0 | 0 | 0 |
| google/gemini-2.5-flash | 70 | 19 | 6 | 74 | 31 | 0 | 200 | 47.5 | 0 | 0 |
| anthropic/claude-haiku-4.5 | 27 | 28 | 1 | 111 | 33 | 0 | 200 | 28.0 | 0 | 0 |
| openai/gpt-5-mini | 32 | 9 | 0 | 111 | 48 | 0 | 200 | 20.5 | 0 | 0 |
| openai/gpt-5.5 | 134 | 16 | 0 | 23 | 27 | 0 | 200 | 75.0 | 0 | 0 |
| anthropic/claude-opus-4.8 | 172 | 18 | 0 | 2 | 8 | 0 | 200 | 95.0 | 0 | 0 |
