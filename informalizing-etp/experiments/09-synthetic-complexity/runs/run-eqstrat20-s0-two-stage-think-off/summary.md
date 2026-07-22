# Benchmark run: seed=0, n=30, form=two-stage, reasoning=off

| model | exact | correct-swapped | correct-dualized | wrong | unparseable | api-error | graded | correct% | rsn rows | med rsn toks |
|---|---|---|---|---|---|---|---|---|---|---|
| deepseek/deepseek-chat-v3.1 | 45 | 40 | 0 | 68 | 47 | 0 | 200 | 42.5 | 0 | 0 |
| qwen/qwen3-32b | 23 | 35 | 1 | 93 | 48 | 0 | 200 | 29.5 | 172 | 1 |
| meta-llama/llama-3.3-70b-instruct | 39 | 25 | 0 | 85 | 51 | 0 | 200 | 32.0 | 0 | 0 |
| mistralai/mistral-small-3.2-24b-instruct | 17 | 41 | 0 | 87 | 55 | 0 | 200 | 29.0 | 0 | 0 |
| openai/gpt-4o-mini | 23 | 14 | 0 | 95 | 68 | 0 | 200 | 18.5 | 0 | 0 |
| google/gemini-2.5-flash | 94 | 12 | 0 | 67 | 27 | 0 | 200 | 53.0 | 0 | 0 |
| anthropic/claude-haiku-4.5 | 39 | 39 | 0 | 56 | 66 | 0 | 200 | 39.0 | 0 | 0 |
| openai/gpt-5-mini | 59 | 3 | 0 | 55 | 83 | 0 | 200 | 31.0 | 0 | 0 |
| openai/gpt-5.5 | 158 | 0 | 0 | 17 | 25 | 0 | 200 | 79.0 | 0 | 0 |
| anthropic/claude-opus-4.8 | 194 | 2 | 0 | 1 | 3 | 0 | 200 | 98.0 | 0 | 0 |
