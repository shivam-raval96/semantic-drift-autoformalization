# GPT-5.5 results

| Condition | Exact | Correct-swapped | Wrong | Correct |
|---|---:|---:|---:|---:|
| Story -> R.G., reasoning on | 39 | 1 | 0 | 40/40 (100%) |
| Story -> R.G., reasoning off | 30 | 7 | 3 | 37/40 (92.5%) |
| Literal -> R.G., reasoning off | 40 | 0 | 0 | 40/40 (100%) |
| Story -> Literal -> R.G., reasoning off | 40 | 0 | 0 | 40/40 (100%) |

`correct-swapped` is semantically correct: it only exchanges the two sides
of an equality.

## Conclusion

The current task is largely saturated for GPT-5.5. Only direct Story -> R.G.
without reasoning produced errors, while the other three conditions reached
100%. Further runs with the same frontier model and setup are unlikely to
add much.
