# Contributing

Contributions are welcome through focused issues and pull requests.

Before submitting a change:

1. Keep algorithm parameters configurable through the lmms-eval model arguments; do not require source edits for experiments.
2. Preserve the paper defaults and document intentional behavioral changes.
3. Run `python -m unittest discover -s tests -v` and `python -m compileall -q pace_vlm`.
4. Do not commit model weights, benchmark data, caches, logs, or generated evaluation outputs.

For accuracy changes, include the model, task, token budget, resolution setting, dependency versions, and random seeds.
