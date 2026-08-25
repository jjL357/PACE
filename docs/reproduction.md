# Reproduction notes

## Reference setup

- Model: Qwen2.5-VL-7B-Instruct
- Precision: bfloat16
- Attention: FlashAttention 2
- Batch size: 1 per GPU
- Decoding: greedy
- Dataset cache: local Hugging Face cache
- Random seeds: lmms-eval defaults (Python 0, NumPy 1234, PyTorch 1234)

## Reference scores

The paper reports the following Qwen2.5-VL-7B scores under the fixed-resolution, 10% retention setting:

| Task | Metric | Paper |
|---|---|---:|
| RealWorldQA | accuracy | 68.37 |
| MMStar | accuracy | 59.08 |
| ChartQA | relaxed accuracy | 73.52 |

Run `scripts/reproduce_10_percent.sh` to evaluate all three tasks. Evaluation outputs are written to the ignored `outputs/` directory by default and must not be committed.
