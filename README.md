# PACE: A Unified Condense-and-Extract Paradigm for Fast VLM Inference

[![arXiv](https://img.shields.io/badge/arXiv-2608.27206-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2608.27206)
[![EMNLP](https://img.shields.io/badge/EMNLP%202026-Findings-0A66C2)](https://2026.emnlp.org/)

Official code for our EMNLP 2026 Findings paper
**[PACE: A Unified Condense-and-Extract Paradigm for Fast VLM Inference](https://arxiv.org/abs/2608.27206)**.

PACE is a training-free Condense-and-Extract inference framework for VLMs. An Adaptive Pixel Compressor (APC) downsamples redundant pixels before the vision encoder; a Dynamic Dual-Attention Extractor (DDAE) then keeps the salient visual tokens for the LLM.

## ⚙️ Experiment Setup

```bash
conda create -n pace python=3.10 -y
conda activate pace
git clone https://github.com/jjL357/PACE.git
cd PACE
pip install -r requirements.txt
```

## 🚀 Quick Start

```bash
export QWEN_MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct
bash scripts/reproduce_10_percent.sh
SETTING=fixed TOKEN_BUDGET=0.10 bash scripts/evaluate.sh
```

`SETTING` is `fixed` or `dynamic`. Override `TASKS`, `TOKEN_BUDGET`, and `OUTPUT_PATH` as needed. Logs go to `outputs/`.

## 📁 Repository Map

```text
PACE/
├── pace_vlm/models/          # APC, DDAE, lmms-eval plugin
├── scripts/                  # reproduce / evaluate
├── tests/                    
├── docs/reproduction.md      
└── requirements.txt
```

## 📚 Citation

```bibtex
@article{liu2026pace,
  title={PACE: A Unified Condense-and-Extract Paradigm for Fast VLM Inference},
  author={Liu, Junjie and Ye, Shengyuan and Chen, Xu},
  journal={arXiv preprint arXiv:2608.27206},
  year={2026}
}
```

## Acknowledgements

We thank the authors of [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), [VisionZip](https://github.com/dvlab-research/VisionZip), and [MMTok](hhttps://github.com/Ironieser/MMTok) for their open-source models, evaluation tools, and visual-token compression baselines.

## License

Released under [Apache-2.0](LICENSE). The Qwen2.5-VL modeling file is derived from Hugging Face Transformers and the Qwen team; see [NOTICE](NOTICE).
