# FocalNets for Object Counting (FSC-147)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)

Student research project (Stiftung Universität Hildesheim, 2023): replace Vision Transformer self-attention with **Focal Modulation Networks (FocalNets)** for few-shot object counting on **FSC-147**.

The goal was a computationally lighter encoder that keeps counting accuracy while improving inference speed, plus a **two-stage** training setup:

1. **Self-supervised reconstruction** — focal encoder + reverse patch-embedding rebuilds the input image.
2. **Supervised counting** — a density-map head is trained on FSC-147 point annotations.

This is an original PyTorch implementation of that research setup. It is **not** a fork of the Microsoft FocalNets repository. Architecture credit belongs to Yang et al., *Focal Modulation Networks*, NeurIPS 2022.

No benchmark numbers are claimed here beyond the project result: improved inference speed while maintaining accuracy on FSC-147 relative to a ViT-attention baseline trained in the same setup.

## Why Focal Modulation

Self-attention is global and quadratic in the number of patches. Focal modulation builds a context signal with stacked depth-wise convolutions (local → coarser → global) and gates it onto a query projection. For dense counting, that context is usually enough and cheaper at inference.

Reverse patch-embedding is a linear map from tokens back to RGB patches, used only in stage 1 so the encoder has to keep spatial structure before the counting head is attached.

## Quickstart

```bash
git clone https://github.com/Vivek1018/focalnets-object-counting.git
cd focalnets-object-counting

uv venv
uv pip install -e .
```

Download [FSC-147](https://github.com/cvlab-stonybrook/LearningToCountEverything) and place it at `data/FSC147/` (images, split JSON, and density maps as released by the dataset authors).

```bash
# Stage 1 — self-supervised reconstruction
focalnet-count pretrain --data-dir data/FSC147 --epochs 10 --batch-size 8

# Stage 2 — density-map counting
focalnet-count train --data-dir data/FSC147 --epochs 20 --batch-size 8 \
  --encoder-checkpoint checkpoints/pretrain.pt

# Evaluate MAE / RMSE on a split
focalnet-count eval --data-dir data/FSC147 --split val \
  --checkpoint checkpoints/counting.pt
```

Smoke-test the architecture without the dataset:

```bash
focalnet-count demo
```

## Layout

```text
src/focalnet_counting/
  modulation.py   # focal modulation block
  model.py        # encoder, reverse patch-embed, counting head
  dataset.py      # FSC-147 loader
  train.py        # two-stage loops
  cli.py
```

## References

- Yang, J., Li, C., Dai, X., & Gao, J. (2022). *Focal Modulation Networks*. NeurIPS.
- Ranjan, V., Sharma, U., Nguyen, T., & Hoai, M. (2021). *Learning To Count Everything*. CVPR. [FSC-147](https://github.com/cvlab-stonybrook/LearningToCountEverything)

## Author

**Vivek Kumar Verma** — student research project, Stiftung Universität Hildesheim, 2023.

[LinkedIn](https://www.linkedin.com/in/vivek-kumar-verma-0206765b/) · Vivek.verma1018@gmail.com
