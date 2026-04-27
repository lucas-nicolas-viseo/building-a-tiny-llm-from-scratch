# Building a Tiny LLM from Scratch

Implementing, training, and optimizing a language model from scratch — following Sebastian Raschka's *"Build a Large Language Model (From Scratch)"* — then writing custom CUDA kernels for inference, all on a single NVIDIA RTX 3090 (24 GB VRAM).

## Goals

1. **Implement** a GPT-style architecture end-to-end (tokenizer, model, training loop, evaluation)
2. **Train** a small model on a curated dataset
3. **Optimize** inference with hand-written CUDA kernels (attention, FFN, sampling, KV-cache)
4. **Profile** throughput and memory vs. PyTorch baseline

## Hardware

- NVIDIA RTX 3090 (24 GB VRAM, 936 GB/s memory bandwidth)
- CPU: (fill in)
- RAM: (fill in)

## Progress

- [ ] Tokenizer (BPE)
- [ ] Model architecture (GPT, multi-head attention, RMSNorm, SwiGLU/MLP)
- [ ] Data pipeline & training loop
- [ ] Training run (loss curves, evaluation)
- [ ] Custom CUDA kernels — attention
- [ ] Custom CUDA kernels — FFN / MLP
- [ ] Custom CUDA kernels — sampling / top-k top-p
- [ ] KV-cache inference
- [ ] Performance comparison (PyTorch vs. CUDA kernels)
