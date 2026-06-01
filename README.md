# HOMURA: Taming the Sand-Glass for Time-Constrained LLM Translation via Reinforcement Learning

<p align="center">
  <a href="#sand-glass-benchmark">Benchmark</a> •
  <a href="#method">Method</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="#training">Training</a> •
  <a href="#citation">Citation</a>
</p>

## Overview

Large Language Models achieve remarkable multilingual translation quality but suffer from **cross-lingual verbosity bias** — systematically producing translations longer than the source audio duration allows, making them unsuitable for dubbing and subtitling.

**HOMURA** is a reinforcement learning framework (GRPO on Qwen3-8B) with a novel *Dynamic Syllable-Ratio Reward* that achieves precise length control without sacrificing semantic adequacy.

This repository contains:
- **Sand-Glass** — a benchmark for evaluating translation under syllable-level duration constraints (1000 entries from real video transcripts)
- **HOMURA Training Code** — a fork of [verl](https://github.com/volcengine/verl) with our custom `PhonemeRewardManager`

## Sand-Glass Benchmark

[`Sandglass.json`](Sandglass.json) contains 1000 Chinese video transcript segments with duration constraints:

```jsonc
{
  "index": 368,
  "text_ocr_modify": "与此同时平行宇宙国王星",     // Source Chinese text
  "translation_text": "Meanwhile, on Planet King...", // Reference English translation
  "start_time": 659.66,                              // Audio segment start (sec)
  "end_time": 660.38,                                // Audio segment end (sec)
  "preceding_text": "...",                           // 3-sentence context before
  "following_text": "...",                           // 3-sentence context after
  "summary": "...",                                  // Video-level summary
  "fenqu": "二次元"                                  // Content domain
}
```


### Benchmark Statistics

The benchmark is balanced across 5 content domains (200 samples each), totaling 42.32 minutes of audio:

| Domain | Samples | Avg. Chars (ZH) | Avg. Duration (s) | Min Duration (s) | Max Duration (s) |
|--------|---------|-----------------|-------------------|-------------------|-------------------|
| ACGN | 200 | 12.02 | 1.91 | 0.56 | 4.11 |
| Film & Television | 200 | 11.71 | 2.37 | 0.76 | 6.31 |
| Travel & Tourism | 200 | 11.52 | 2.82 | 0.60 | 7.52 |
| Gaming | 200 | 12.13 | 2.68 | 0.92 | 6.61 |
| General Knowledge | 200 | 12.14 | 2.92 | 0.64 | 9.06 |
| **Overall** | **1,000** | **11.90** | **2.54** | **0.56** | **9.06** |

### Verbosity Bias (Motivation)

We find that frontier LLMs exhibit systematic cross-lingual verbosity bias — the Roundtrip Expansion Ratio ($R_{rtp}$) consistently exceeds 1.0 across all models and language pairs, with >60% of segments inflated:

| Language | Model | $R_{fwd}$ | $R_{rtp}$ | $R_{rtp} > 1$ (%) |
|----------|-------|-----------|-----------|---------------------|
| En | Claude-4.1-Opus | 1.36 | 1.41 | 64.8 |
| En | GPT-5 | 1.33 | 1.36 | 65.9 |
| De | DeepSeek-V3 | 1.67 | 1.54 | 63.8 |
| Es | DeepSeek-V3 | 1.99 | 1.99 | 64.0 |

This "sand-glass" effect — where translations expand beyond the source duration budget — motivates the need for explicit syllable-level control via RL.

## Project Structure

```
.
├── Sandglass.json                              # Sand-Glass benchmark
└── verl-HOMURA/                                # verl fork with HOMURA modifications
    ├── verl/
    │   ├── workers/reward_manager/
    │   │   ├── phoneme.py                      # PhonemeRewardManager (core contribution)
    │   │   └── syllable_calculation.py         # Multi-language syllable counter
    │   └── trainer/ppo/ray_trainer.py          # Modified for phoneme metric logging
    ├── examples/grpo_trainer/
    │   └── run_qwen3-8b.sh                     # Training script template
    ├── requirements.txt
    └── setup.py
```

## Getting Started

### Requirements

```bash
cd verl-HOMURA
pip install -e .
```

Key dependencies: `torch`, `vllm`, `transformers`, `pyphen`, `fugashi`, `sacrebleu`, `ftlangdetect`, `json-repair`

### Data Preparation

Convert `Sandglass.json` to the training format (parquet) expected by verl:

```python
import json
import pandas as pd

with open("Sandglass.json") as f:
    data = json.load(f)

# Prepare prompts with context and duration constraints
# See verl-HOMURA/verl/workers/reward_manager/phoneme.py for expected input_dic format
```

## Training

Train HOMURA with GRPO on 8×GPUs:

```bash
cd verl-HOMURA

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=<your_train.parquet> \
    data.val_files=<your_val.parquet> \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=Qwen/Qwen3-8B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=5 \
    reward_model.phoneme_left_bound=0.8 \
    reward_model.phoneme_right_bound=0.9 \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=15
```

Adjust `phoneme_left_bound` and `phoneme_right_bound` to target different syllable-ratio ranges (e.g., `0.6-0.7` for more compressed translations).

## Supported Languages

| Language | Syllable Method |
|----------|----------------|
| Chinese (zh) | Character count |
| English (en) | Pyphen hyphenation |
| German (de) | Pyphen hyphenation |
| Spanish (es) | Pyphen hyphenation |
| Japanese (ja) | Fugashi mora counting |



## Acknowledgement

The training framework is built upon [verl](https://github.com/volcengine/verl) (Volcano Engine Reinforcement Learning for LLMs).

## License

This project is released for research purposes. Please cite our paper if you use this code or the Sand-Glass benchmark.
