# Pixels to Predictions — SmolVLM Fine-tuning for Science VQA

Fine-tuning SmolVLM-500M-Instruct for science multiple-choice visual 
question answering using QLoRA + DoRA with a 5-seed ensemble.

**Best Result:** 0.76 LB (single seed) | 0.75 LB (ensemble)  
**Zero-shot Baseline:** 0.635  

---

## Task

Given an image and a science question with 2–5 multiple choice options 
(A–E), predict the correct answer index (0–4). Each sample optionally 
includes lecture notes, hints, and metadata (subject, grade, topic).

---

## Model & Approach

- **Base model:** HuggingFaceTB/SmolVLM-500M-Instruct
- **Fine-tuning:** QLoRA (4-bit NF4) + DoRA
- **LoRA targets:** q/k/v/o/gate/up/down projections (9.87M trainable params)
- **Training:** LR=3e-5, 2 epochs, cosine schedule, 5% warmup
- **Inference:** Log-probability scoring at native image resolution
- **Ensemble:** 5 seeds with score averaging

---

## Key Finding

The single most impactful improvement was fixing batched inference to read 
log-probabilities at the **real last token position** rather than position 
`-1` (which reads a padding token for shorter sequences in a batch). This 
alone recovered ~30 percentage points of accuracy.

---

## Results

| Seed | Val Accuracy | LB Score |
|------|-------------|----------|
| 42   | 0.6994      | 0.75     |
| 123  | 0.7195      | 0.76     |
| 456  | 0.7071      | 0.73     |
| 789  | 0.7118      | 0.72     |
| 2024 | 0.7195      | 0.76     |

| Ensemble Method  | Val Accuracy |
|-----------------|-------------|
| Majority Vote   | 0.7166      |
| Score Average   | **0.7233**  |
| Weighted Average| 0.7223      |

Seed 123 was selected as the final submission (0.76 LB).

---

## Setup

```bash
pip install transformers==4.57.6 peft==0.18.1 bitsandbytes>=0.43.0 \
            accelerate>=0.33.0 datasets pillow
```

> Do not upgrade transformers to 5.x — multiple breaking changes with SmolVLM.

