# ML8502 Lab 1 - Adversarial ML Exercises

**Course:** ML8502 @ MBZUAI  
**Assignment:** Lab 1 - KGW Watermark and Neural Cleanse Exercises

This repository contains two completed exercises:
1. **KGW Watermark Exercise** - Implementation of red-green list watermarking for text generation
2. **Neural Cleanse Exercise** - Backdoor detection and reverse engineering using Neural Cleanse

---

## Table of Contents

- [KGW Watermark Exercise](#kgw-watermark-exercise)
- [Neural Cleanse Exercise](#neural-cleanse-exercise)
- [Setup](#setup)
- [Running the Exercises](#running-the-exercises)
- [Results](#results)

---

## KGW Watermark Exercise

### Overview

Implementation of the Kirchenbauer et al. red-green list watermark for text generation. The watermark uses a pseudorandom function (PRF) to partition the vocabulary into "green" and "red" tokens, then boosts green token probabilities during generation.

### Features

- ✅ Watermarked text generation with logit boosting
- ✅ Watermark detection using z-score analysis
- ✅ Green list generation via PRF (SHA-256 based)
- ✅ Configurable gamma (green list fraction) and delta (boost strength)

### Implementation Details

**Completed Components:**

1. **`generate_with_watermark()`** - Implements watermarked text generation:
   - Uses PRF to generate green list for each previous token
   - Boosts logits of green tokens by `delta`
   - Samples next token using softmax and multinomial

2. **`detect_watermark()`** - Implements watermark detection:
   - Counts green tokens based on previous token's green list
   - Computes green rate (hits / total)
   - Calculates z-score: `(green_rate - gamma) / sqrt(gamma * (1-gamma) / total)`

### Success Criteria

- ✅ Watermarked text has z-score > 2.0
- ✅ Watermarked z-score > Unwatermarked z-score
- ✅ All unit tests pass

---

## Neural Cleanse Exercise

### Overview

Implementation of Neural Cleanse (Wang et al., 2019) for backdoor detection and reverse engineering. The method optimizes triggers for each class and identifies backdoors by finding classes that require anomalously small triggers.

### Features

- ✅ Backdoor trigger reverse engineering
- ✅ Automatic backdoor class identification
- ✅ Trigger and mask visualization
- ✅ L1 regularization for sparse triggers

### Implementation Details

**Completed Components:**

1. **`neural_cleanse()`** - Implements the optimization loop:
   - Applies trigger to clean images using soft mask blending
   - Optimizes trigger and mask to induce target label
   - Uses cross-entropy loss + L1 regularization on mask
   - Projects trigger and mask to valid ranges

**Algorithm:**
```
For each suspected class:
  1. Initialize trigger (small random noise) and mask (small values)
  2. Optimize over N steps:
     - Apply trigger: blended = images * (1 - mask) + trigger * mask
     - Compute loss: CE(predicted, target) + lambda * ||mask||_1
     - Update trigger and mask
     - Project to valid ranges
  3. Record mask norm
Identify class with smallest mask norm as backdoor
```

### Success Criteria

- ✅ Successfully optimizes triggers for all classes
- ✅ Identifies poisoned class (smallest mask norm)
- ✅ Generates visualization of recovered trigger
- ✅ Saves `neural_cleanse_trigger.png`

---

## Setup

### Requirements

```bash
pip install torch torchvision transformers pillow matplotlib numpy
```

### Files Required

**For KGW Watermark:**
- Model is automatically downloaded (Qwen/Qwen2.5-0.5B)

**For Neural Cleanse:**
- `backdoored_resnet50.pth` - Backdoored model checkpoint
- `imagenet_sample/` - Directory with ImageNet sample images (JPEG format)

---

## Running the Exercises

### KGW Watermark Exercise

```bash
python kgw_watermark_exercise.py
```

**Expected Output:**
- Watermarked text generation
- Detection results with green rate and z-score
- Unwatermarked baseline comparison
- Summary showing z-score difference
- Success message if all tests pass

**Example Success Output:**
```
✅ Watermark detected! (z-score > 2.0)
✅ Watermarked text has higher z-score than unwatermarked
ALL TESTS PASSED! ✓
```

### Neural Cleanse Exercise

```bash
python neural_cleanse_exercise.py
```

**Expected Output:**
- Progress for each class being tested
- Mask norm for each class
- Sorted list of classes by mask norm
- Identification of suspicious class (⚠️ SUSPICIOUS!)
- Visualization saved to `neural_cleanse_trigger.png`
- Summary with mission answers

**Example Success Output:**
```
[1/4] Testing great_white_shark (class 2)... Mask norm: 0.123456
[2/4] Testing jay (class 17)... Mask norm: 0.234567
[3/4] Testing tree_frog (class 42)... Mask norm: 0.012345  ⚠️ SUSPICIOUS!
[4/4] Testing green_snake (class 52)... Mask norm: 0.198765

Most Suspicious Class: 42 (tree_frog)
```

---

## Results

### KGW Watermark Results

**Key Findings:**
- Watermarked generation successfully boosts green token probability
- Detection z-score typically > 3.0 for watermarked text
- Unwatermarked baseline z-score close to 0
- Clear separation between watermarked and unwatermarked text

### Neural Cleanse Results

**Key Findings:**
- Successfully identifies backdoor class by anomalously small mask norm
- Recovered trigger typically appears as a small pattern in a corner
- Mask shows sparse activation (most pixels near 0)
- Clean classes require much larger triggers to induce target label

---

## Code Structure

```
ml8502_lab1/
├── kgw_watermark_exercise.py    # KGW watermark implementation
├── neural_cleanse_exercise.py   # Neural Cleanse implementation
├── backdoored_resnet50.pth      # Backdoored model (for Neural Cleanse)
├── imagenet_sample/             # Sample images (for Neural Cleanse)
└── README.md                    # This file
```

---

## Technical Details

### KGW Watermark Algorithm

1. **Green List Generation:**
   - Hash: `SHA-256(secret_key + ":" + prev_token_id)`
   - Use first 8 bytes as RNG seed
   - Randomly select `gamma * vocab_size` tokens as green

2. **Generation:**
   - For each position, get green list based on previous token
   - Boost green token logits by `delta`
   - Sample next token from boosted distribution

3. **Detection:**
   - For each token, check if it's in green list of previous token
   - Count green tokens (hits) vs total tokens
   - Compute z-score for statistical significance

### Neural Cleanse Algorithm

1. **Optimization Objective:**
   - Minimize: `CE_loss + lambda * ||mask||_1`
   - Subject to: model(apply_trigger(image)) = target_label

2. **Key Insight:**
   - Clean classes require large, complex triggers
   - Backdoored classes recover the actual small backdoor trigger
   - Mask norm serves as anomaly detection metric

3. **Trigger Application:**
   - Soft blending: `blended = image * (1 - mask) + trigger * mask`
   - Mask values in [0, 1] control blend strength
   - Allows for sparse triggers

---

## References

1. **KGW Watermark:**
   - Kirchenbauer, J., et al. (2023). A Watermark for Large Language Models. *ICML 2023*.

2. **Neural Cleanse:**
   - Wang, B., et al. (2019). Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks. *IEEE S&P 2019*.

---

## Author

ML8502 @ MBZUAI  
Lab 1 Implementation

