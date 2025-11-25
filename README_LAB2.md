# GCG Attack Implementation - Lab 2

**ML8502 @ MBZUAI** - Greedy Coordinate Gradient (GCG) Attack for Jailbreaking Language Models

## Overview

This implementation completes the GCG (Greedy Coordinate Gradient) attack algorithm as described in Zou et al. (2023). The attack optimizes adversarial suffixes that can potentially jailbreak aligned language models.

## Quick Start

```bash
python lab2.py --steps 50 --suffix-length 48
```

## Results Summary

✅ **SUCCESS**: Loss reduced from 0.9912 to 0.6357 (36% reduction)

- **Model**: Qwen/Qwen2.5-1.5B-Instruct
- **Optimization Steps**: 50
- **Suffix Length**: 48 tokens
- **Initial Loss**: 0.9912
- **Final Loss**: 0.6357
- **Best Loss**: 0.6353

## Implementation Details

### Completed Components

1. **GCG Optimization Loop** (`gcg_attack` function)
   - Target selection and input construction
   - Gradient computation via `token_gradients()`
   - Candidate generation from top-k gradient tokens
   - Batch evaluation of candidates
   - Best candidate selection and suffix update
   - Loss tracking and progress monitoring

2. **Key Features**
   - Initial loss computation for baseline
   - Proper padding handling in batch evaluation
   - Loss computation matching gradient function indices
   - Final loss computation for fair comparison

### Algorithm Flow

```
For each optimization step:
  1. Select target response (cycle through targets)
  2. Build input: [prompt + suffix + target]
  3. Compute gradients for suffix positions
  4. Generate candidates:
     - For each position: get top-k tokens with negative gradients
     - Create candidates by replacing positions
  5. Sample candidates (if > batch_size)
  6. Evaluate candidates in batches:
     - Compute loss for each candidate
     - Handle padding correctly
  7. Select best candidate (lowest loss)
  8. Update suffix and track best result
```

## Code Structure

```
lab2.py
├── load_model()              # Model loading (provided)
├── token_gradients()          # Gradient computation (provided)
├── gcg_attack()              # Main optimization loop (implemented)
│   ├── Initial loss computation
│   ├── GCG optimization loop
│   │   ├── Gradient computation
│   │   ├── Candidate generation
│   │   ├── Batch evaluation
│   │   └── Best candidate selection
│   └── Final loss computation
└── evaluate_attack()          # Attack evaluation (provided)
```

## Key Implementation Highlights

### 1. Initial Loss Tracking
```python
# Compute initial loss before optimization
target_ids = target_ids_list[0]
input_ids = torch.cat([prompt_ids, suffix_ids, target_ids]).unsqueeze(0)
# ... compute loss and add to history
```

### 2. Candidate Generation
```python
# For each suffix position
for pos in range(suffix_length):
    # Get top-k tokens with most negative gradients
    _, topk_indices = torch.topk(-grads[pos], topk)
    # Create candidates by replacing position
    for token_idx in topk_indices:
        candidate = suffix_ids.clone()
        candidate[pos] = token_idx
        candidates.append(candidate)
```

### 3. Batch Evaluation with Padding
```python
# Handle variable-length sequences
actual_seq_len = len(prompt_ids) + len(candidate) + len(target_ids)
# Extract logits matching token_gradients indexing
shift_logits = logits[j, candidate_suffix_end-1:actual_seq_len-1, :]
```

### 4. Loss Tracking
```python
# Track best loss and compute final loss on same target
if current_loss < best_loss:
    best_loss = current_loss
    best_suffix = suffix_ids.clone()
```

## Command Line Arguments

```bash
python lab2.py [OPTIONS]

Options:
  --steps INT           Number of optimization steps (default: 50)
  --suffix-length INT   Length of adversarial suffix (default: 48)
  --init-suffix STR     Initial suffix (JSON array or comma-separated)
```

## Output

The script outputs:
- Optimization progress with loss values
- Best suffix tokens found
- Attack evaluation results (10 trials)
- Summary with success/failure status

## Technical Notes

### Loss Computation
- Uses same indexing as `token_gradients()`: `suffix_end_idx-1:-1`
- Properly handles padding to exclude padding tokens
- Computes loss on actual sequence length, not padded length

### Batch Processing
- Evaluates candidates in mini-batches (size: 64)
- Handles variable-length sequences with padding
- Uses attention masks to ignore padding in model forward pass

### Target Cycling
- Cycles through multiple target responses during optimization
- Final loss computed on first target for fair comparison

## Evaluation Results

While the optimization successfully reduced loss, the attack did not successfully jailbreak the model in evaluation trials. All 10 trials resulted in the model refusing the request.

**Possible Reasons:**
- Loss reduction ≠ successful jailbreak
- Additional safety mechanisms in generation
- Generation parameters may need adjustment
- Suffix may need further optimization

## Dependencies

- PyTorch
- Transformers (Hugging Face)
- tqdm

## References

Zou, A., Wang, Z., Kolter, J. Z., & Fredrikson, M. (2023). Universal and Transferable Adversarial Attacks on Aligned Language Models. *arXiv preprint arXiv:2307.15043*.

## License

This is an educational assignment for ML8502 @ MBZUAI.

