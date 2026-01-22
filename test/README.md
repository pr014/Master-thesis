# Test Directory

This directory contains tests for the ECG training pipeline.

## What is a Smoke Test?

A **smoke test** is a quick, basic test that verifies the core functionality of your pipeline works without errors. It's called a "smoke test" because if there's a major problem, you'll see "smoke" (errors) immediately.

## `smoke_test_pipeline.py`

### Purpose
Tests the complete training pipeline before starting a long training run:
- ✅ Data loading with LOS bin labels
- ✅ Model forward pass
- ✅ Loss computation
- ✅ Batch format validation
- ✅ Stay-level aggregation

### When to Use
- **Before training**: Run this before submitting a long training job to catch errors early
- **After code changes**: Verify that your changes didn't break the pipeline
- **On new server/environment**: Check that everything is set up correctly

### How to Run
```bash
# Activate virtual environment
source venv/bin/activate

# Set environment variables
export ICUSTAYS_PATH="/path/to/icustays.csv"

# Run smoke test
python test/smoke_test_pipeline.py
```

### What It Checks
1. **Config loading**: Can configs be loaded?
2. **ICU stays loading**: Can ICU stay data be loaded?
3. **DataLoader creation**: Can DataLoaders be created?
4. **Batch format**: Are batches in the correct format?
5. **Label validation**: Are labels in range [0, 9]?
6. **Model creation**: Can the model be instantiated?
7. **Forward pass**: Does the model forward pass work?
8. **Loss computation**: Can loss be computed?
9. **Stay-level aggregation**: Can ECGs be aggregated by stay?

## Recommended Test Structure

For a complete test suite, consider adding:

```
test/
├── smoke_test_pipeline.py      # ✅ Already exists - Quick pipeline test
├── test_data_loading.py        # 🔜 Unit tests for data loading
├── test_preprocessing.py        # 🔜 Unit tests for preprocessing
├── test_models.py              # 🔜 Unit tests for model architectures
├── test_losses.py              # 🔜 Unit tests for loss functions
├── test_evaluation.py           # 🔜 Unit tests for evaluation metrics
└── conftest.py                 # 🔜 pytest configuration (fixtures, etc.)
```

## Testing Best Practices

### 1. **Smoke Tests** (Quick checks)
- Run before every training session
- Test complete pipeline end-to-end
- Should complete in < 1 minute

### 2. **Unit Tests** (Detailed checks)
- Test individual components in isolation
- Use pytest framework
- Should be fast (< 10 seconds each)

### 3. **Integration Tests** (Component interaction)
- Test how components work together
- Test edge cases
- Should be comprehensive

## Example: Adding Unit Tests

If you want to add proper unit tests using pytest:

```bash
# Install pytest
pip install pytest pytest-cov

# Run all tests
pytest test/

# Run with coverage
pytest test/ --cov=src --cov-report=html
```

Example unit test structure:
```python
# test/test_models.py
import torch
from src.models import CNNScratch

def test_cnn_scratch_forward():
    """Test CNN forward pass."""
    config = {"num_classes": 10}
    model = CNNScratch(config)
    x = torch.randn(2, 12, 5000)  # Batch of 2, 12 leads, 5000 samples
    output = model(x)
    assert output.shape == (2, 10)
```

## Current Status

- ✅ **Smoke test**: Implemented and working
- ⚠️ **Unit tests**: Not yet implemented (optional but recommended)
- ⚠️ **Integration tests**: Not yet implemented (optional)

## For Master's Thesis

For a Master's thesis, the smoke test is **sufficient**. Unit tests are nice-to-have but not required. The smoke test ensures your pipeline works before starting expensive training runs.

