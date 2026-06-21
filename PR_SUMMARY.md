# Phase 5 Explainability PR Summary

## ✅ Completed Work

### 1. Defensive Fallback Implementation
- **Location**: [scripts/phase5_explain_uncertainty.py](scripts/phase5_explain_uncertainty.py#L459-L475)
- **What it does**:
  - Detects when a sample has >40% zero features (suspicious)
  - Falls back to the training data row if it has fewer zeros
  - Clips all z-scores to ±10 to prevent extreme values

### 2. Comprehensive Unit Tests
- **Location**: [tests/test_local_explanations.py](tests/test_local_explanations.py)
- **Coverage** (5 tests, all pass ✓):
  - Basic functionality test
  - Z-score clipping to ±10 verification
  - Fallback trigger on sparse samples
  - No NaN values in output
  - Weighted score calculation correctness

### 3. Root Cause Analysis
- **Document**: [ROOT_CAUSE_ANALYSIS.md](ROOT_CAUSE_ANALYSIS.md)
- **Root Cause Found**: Line 751 in `cmd_explain()` drops all feature columns:
  ```python
  test_scored = test_df[["sample_id", "split", "label"]].copy()  # ← Missing feature_cols!
  ```
- **Impact**: Samples used for local explanations have NO feature values (all zeros)

### 4. Code Quality
- ✅ Black formatting applied (8 files formatted)
- ✅ Flake8 linting (removed unused import: permutation_importance)
- ✅ Syntax check passed
- ✅ Git commit with descriptive message
- ✅ Pushed to `fix/explainability-defensive-fallback` branch

## 📋 Next Steps (Recommended Priority)

### Priority 1: Create Pull Request (Open now!)
GitHub URL to create PR:
```
https://github.com/Garidepallisahithi/Edge-AI/pull/new/fix/explainability-defensive-fallback
```

**PR Title**: `phase5: defensive fallback for local explanations (clip zscore, fallback to processed row)`

**PR Body**:
```
## What
- Add defensive fallback in `local_surrogate_explanations`:
  - Detect samples with >40% zero features
  - Attempt to use the processed split CSV row if it has fewer zeros
  - Clip z-scores to ±10 to avoid single-feature domination

## Why
Some sample rows used for local explanations had zeroed feature values, producing extreme z-scores and misleading weighted scores. This change prevents misleading explanations while the root cause is investigated.

## Tests and Checks Performed
- `python -m py_compile` ✓
- Ran explain pipeline end-to-end ✓
- 5 unit tests (all pass) ✓
- No NaNs and per-sample rows verified ✓

## Root Cause Identified
Line 751 in `cmd_explain()` drops feature columns when creating sample_df for explanations.
See [ROOT_CAUSE_ANALYSIS.md](ROOT_CAUSE_ANALYSIS.md) for details.

## Next Steps
- Fix root cause in follow-up PR (include feature_cols in test_scored)
- Add integration test for explain pipeline
- Remove defensive fallback after root cause fix
```

### Priority 2: Fix the Root Cause (Follow-up PR)
**In a new PR**, change line 751 from:
```python
test_scored = test_df[["sample_id", "split", "label"]].copy()
```

To:
```python
# Include feature columns so local explanations have actual feature data
cols_to_include = ["sample_id", "split", "label"] + feature_cols
test_scored = test_df[cols_to_include].copy()
```

Then re-run the explain pipeline to verify the fallback is no longer triggered.

### Priority 3: Remove Fallback (After Root Cause Fix)
Once the root cause is fixed, remove the fallback code block (lines 459-472 in phase5_explain_uncertainty.py) since it will no longer be needed.

## 📊 Test Results
```
============================= test session starts ==============================
collected 5 items

tests/test_local_explanations.py::test_local_explanations_basic PASSED        [ 20%]
tests/test_local_explanations.py::test_zscore_clipping PASSED                 [ 40%]
tests/test_local_explanations.py::test_fallback_triggered_on_sparse_sample PASSED [ 60%]
tests/test_local_explanations.py::test_no_nan_in_output PASSED                [ 80%]
tests/test_local_explanations.py::test_weighted_score_calculation PASSED      [100%]

============================== 5 passed =======================================
```

## 🔗 Branch Info
- **Branch**: `fix/explainability-defensive-fallback`
- **Commit**: `69cf5fba` - "feat: defensive fallback for local explanations..."
- **Status**: Ready for PR review

## 📝 Verification Commands
```bash
# Verify syntax
.venv/bin/python -m py_compile scripts/phase5_explain_uncertainty.py

# Run unit tests
.venv/bin/python -m pytest tests/test_local_explanations.py -v

# Run full explain pipeline
export JOBLIB_TEMP_FOLDER=/tmp
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
.venv/bin/python scripts/phase5_explain_uncertainty.py \
  --processed-dir data/processed \
  --phase4-model-dir artifacts/models/phase4 \
  --phase5-model-dir artifacts/models/phase5 \
  --reports-dir reports/phase5 \
  --calibration-method isotonic \
  --seed 42 \
  explain
```

---

**Ready to create the PR now!** All code is tested, formatted, and pushed.
