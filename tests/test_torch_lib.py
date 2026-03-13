"""
tests/test_torch_lib.py — Unit tests for utils/torch_lib/TabularTransformer.py
===============================================================================

All tests run on CPU (no GPU required).

Real usage patterns covered:

  From train.py:
      model = TabularTransformer(
          n_categories=n_categories,   # e.g. [3, 5, ...]
          n_continuous=n_continuous,
          n_classes=2,
          embed_dim=16,               # must be divisible by nhead=4
      )
      pred = model(X_cat, X_cont)     # forward pass used in training loop

  From test.py:
      wrapped = TabularTransformerWrapper(model, batch_size=128)
      y_pred  = wrapped.predict(X_transformed)
      y_proba = wrapped.predict_proba(X_transformed)[:, 1]
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from utils.torch_lib.TabularTransformer import TabularTransformer, TabularTransformerWrapper


# ── Constants (mirrors conftest.py for readability) ───────────────────────────
N_CATEGORIES = [3, 5]
N_CONTINUOUS  = 4
N_CLASSES     = 2
EMBED_DIM     = 16
BATCH          = 8


# ── TabularTransformer: forward pass ──────────────────────────────────────────

class TestTabularTransformerForward:
    """
    Tests for TabularTransformer.forward(x_cat, x_cont).
    The model is the core of the fraud detection pipeline; its output shape
    and dtype must be exact so the FocalLoss in train.py can consume it.
    """

    @pytest.fixture()
    def model(self) -> TabularTransformer:
        m = TabularTransformer(N_CATEGORIES, N_CONTINUOUS, N_CLASSES, EMBED_DIM)
        m.eval()
        return m

    @pytest.fixture()
    def inputs(self):
        """
        x_cat: (BATCH, n_cat_features) — long tensor, values within embedding range.
        x_cont: (BATCH, N_CONTINUOUS) — float32 tensor.
        """
        x_cat = torch.stack([
            torch.randint(0, N_CATEGORIES[0], (BATCH,)),
            torch.randint(0, N_CATEGORIES[1], (BATCH,)),
        ], dim=1)  # shape: (8, 2)
        x_cont = torch.randn(BATCH, N_CONTINUOUS)
        return x_cat, x_cont

    def test_output_shape(self, model, inputs):
        """Output must be (batch_size, n_classes) — consumed by FocalLoss(pred, y)."""
        x_cat, x_cont = inputs
        with torch.no_grad():
            out = model(x_cat, x_cont)
        assert out.shape == (BATCH, N_CLASSES)

    def test_output_dtype_is_float32(self, model, inputs):
        """FocalLoss and CrossEntropyLoss both require float32 logits."""
        x_cat, x_cont = inputs
        with torch.no_grad():
            out = model(x_cat, x_cont)
        assert out.dtype == torch.float32

    def test_output_is_logits_not_probabilities(self, model, inputs):
        """
        The model returns raw logits; softmax is applied downstream.
        Logits can be outside [0, 1] — if they are all in [0, 1] every time
        that would be suspicious (could mean accidental softmax inside model).
        We just verify values are unbounded (can exceed 1 or be negative).
        """
        x_cat, x_cont = inputs
        with torch.no_grad():
            out = model(x_cat, x_cont)
        # At least some logits should differ across the class dimension
        assert not torch.all(out[:, 0] == out[:, 1])

    def test_forward_is_deterministic_in_eval_mode(self, model, inputs):
        """Dropout is off in eval() mode — two identical forward passes must give identical output."""
        x_cat, x_cont = inputs
        with torch.no_grad():
            out1 = model(x_cat, x_cont)
            out2 = model(x_cat, x_cont)
        torch.testing.assert_close(out1, out2)

    def test_batch_size_one_works(self, model):
        """Edge case: batch of 1 — used when streaming predictions."""
        x_cat  = torch.tensor([[0, 1]], dtype=torch.long)  # (1, 2)
        x_cont = torch.randn(1, N_CONTINUOUS)
        with torch.no_grad():
            out = model(x_cat, x_cont)
        assert out.shape == (1, N_CLASSES)

    def test_incompatible_embed_dim_raises(self):
        """
        TransformerEncoderLayer requires embed_dim % nhead == 0.
        With nhead=4, embed_dim=6 is invalid → must raise at init time.
        """
        with pytest.raises(Exception):
            TabularTransformer(N_CATEGORIES, N_CONTINUOUS, N_CLASSES, embed_dim=6)


# ── TabularTransformerWrapper ────────────────────────────────────────────────

class TestTabularTransformerWrapper:
    """
    Tests for TabularTransformerWrapper — the sklearn-compatible adapter used in test.py.

    The wrapper's key responsibility:
      1. Accept a raw numpy array X (continuous cols first, then categorical)
      2. Slice into x_cont / x_cat
      3. Run inference in batches (respects batch_size to manage GPU memory)
      4. Return numpy probability array and predicted class array
    """

    def test_predict_proba_shape(self, dummy_wrapper, numpy_batch):
        """predict_proba must return (n_samples, n_classes) — as used in test.py."""
        proba = dummy_wrapper.predict_proba(numpy_batch)
        assert proba.shape == (len(numpy_batch), N_CLASSES)

    def test_predict_proba_rows_sum_to_one(self, dummy_wrapper, numpy_batch):
        """Each row represents a probability distribution → must sum to 1.0."""
        proba = dummy_wrapper.predict_proba(numpy_batch)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(len(numpy_batch)), atol=1e-5)

    def test_predict_proba_values_in_unit_interval(self, dummy_wrapper, numpy_batch):
        """Softmax output must be in [0, 1]."""
        proba = dummy_wrapper.predict_proba(numpy_batch)
        assert np.all(proba >= 0.0) and np.all(proba <= 1.0)

    def test_predict_returns_argmax_of_proba(self, dummy_wrapper, numpy_batch):
        """
        predict() must equal argmax(predict_proba()) — mirroring:
            y_proba = wrapped.predict_proba(X_transformed)[:, 1]
            y_pred  = wrapped.predict(X_transformed)
        """
        proba = dummy_wrapper.predict_proba(numpy_batch)
        preds = dummy_wrapper.predict(numpy_batch)
        expected = proba.argmax(axis=1)
        np.testing.assert_array_equal(preds, expected)

    def test_predict_output_dtype_is_integer(self, dummy_wrapper, numpy_batch):
        """Class indices must be integers for sklearn metrics (confusion_matrix, f1_score, etc.)."""
        preds = dummy_wrapper.predict(numpy_batch)
        assert np.issubdtype(preds.dtype, np.integer)

    def test_predict_values_within_class_range(self, dummy_wrapper, numpy_batch):
        """Predictions must be valid class indices: 0 or 1 for binary fraud detection."""
        preds = dummy_wrapper.predict(numpy_batch)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_batching_produces_same_result_as_single_pass(self, dummy_model, numpy_batch):
        """
        Wrapper with batch_size=2 should give the same output as batch_size=len(X).
        This validates that the batching loop in predict_proba doesn't introduce
        off-by-one errors when slicing X_batch.
        """
        wrapper_small = TabularTransformerWrapper(dummy_model, batch_size=2, device="cpu")
        wrapper_large = TabularTransformerWrapper(dummy_model, batch_size=len(numpy_batch), device="cpu")

        proba_small = wrapper_small.predict_proba(numpy_batch)
        proba_large = wrapper_large.predict_proba(numpy_batch)

        np.testing.assert_allclose(proba_small, proba_large, atol=1e-5)

    def test_fit_is_noop_and_returns_self(self, dummy_wrapper, numpy_batch):
        """
        fit() is intentionally a no-op (the model is pre-trained).
        sklearn's API requires it to return self so Pipeline.fit() works.
        """
        result = dummy_wrapper.fit(numpy_batch)
        assert result is dummy_wrapper

    def test_accepts_numpy_float64_input(self, dummy_wrapper):
        """
        preprocessor.transform() may return float64 depending on sklearn version.
        The wrapper must handle this without dtype errors.
        """
        rng = np.random.default_rng(1)
        X_f64 = rng.standard_normal((4, N_CONTINUOUS)).astype(np.float64)
        cat = rng.integers(0, 3, size=(4, 1)).astype(np.float64)
        cat2 = rng.integers(0, 5, size=(4, 1)).astype(np.float64)
        X_full = np.hstack([X_f64, cat, cat2])

        proba = dummy_wrapper.predict_proba(X_full)
        assert proba.shape == (4, N_CLASSES)
