"""
tests/conftest.py — Shared fixtures for the utils test suite.

The model dimensions here are chosen to be as small as possible while still
exercising every code path (embedding → transformer → classifier forward pass,
predict_proba batching, etc.).  They mirror the pattern from train.py:

    model = TabularTransformer(
        n_categories=train_dataset.get_n_categories(),   # list of ints
        n_continuous=train_dataset.num_idx[1],            # int
        n_classes=2,
        embed_dim=16,
    )
"""

import numpy as np
import pytest
import torch

from utils.torch_lib.TabularTransformer import TabularTransformer, TabularTransformerWrapper


# ── Model constants ───────────────────────────────────────────────────────────
# n_categories: 2 categorical features with 3 and 5 distinct values respectively
# n_continuous:  4 continuous features
# embed_dim:    must be divisible by nhead=4 → 16 works, 6 does not
N_CATEGORIES = [3, 5]
N_CONTINUOUS  = 4
N_CLASSES     = 2
EMBED_DIM     = 16
BATCH_SIZE    = 8


@pytest.fixture(scope="session")
def dummy_model() -> TabularTransformer:
    """
    A tiny TabularTransformer on CPU.
    Scope=session so it is built once and reused by all test files.
    """
    model = TabularTransformer(
        n_categories=N_CATEGORIES,
        n_continuous=N_CONTINUOUS,
        n_classes=N_CLASSES,
        embed_dim=EMBED_DIM,
    )
    model.eval()
    return model


@pytest.fixture(scope="session")
def dummy_wrapper(dummy_model: TabularTransformer) -> TabularTransformerWrapper:
    """
    TabularTransformerWrapper around dummy_model — mirrors how test.py uses it:

        wrapped = TabularTransformerWrapper(model, batch_size=128)
    """
    return TabularTransformerWrapper(dummy_model, batch_size=4, device="cpu")


@pytest.fixture(scope="session")
def numpy_batch() -> np.ndarray:
    """
    A float32 numpy array with shape (BATCH_SIZE, N_CONTINUOUS + N_CAT_COLS).

    TabularTransformerWrapper.predict_proba() expects X[:, 0:n_cont] as
    continuous features and X[:, n_cont:] as categorical features (long),
    exactly as assembled in test.py:

        X_transformed = preprocessor.transform(X_test)
        wrapped.predict(X_transformed)

    Categorical columns hold small non-negative integers so that
    nn.Embedding(num_cat=3, ...) and nn.Embedding(num_cat=5, ...) don't OOB.
    """
    rng = np.random.default_rng(42)

    # continuous block — any float values
    cont = rng.standard_normal((BATCH_SIZE, N_CONTINUOUS)).astype(np.float32)

    # categorical block — integers within valid embedding range
    cat_col_0 = rng.integers(0, N_CATEGORIES[0], size=(BATCH_SIZE, 1))
    cat_col_1 = rng.integers(0, N_CATEGORIES[1], size=(BATCH_SIZE, 1))
    cat = np.hstack([cat_col_0, cat_col_1]).astype(np.float32)

    return np.hstack([cont, cat])  # shape: (8, 6)
