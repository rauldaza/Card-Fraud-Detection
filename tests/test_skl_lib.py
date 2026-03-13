"""
tests/test_skl_lib.py — Unit tests for utils/skl_lib/LibPreTabTransformer.py
============================================================================

Real usage from tab_preprocessing.py:

    cat_pipeline = Pipeline([
        ...
        ("shifter", FunctionTransformer(shift_plus_one)),   # after OrdinalEncoder
    ])

    pca_parallel_steps = [
        ("pca_group_1", MaskedPCA(n_components=0.9), cols_V_group),
        ...
    ]
    pca_engine = ColumnTransformer(pca_parallel_steps, remainder="passthrough")
    num_pipeline = Pipeline([
        ("pca",     pca_engine),
        ("imputer", SimpleImputer(...)),
        ("scaler",  StandardScaler()),
    ])

Key properties to cover:
  - shift_plus_one: used to shift OrdinalEncoder outputs by +1 so 0 is reserved
    as "missing" (the category not seen during training gets -1 → 0 after shift)
  - MaskedPCA: handles V-column groups which have varying NaN patterns;
    only non-NaN rows are fitted/transformed, NaN rows propagate NaN through
"""

import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from utils.skl_lib.LibPreTabTransformer import MaskedPCA, shift_plus_one


# ── shift_plus_one ────────────────────────────────────────────────────────────

class TestShiftPlusOne:
    """
    After OrdinalEncoder, categories are 0-based integers.
    Unknown categories get -1.  shift_plus_one maps -1 → 0, 0 → 1, etc.
    so downstream nn.Embedding indices are always ≥ 0.
    """

    def test_adds_one_to_numpy_array(self):
        X = np.array([0.0, 1.0, 2.0, -1.0])
        result = shift_plus_one(X)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0, 0.0])

    def test_adds_one_to_2d_array(self):
        """ColumnTransformer passes 2-D arrays — shape must be preserved."""
        X = np.array([[0, 1], [2, -1]], dtype=float)
        result = shift_plus_one(X)
        expected = np.array([[1, 2], [3, 0]], dtype=float)
        np.testing.assert_array_equal(result, expected)

    def test_works_on_scalar(self):
        assert shift_plus_one(5) == 6
        assert shift_plus_one(-1) == 0

    def test_works_via_function_transformer(self):
        """Verify compatibility with sklearn FunctionTransformer (used in tab_preprocessing.py)."""
        transformer = FunctionTransformer(shift_plus_one)
        X = np.array([[0, 1], [2, 3]], dtype=float)
        result = transformer.transform(X)
        np.testing.assert_array_equal(result, [[1, 2], [3, 4]])

    def test_does_not_mutate_input(self):
        X = np.array([1.0, 2.0, 3.0])
        original = X.copy()
        shift_plus_one(X)
        np.testing.assert_array_equal(X, original)


# ── MaskedPCA ─────────────────────────────────────────────────────────────────

class TestMaskedPCA:
    """
    MaskedPCA is applied per V-column group in the num_pipeline.
    Each group can have a different NaN pattern (rows with all-NaN in that group
    are skipped during fit and produce NaN rows in the output).

    This is crucial for the fraud dataset because many V-columns are only
    populated for certain transaction types.
    """

    @pytest.fixture()
    def clean_data(self) -> np.ndarray:
        """10 samples × 5 features, no NaNs."""
        rng = np.random.default_rng(0)
        return rng.standard_normal((10, 5)).astype(float)

    @pytest.fixture()
    def data_with_nans(self) -> np.ndarray:
        """
        10 samples × 5 features.
        Rows 2, 5, 8 contain NaNs — exactly as happens with V-group columns
        where not all transaction types populate every V feature.
        """
        rng = np.random.default_rng(0)
        X = rng.standard_normal((10, 5))
        X[[2, 5, 8], :] = np.nan
        return X

    # ── fit / transform with clean data ──────────────────────────────────────

    def test_fit_returns_self(self, clean_data):
        """sklearn contract: fit must return self."""
        pca = MaskedPCA(n_components=2)
        result = pca.fit(clean_data)
        assert result is pca

    def test_transform_clean_data_shape(self, clean_data):
        """Output has (n_samples, n_components) when n_components is an int."""
        pca = MaskedPCA(n_components=2)
        pca.fit(clean_data)
        out = pca.transform(clean_data)
        assert out.shape == (10, 2)

    def test_transform_variance_ratio_shape(self, clean_data):
        """
        When n_components is a float (0.9), PCA picks enough components
        to explain 90% variance — shape[1] must match pca.n_components_.
        """
        pca = MaskedPCA(n_components=0.9)
        pca.fit(clean_data)
        out = pca.transform(clean_data)
        assert out.shape == (len(clean_data), pca.pca.n_components_)

    def test_sklearn_pipeline_compatible(self, clean_data):
        """MaskedPCA must work inside a sklearn Pipeline (used in num_pipeline)."""
        pipe = Pipeline([("pca", MaskedPCA(n_components=2))])
        out = pipe.fit_transform(clean_data)
        assert out.shape == (10, 2)

    # ── NaN handling ──────────────────────────────────────────────────────────

    def test_nan_rows_produce_nan_output(self, data_with_nans):
        """
        Rows 2, 5, 8 are all-NaN in the input → they must be all-NaN in the output.
        The SimpleImputer in the next pipeline step will then fill them with 0.
        """
        pca = MaskedPCA(n_components=2)
        pca.fit(data_with_nans)
        out = pca.transform(data_with_nans)
        assert np.all(np.isnan(out[[2, 5, 8], :]))

    def test_clean_rows_are_not_nan(self, data_with_nans):
        """Non-NaN input rows must produce finite output values."""
        pca = MaskedPCA(n_components=2)
        pca.fit(data_with_nans)
        out = pca.transform(data_with_nans)
        clean_rows = [i for i in range(10) if i not in (2, 5, 8)]
        assert np.all(np.isfinite(out[clean_rows, :]))

    def test_output_shape_preserved_with_nans(self, data_with_nans):
        """Output row count must equal input row count even with NaN rows."""
        pca = MaskedPCA(n_components=2)
        pca.fit(data_with_nans)
        out = pca.transform(data_with_nans)
        assert out.shape[0] == data_with_nans.shape[0]

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_all_nan_rows_raises_value_error(self):
        """
        If the entire column group is NaN, PCA cannot fit.
        This surfaces a clear ValueError so the user knows which group failed.
        """
        X_all_nan = np.full((5, 3), np.nan)
        pca = MaskedPCA(n_components=2)
        with pytest.raises(ValueError, match="No rows without NaNs"):
            pca.fit(X_all_nan)

    def test_partial_nan_fit_ignores_nan_rows(self, data_with_nans):
        """
        PCA is fitted only on the 7 clean rows.
        Verify n_samples_seen_ == 7 (skipping rows 2, 5, 8).
        """
        pca = MaskedPCA(n_components=2)
        pca.fit(data_with_nans)
        assert pca.pca.n_samples_ == 7
