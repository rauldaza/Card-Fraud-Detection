"""
steps/tab_preprocessing.py — SageMaker Processing Step: Tabular Preprocessing
==============================================================================
Fits the preprocessing pipeline (PCA, imputation, scaling, ordinal encoding)
on the training split and saves preprocessing_config.pkl for downstream steps.

SageMaker mounts:
  /opt/ml/processing/train/      → input: transactions_train.feather
  /opt/ml/processing/artifacts/  → output: preprocessing_config.pkl
"""

import os
import re
import pickle
import argparse

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer

from utils.skl_lib.LibPreTabTransformer import MaskedPCA, shift_plus_one


def name_normalization(columns):
    """Lowercase and replace hyphens with underscores."""
    if isinstance(columns, str):
        return columns.lower().replace('-', '_')
    return [c.lower().replace('-', '_') for c in columns]


def main(args):
    # ── Load training data ────────────────────────────────────────────────────
    train_file = '/opt/ml/processing/train/transactions_train.feather'
    df = pd.read_feather(train_file)
    df = df.drop(columns=[name_normalization(args.target)])

    print(f"Loaded {len(df):,} rows from {train_file}")

    # ── Split numerical / categorical columns ─────────────────────────────────
    # FIX: 'integer' is not a valid pandas dtype alias — it silently matches
    # nothing. Integer columns (card types, counts, etc.) must be selected via
    # numpy integer subtypes. We select object/category as categoricals and
    # treat all numeric integer columns as categoricals too (they are encoded
    # ids / counts, not continuous measurements).
    from pandas.api.types import is_integer_dtype, is_object_dtype, is_categorical_dtype, is_string_dtype

    cat_cols = []
    int_cols = []
    num_cols = []

    for c in df.columns:
        # Check for categorical/object/string types
        if is_object_dtype(df[c]) or is_categorical_dtype(df[c]) or is_string_dtype(df[c]):
            cat_cols.append(c)
        # Check for integer types (counts, IDs) and treat as categorical
        elif is_integer_dtype(df[c]):
            int_cols.append(c)
            cat_cols.append(c)
        # Remaining floats are numerical
        elif np.issubdtype(df[c].dtype, np.floating):
            num_cols.append(c)

    print(f"Categorical columns ({len(cat_cols)}): {cat_cols[:5]}...")
    print(f"Numerical columns  ({len(num_cols)}): {num_cols[:5]}...")

    # ── PCA column groups ─────────────────────────────────────────────────────
    cols_V = [c for c in df.columns if re.match(r"^v\d+$", c)]
    cols_C = sorted([c for c in df.columns if re.match(r"^c\d+$", c)])

    nan_series = df[cols_V].isna().sum() if cols_V else pd.Series(dtype=int)
    unique_counts = nan_series.unique() if len(nan_series) > 0 else []

    set_V = {}
    for count in sorted(unique_counts):
        cols = sorted(nan_series[nan_series == count].index.tolist())
        print(f"  V group (NaN={int(count)}): {cols[:5]}...")
        set_V[f'Count V{count}'] = cols
    set_V['Count C'] = cols_C
    print(f"  C group: {cols_C[:5]}...")

    # ── Build preprocessing pipeline ──────────────────────────────────────────
    pca_parallel_steps = []
    for i, (group_name, cols) in enumerate(set_V.items(), 1):
        pca_parallel_steps.append((f'pca_group_{i}', MaskedPCA(n_components=0.9), cols))

    pca_engine = ColumnTransformer(pca_parallel_steps, remainder='passthrough')

    num_pipeline = Pipeline([
        ("pca",     pca_engine),
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0, add_indicator=True)),
        ("scaler",  StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
        ("shifter", FunctionTransformer(shift_plus_one)),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols),
    ])

    # ── Fit preprocessor ──────────────────────────────────────────────────────
    print("Fitting preprocessing pipeline...")
    preprocessing_pipeline = preprocessor.fit(df)

    # ── Compute feature map ───────────────────────────────────────────────────
    column_breakdown = {}
    current_idx = 0

    if "num" in preprocessor.named_transformers_:
        sample_num = preprocessor.named_transformers_['num'].transform(df[num_cols].iloc[:5])
        n_num = sample_num.shape[1]
        column_breakdown['num'] = (current_idx, current_idx + n_num)
        current_idx += n_num

    if "cat" in preprocessor.named_transformers_:
        sample_cat = preprocessor.named_transformers_['cat'].transform(df[cat_cols].iloc[:5])
        n_cat = sample_cat.shape[1]
        column_breakdown['cat'] = (current_idx, current_idx + n_cat)
        current_idx += n_cat

    print(f"Final Feature Map: {column_breakdown}")

    # ── Save preprocessing config ─────────────────────────────────────────────
    preprocessing = {
        'preprocessor': preprocessing_pipeline,
        'num': column_breakdown['num'],
        'cat': column_breakdown['cat'],
    }

    output_dir = '/opt/ml/processing/artifacts'
    os.makedirs(output_dir, exist_ok=True)

    pkl_path = f'{output_dir}/preprocessing_config.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(preprocessing, f)

    print(f"✓ Saved preprocessing config → {pkl_path}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='isFraud')
    args = parser.parse_args()

    main(args)
