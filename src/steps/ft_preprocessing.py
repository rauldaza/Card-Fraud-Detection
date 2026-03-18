"""
steps/ft_preprocessing.py — SageMaker Processing Step: Feature-Transformer Preprocessing
==========================================================================================
Fits the preprocessing pipeline (imputation, scaling, ordinal encoding + NaN indicators)
on the training split and saves preprocessing_config.pkl for downstream steps.

Key differences from tab_preprocessing.py:
  - No PCA on numerical columns.
  - Every numerical column is guaranteed to have a missingness-indicator column,
    achieved via a parallel ColumnTransformer (imputer branch + MissingIndicator branch).
  - Saved config includes `num_cols` (list) and `cat_cols` (dict: col → n_categories).

SageMaker mounts:
  /opt/ml/processing/train/      → input: transactions_train.feather
  /opt/ml/processing/artifacts/  → output: preprocessing_config.pkl
"""

import os
import pickle
import argparse

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer, MissingIndicator

from utils.skl_lib.LibPreTabTransformer import shift_plus_one


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
    # Integer columns (encoded IDs, counts) are treated as categorical.
    # Only floating-point columns go through the numerical pipeline.
    from pandas.api.types import is_integer_dtype, is_object_dtype, is_categorical_dtype, is_string_dtype

    cat_cols = []
    num_cols = []

    for c in df.columns:
        if is_object_dtype(df[c]) or is_categorical_dtype(df[c]) or is_string_dtype(df[c]):
            cat_cols.append(c)
        elif is_integer_dtype(df[c]):
            cat_cols.append(c)
        elif np.issubdtype(df[c].dtype, np.floating):
            num_cols.append(c)


    print(f"Categorical columns ({len(cat_cols)}): {cat_cols[:5]}...")
    print(f"Numerical columns  ({len(num_cols)}): {num_cols[:5]}...")

    # ── Build preprocessing pipeline ──────────────────────────────────────────

    # Numerical: parallel branches so every column gets an indicator,
    # even if it had no NaNs in training.
    #
    #   ┌─ values branch ──────────────────────────────────┐
    #   │  SimpleImputer(constant=0) → StandardScaler      │
    #   └──────────────────────────────────────────────────┘  → hstack → n_num × 2 columns
    #   ┌─ indicator branch ───────────────────────────────┐
    #   │  MissingIndicator(features='all')                │
    #   └──────────────────────────────────────────────────┘

    values_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("scaler",  StandardScaler()),
    ])

    num_pipeline = ColumnTransformer([
        ("values",     values_pipeline,                  num_cols),
        ("indicators", MissingIndicator(features='all'), num_cols),
    ])

    # Categorical: impute → ordinal-encode → shift so 0 is reserved for unknowns
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

    # ── Compute cat_cols cardinality map ──────────────────────────────────────
    # For each categorical column, record how many categories the encoder learned.
    # +1 to account for the unknown-category slot (index 0 after shift_plus_one).
    encoder = preprocessing_pipeline.named_transformers_['cat'].named_steps['encoder']
    cat_cardinality = {
        col: len(cats) + 1
        for col, cats in zip(cat_cols, encoder.categories_)
    }

    print(f"Categorical cardinalities (first 5): {dict(list(cat_cardinality.items())[:5])}")

    # ── Save preprocessing config ─────────────────────────────────────────────
    preprocessing = {
        'preprocessor': preprocessing_pipeline,
        'num':      column_breakdown['num'],
        'cat':      column_breakdown['cat'],
        'num_cols': num_cols,
        'cat_cols': cat_cardinality,
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
