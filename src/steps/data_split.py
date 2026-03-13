"""
steps/data_split.py — SageMaker Processing Step: Data Split
============================================================
Reads the full dataset from RDS (PostgreSQL), splits into train/test,
and writes both as .feather files to the SageMaker output channels.

SageMaker mounts:
  /opt/ml/processing/train/   → output: transactions_train.feather
  /opt/ml/processing/test/    → output: transactions_test.feather
"""

import argparse
import os

import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split


def name_normalization(columns):
    """Lowercase and replace hyphens with underscores."""
    if isinstance(columns, str):
        return columns.lower().replace('-', '_')
    return [c.lower().replace('-', '_') for c in columns]


def get_data(host, port, db_user, db_password, limit):
    """
    Pull features from RDS by joining transaction and identity tables directly.
    Identity columns are suffixed with _new to avoid the transactionid collision
    on the join, matching the column names produced during raw table creation.
    Uses a LEFT JOIN so transactions with no identity record are kept (identity
    columns will be NULL for those rows rather than the row being dropped).
    """
    engine = create_engine(
        f'postgresql+psycopg2://{db_user}:{db_password}@{host}:{port}/postgres',
        connect_args={"sslmode": "require"},
    )
    query = """
        SELECT t.*, i.*
        FROM raw.train_transaction AS t
        LEFT JOIN raw.train_identity AS i
            USING (transactionid)
    """
    if limit is not None:
        query += f" LIMIT {limit}"

    df = pd.read_sql(query, engine)
    return df


def main(args):
    df = get_data(args.db_host, args.db_port, args.db_user, args.db_password, args.limit)
    print(f"Loaded {len(df):,} rows from RDS")
    df.columns = name_normalization(df.columns.tolist())
    y = df[name_normalization(args.target)]
    X = df.drop(columns=name_normalization([args.target, args.id_col]))

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y if not args.no_stratify else None,
    )

    train_path = '/opt/ml/processing/train'
    val_path   = '/opt/ml/processing/test'
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path,   exist_ok=True)

    train_df = pd.concat([X_train, y_train], axis=1)
    val_df   = pd.concat([X_val,   y_val],   axis=1)

    train_df.to_feather(f'{train_path}/transactions_train.feather')
    val_df.to_feather(f'{val_path}/transactions_test.feather')

    print(f"  Train: {len(train_df):,} rows → {train_path}/transactions_train.feather")
    print(f"  Test:  {len(val_df):,} rows  → {val_path}/transactions_test.feather")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--db-host',      type=str, required=True)
    # NOTE: --db-port is declared as int here. build_split_args() in
    # MLOps_pipeline.py passes vpc["rds_port"] which is already a string,
    # but argparse will correctly convert it via type=int before main() sees it.
    parser.add_argument('--db-port',      type=int, default=5432)
    parser.add_argument('--db-user',      type=str, default='postgres')
    parser.add_argument('--db-password',  type=str, default='password')
    parser.add_argument('--target',       default='isFraud')
    parser.add_argument('--id-col',       default='TransactionID')
    parser.add_argument('--limit',        type=int, default=None)
    parser.add_argument('--test-size',    type=float, default=0.2)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--no-stratify',  action='store_true', default=False)

    args = parser.parse_args()

    main(args)
