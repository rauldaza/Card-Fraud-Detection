"""
This module contains the FTTransformer model, the FTTransactionDataset class,
and the FTTransformerWrapper class.
"""

# Models deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
# Data: FTTransactionDataset
import pandas as pd
from torch.utils.data import Dataset
# Importing: preprocessing config
import pickle
# Math
import numpy as np
# Sklearn: Wrapper
from sklearn.base import BaseEstimator, ClassifierMixin


class FTTransformer(nn.Module):
    """
    Feature Tokenizer + Transformer (FT-Transformer) for tabular data.

    Each feature — both categorical and numerical — is projected into a shared
    ``d_model``-dimensional token space before being processed by a standard
    Transformer encoder. A learnable [CLS] token is prepended and its contextual
    representation is used for the final binary classification.

    Numerical tokenisation follows the FT-Transformer paper: each continuous
    value ``x`` and its missingness flag ``nan`` contribute additively to the
    token via learned projection vectors, plus a learned bias:

    .. code-block:: text

        token_i = W_num_i * x_i + W_nan_i * nan_i + b_i

    Categorical features are tokenised via per-column ``nn.Embedding`` layers
    stored in a ``ModuleDict`` (keyed by column name).
    """

    def __init__(self, num_cols, cat_cols, d_model, n_head, num_encoder_layers):
        """
        Parameters
        ----------
        num_cols : list of str
            Ordered list of numerical column names.  Length determines the
            number of numerical tokens.
        cat_cols : dict
            Mapping of ``{column_name: n_categories}`` for every categorical
            feature.  Each entry creates a dedicated ``nn.Embedding`` of size
            ``(n_categories, d_model)``.
        d_model : int
            Embedding / token dimensionality shared across all features and the
            Transformer encoder.
        n_head : int
            Number of attention heads in each ``TransformerEncoderLayer``.
            Must divide ``d_model`` evenly.
        num_encoder_layers : int
            Number of stacked ``TransformerEncoderLayer`` blocks.
        """
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.num_encoder_layers = num_encoder_layers

        self.cat_embedding_layers = nn.ModuleDict(
            {
                cat_col: nn.Embedding(num_embeddings=num_cat, embedding_dim=self.d_model)
                for cat_col, num_cat in cat_cols.items()
            }
        )

        self.num_projection_tensor = nn.Parameter(torch.rand(len(num_cols), self.d_model))

        self.nan_projection_tensor = nn.Parameter(torch.rand(len(num_cols), self.d_model))

        self.bias_tensor = nn.Parameter(torch.rand(len(num_cols), self.d_model))

        self.cls_token = nn.Parameter(data=torch.rand(1, 1, d_model))

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_head, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=self.num_encoder_layers)

        self.linear = nn.Linear(in_features=d_model, out_features=1)

    def forward(self, x_num, x_nan, x_cat):
        """
        Forward pass of the FTTransformer.

        Parameters
        ----------
        x_num : torch.Tensor
            Float tensor of shape ``(batch, n_num)`` with imputed, scaled
            numerical values.
        x_nan : torch.Tensor
            Float tensor of shape ``(batch, n_num)`` with missingness indicators
            (``1.0`` where the original value was NaN, ``0.0`` otherwise).
        x_cat : torch.Tensor
            Long tensor of shape ``(batch, n_cat)`` with ordinal-encoded
            categorical indices (0 reserved for unknown after ``shift_plus_one``).

        Returns
        -------
        torch.Tensor
            Float tensor of shape ``(batch, 1)`` — raw logit for binary
            classification.  Apply ``torch.sigmoid`` to obtain a probability.
        """
        cat_tokens = [
            embedding_layer(x_cat[:, i])
            for i, embedding_layer in enumerate(self.cat_embedding_layers.values())
        ]

        cat_tokens = torch.stack(cat_tokens, dim=1)

        num_tokens = self.num_projection_tensor * x_num.unsqueeze(-1) \
                    + self.nan_projection_tensor * x_nan.unsqueeze(-1) \
                    + self.bias_tensor

        batch_size = cat_tokens.size(0)
        cls_batch_token = self.cls_token.expand(batch_size, -1, -1)

        tokens = torch.cat((cls_batch_token, cat_tokens, num_tokens), dim=1)

        contextual_tokens = self.transformer_encoder(tokens)

        cls_contextual_tokens = contextual_tokens[:, 0, :]

        return self.linear(cls_contextual_tokens)


class FTTransactionDataset(Dataset):
    """
    Dataset class for training the FTTransformer.

    Loads a feather file, applies the ``ft_preprocessing`` pipeline, and
    exposes three separate tensors (``x_num``, ``x_nan``, ``x_cat``) that
    match the ``FTTransformer.forward`` signature.

    The ``ft_preprocessing`` numerical output is structured as:

    .. code-block:: text

        [ values_0 … values_{n-1} | indicators_0 … indicators_{n-1} ]
        └────────── n_num ──────────┘└──────────── n_num ─────────────┘

    where ``n_num = len(config['num_cols'])``.  The split is computed from
    ``config['num']`` (the ``(start, end)`` index range of the full numerical
    block) and ``n_num``.
    """

    def __init__(self, feather_path, config_pkl_path, target_col=None):
        """
        Parameters
        ----------
        feather_path : str
            Path to the ``.feather`` file with raw transaction data.
        config_pkl_path : str
            Path to the ``preprocessing_config.pkl`` produced by
            ``ft_preprocessing.py``.  Must contain the keys:
            ``preprocessor``, ``num``, ``cat``, ``num_cols``, ``cat_cols``.
        target_col : str, optional
            Name of the target column.  If provided and present in the data,
            labels are loaded as a ``torch.long`` tensor; otherwise ``y`` is
            ``None``.
        """
        with open(config_pkl_path, 'rb') as f:
            config = pickle.load(f)

        self.preprocessor = config['preprocessor']
        self.num_idx = config['num']   # (start, end) of the full num block
        self.cat_idx = config['cat']   # (start, end) of the cat block
        self.num_cols = config['num_cols']

        # Load the data and split X / y
        df = pd.read_feather(feather_path)

        if target_col and target_col in df.columns:
            self.y = torch.tensor(df[target_col].values, dtype=torch.long)
            X_raw = df.drop(columns=[target_col])
        else:
            self.y = None
            X_raw = df

        print("Preprocessing data... this may take a moment.")
        X_processed = self.preprocessor.transform(X_raw)

        # The numerical block is [values | indicators], each half has n_num cols.
        n_num = len(self.num_cols)
        num_start = self.num_idx[0]

        self.Xp_num = torch.tensor(
            X_processed[:, num_start : num_start + n_num],
            dtype=torch.float32,
        )
        self.Xp_nan = torch.tensor(
            X_processed[:, num_start + n_num : self.num_idx[1]],
            dtype=torch.float32,
        )
        self.Xp_cat = torch.tensor(
            X_processed[:, self.cat_idx[0] : self.cat_idx[1]],
            dtype=torch.long,
        )

    def __len__(self):
        return len(self.Xp_num)

    def __getitem__(self, idx):
        """
        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        tuple
            ``(x_num, x_nan, x_cat, y)`` if labels are available, else
            ``(x_num, x_nan, x_cat)``.
        """
        x_num = self.Xp_num[idx]
        x_nan = self.Xp_nan[idx]
        x_cat = self.Xp_cat[idx]

        if self.y is not None:
            return x_num, x_nan, x_cat, self.y[idx]

        print('Missing <y> value')
        return x_num, x_nan, x_cat


class FTTransformerWrapper(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn-compatible wrapper for ``FTTransformer``.

    Accepts a preprocessed 2-D numpy array (as produced by the
    ``ft_preprocessing`` pipeline) and handles the internal split into
    ``x_num``, ``x_nan``, and ``x_cat`` before calling the model.
    """

    def __init__(self, model: FTTransformer, num_idx, cat_idx, n_num_cols, batch_size=64, device=None):
        """
        Parameters
        ----------
        model : FTTransformer
            Trained ``FTTransformer`` instance.
        num_idx : tuple of int
            ``(start, end)`` index range of the full numerical block in the
            preprocessed array (value of ``config['num']``).
        cat_idx : tuple of int
            ``(start, end)`` index range of the categorical block
            (value of ``config['cat']``).
        n_num_cols : int
            Number of numerical columns (``len(config['num_cols'])``). Used to
            split the numerical block into values and indicators halves.
        batch_size : int, optional
            Number of samples processed per forward pass. Default ``64``.
        device : str, optional
            Device string (e.g. ``'cuda'``, ``'cpu'``). Defaults to CUDA if
            available, otherwise CPU.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.model.eval()

        self.num_idx = num_idx
        self.cat_idx = cat_idx
        self.n_num_cols = n_num_cols
        self.batch_size = batch_size

    def fit(self, X, y=None):
        """The fit method is not implemented for this wrapper."""
        return self

    def predict_proba(self, X):
        """
        Predict fraud probability for each sample in ``X``.

        Parameters
        ----------
        X : np.ndarray
            Preprocessed feature matrix of shape ``(n_samples, n_features)``,
            as output by the ``ft_preprocessing`` pipeline.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_samples, 2)`` with columns
            ``[P(not fraud), P(fraud)]``.
        """
        X = np.array(X)
        n_samples = X.shape[0]
        all_probs = []

        num_start = self.num_idx[0]
        n_num = self.n_num_cols

        with torch.no_grad():
            for i in range(0, n_samples, self.batch_size):
                X_batch = X[i : i + self.batch_size]

                Xb_num = torch.tensor(
                    X_batch[:, num_start : num_start + n_num],
                    dtype=torch.float32,
                ).to(self.device)
                Xb_nan = torch.tensor(
                    X_batch[:, num_start + n_num : self.num_idx[1]],
                    dtype=torch.float32,
                ).to(self.device)
                Xb_cat = torch.tensor(
                    X_batch[:, self.cat_idx[0] : self.cat_idx[1]],
                    dtype=torch.long,
                ).to(self.device)

                logits = self.model(Xb_num, Xb_nan, Xb_cat)        # (batch, 1)
                p_fraud = torch.sigmoid(logits).squeeze(1)          # (batch,)
                p_not_fraud = 1.0 - p_fraud

                probs = torch.stack([p_not_fraud, p_fraud], dim=1)
                all_probs.append(probs.cpu().numpy())

        return np.vstack(all_probs)

    def predict(self, X):
        """
        Predict the class label for each sample in ``X``.

        Parameters
        ----------
        X : np.ndarray
            Preprocessed feature matrix of shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            Integer array of shape ``(n_samples,)`` with predicted class labels
            (``0`` = not fraud, ``1`` = fraud).
        """
        probs = self.predict_proba(X)
        return probs.argmax(axis=1)
