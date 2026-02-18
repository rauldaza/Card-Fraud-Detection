import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import pickle

from utils.skl_lib.MaskedPCA import MaskedPCA

import __main__
__main__.MaskedPCA = MaskedPCA

class TabularTransformer(nn.Module):
    def __init__(self, n_categories, n_continuous, n_classes, embed_dim = 16):
        super().__init__()
        

        self.embeddings = nn.ModuleList([
            nn.Embedding(num_cat, embed_dim) for num_cat in n_categories
        ])
        

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        

        flat_dim = (len(n_categories) * embed_dim) + n_continuous
        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )

    def forward(self, x_cat, x_cont):

        embeddings = [embed(x_cat[:, i]) for i, embed in enumerate(self.embeddings)]
        x = torch.stack(embeddings, dim=1) 
        

        x = self.transformer(x)
        x = x.flatten(1) 
        

        combined = torch.cat([x, x_cont], dim=1)
        

        return self.classifier(combined)


class TransactionDataset(Dataset):
    def __init__(self, csv_path, config_pkl_path, target_col=None):

        with open(config_pkl_path, 'rb') as f:
            config = pickle.load(f)
        
        self.preprocessor = config['preprocessor']

        self.num_idx = config['num'] 
        self.cat_idx = config['cat']  


        df = pd.read_csv(csv_path)
        
        if target_col and target_col in df.columns:
            self.y = torch.tensor(df[target_col].values, dtype=torch.long)
            X_raw = df.drop(columns=[target_col])
        else:
            self.y = None
            X_raw = df


        print("Preprocessing data... this may take a moment.")

        X_processed = self.preprocessor.transform(X_raw)
        

        self.Xp_cont = torch.tensor(X_processed[:, self.num_idx[0]:self.num_idx[1]], dtype=torch.float32)
        

        cats = torch.tensor(X_processed[:, self.cat_idx[0]:self.cat_idx[1]], dtype=torch.long)
        self.Xp_cat = torch.clamp(cats, min=0) # Replaces -1 with 0 for unoun values on the preprocesing

    def __len__(self):
        return len(self.Xp_cont)

    def __getitem__(self, idx):

        x_cont = self.Xp_cont[idx]
        x_cat = self.Xp_cat[idx]
        
        if self.y is not None:
            return x_cat, x_cont, self.y[idx]
        return x_cat, x_cont

    def get_n_categories(self):
        """Returns a list of category counts for each categorical column."""

        counts = []
        for i in range(self.Xp_cat.shape[1]):

            counts.append(int(self.Xp_cat[:, i].max()) + 1)
        return counts