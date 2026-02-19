'''
This module contains the TabularTransformer model and the TransactionDataset class.
'''

# Models deep L
import torch
import torch.nn as nn
# Data
import pandas as pd
from torch.utils.data import Dataset
# Importing
import pickle
# Math
import numpy as np

# Importing custom classes
from utils.skl_lib.LibPreTabTransformer import MaskedPCA, shift_plus_one

# Makes sure that the proper libaries are used for the preprocesing pipeline
import __main__
__main__.MaskedPCA = MaskedPCA
__main__.shift_plus_one = shift_plus_one




class TabularTransformer(nn.Module):
    '''
    TabularTransformer is a neural network that uses transformers to process tabular data.
    It takes as input a set of categorical features and a set of numerical features.
    The categorical features are first embedded in a vector space, then processed by a transformer.
    The numerical features are concatenated with the transformer outputs and then passed through a neural classifier.
    '''
    def __init__(self, n_categories, n_continuous, n_classes, embed_dim = 16):
        '''
        Args:
            n_categories (list): List of the number of categories for each categorical feature.
            n_continuous (int): Number of continuous features.
            n_classes (int): Number of output classes.
            embed_dim (int): Dimension of the embeddings for the categorical features.
        '''
        super().__init__()
        
        # Embeddings definition for the categorical features
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_cat, embed_dim) for num_cat in n_categories
        ])
        
        # Transformer definition
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        

        # Neural classifier definition, works with the numerical features and the transformer outputs
        flat_dim = (len(n_categories) * embed_dim) + n_continuous
        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )

    def forward(self, x_cat, x_cont):
        '''
        Performs a forward operation for the TabularTransformer.

        Args:
            x_cat (torch.Tensor): Matrix of long values with the categorical features.
            x_cont (torch.Tensor): Matrix of float values with the numerical features.

        Returns:
            torch.Tensor: Matrix of float values with the output classes.
        '''

        embeddings = [embed(x_cat[:, i]) for i, embed in enumerate(self.embeddings)]
        x = torch.stack(embeddings, dim=1) 
        

        x = self.transformer(x)
        x = x.flatten(1) 
        

        combined = torch.cat([x, x_cont], dim=1)
        

        return self.classifier(combined)


class TransactionDataset(Dataset):
    '''
    Dataset class for training the TabularTransformer. It performs the necessary preprocessing
    using the corresponding preprocessing pipeline and separates categorical and numerical
    features for training and testing.
    '''

    def __init__(self, csv_path, config_pkl_path, target_col=None):
        '''
        Args:
            csv_path (str): Path to the csv file with the data.
            config_pkl_path (str): Path to the pkl file with the preprocessor and feature indices.
            target_col (str): Name of the target column.
        '''
        with open(config_pkl_path, 'rb') as f:
            config = pickle.load(f)

        # The pkl has the pipeline wich does the preprocesing 
        # and the distinction of the categorical an numerical features
        self.preprocessor = config['preprocessor']

        self.num_idx = config['num'] 
        self.cat_idx = config['cat']  

        # Load the data and splits X and y
        df = pd.read_csv(csv_path)
        
        if target_col and target_col in df.columns:
            self.y = torch.tensor(df[target_col].values, dtype=torch.long)
            X_raw = df.drop(columns=[target_col])
        else:
            self.y = None
            X_raw = df

        # Preprocess the data and separates categorical and numerical features
        print("Preprocessing data... this may take a moment.")

        X_processed = self.preprocessor.transform(X_raw)
        

        self.Xp_cont = torch.tensor(X_processed[:, self.num_idx[0]:self.num_idx[1]], dtype=torch.float32)
        
        self.Xp_cat = torch.tensor(X_processed[:, self.cat_idx[0]:self.cat_idx[1]], dtype=torch.long)

    def __len__(self):
        return len(self.Xp_cont)

    def __getitem__(self, idx):

        x_cont = self.Xp_cont[idx]
        x_cat = self.Xp_cat[idx]
        
        if self.y is not None:
            return x_cat, x_cont, self.y[idx]
        print('Missing <y> value')
        return x_cat, x_cont

    def get_n_categories(self):
        """Returns a list of category counts for each categorical column."""

        counts = []
        for i in range(self.Xp_cat.shape[1]):

            counts.append(int(self.Xp_cat[:, i].max()) + 1)
        return counts