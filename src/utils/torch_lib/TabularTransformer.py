'''
This module contains the TabularTransformer model and the TransactionDataset class.
'''

# Models deep L
import torch
import torch.nn as nn
import torch.nn.functional as F
# Data: TransactionDataset
import pandas as pd
from torch.utils.data import Dataset
# Importing: TransactionDataset
import pickle
# Math
import numpy as np
# Sklearn: Wrapper
from sklearn.base import BaseEstimator, ClassifierMixin

# Importing custom classes
from utils.skl_lib.LibPreTabTransformer import MaskedPCA, shift_plus_one



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
        self.n_categories = n_categories
        self.n_continuous = n_continuous
        self.n_classes = n_classes
        self.embed_dim = embed_dim
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


class TabularTransformerWrapper(BaseEstimator, ClassifierMixin):
    '''
    Wrapper for the TabularTransformer model to make it compatible with scikit-learn.
    '''
    def __init__(self, model: TabularTransformer, batch_size=64, device=None):
        '''
        Args:
            model (TabularTransformer): The TabularTransformer model to wrap.
            batch_size (int): The batch size to use for predicting.
            device (str): The device to use for predicting.
        '''
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.model = model.to(self.device) 
        self.model.eval()


        self.n_cont = model.n_continuous
        self.n_cat = len(model.n_categories)
        self.batch_size = batch_size

    def fit(self, X, y=None):
        '''the fit method is not implemented for this wrapper'''
        return self

    def predict_proba(self, X):
        '''
        Predicts the probability of each class for each sample in X.

        Args:
            X (np.ndarray): Matrix of float values with the features.

        Returns:
            np.ndarray: Matrix of float values with the output probabilities.
        '''
        
        X = np.array(X)
        n_samples = X.shape[0]
        all_probs = []
        # disables the DAG of pytorch
        with torch.no_grad():
            # Iterates over the data in batches to manage memory usage
            for i in range(0, n_samples, self.batch_size):
                
                X_batch = X[i : i + self.batch_size]

                Xb_cat = torch.tensor(X_batch[:, self.n_cont:], dtype=torch.long).to(self.device)
                Xb_cont = torch.tensor(X_batch[:, 0:self.n_cont], dtype=torch.float32).to(self.device)
        
                
                logits = self.model(Xb_cat, Xb_cont)
                probs = F.softmax(logits, dim=1)
                
                
                all_probs.append(probs.cpu().numpy())

        
        return np.vstack(all_probs)

    def predict(self, X):
        '''
        Predicts the class for each sample in X.

        Args:
            X (np.ndarray): Matrix of float values with the features.

        Returns:
            np.ndarray: Array of int values with the output classes.
        '''
        probs = self.predict_proba(X)
        return probs.argmax(axis=1)