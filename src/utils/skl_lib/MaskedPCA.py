import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA

class MaskedPCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=0.9):
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)
        
    def fit(self, X, y=None):
        X_array = np.array(X)
        mask = ~np.isnan(X_array).any(axis=1)
        
        if np.any(mask):
            self.pca.fit(X_array[mask])
        else:
            raise ValueError("No rows without NaNs found in this column group. PCA cannot fit.")
        return self
    
    def transform(self, X):
        X_array = np.array(X)
        mask = ~np.isnan(X_array).any(axis=1)
        
        # FIX: self.pca.n_components_ is the INTEGER number of components found
        # whereas self.n_components is the FLOAT 0.9
        n_outputs = self.pca.n_components_
        
        # Create the output skeleton
        output = np.full((X_array.shape[0], n_outputs), np.nan)
        
        if np.any(mask):
            output[mask] = self.pca.transform(X_array[mask])
            
        return output