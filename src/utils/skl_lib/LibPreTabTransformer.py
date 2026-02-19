'''
Necesary libaries for the TabularTransformer
'''
# Math
import numpy as np
# Scikit-learn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA

def shift_plus_one(X):
    """Shifts the values of X by adding 1 to each value."""
    return X + 1

class MaskedPCA(BaseEstimator, TransformerMixin):
    """
    A PCA transformer that handles missing values (NaNs) by masking.
    Fits the PCA only on rows without missing values and transforms only those rows.
    """

    def __init__(self, n_components=0.9):
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)
        
    def fit(self, X, y=None):
        """
        Fits the PCA only on rows without missing values.
        """
        X_array = np.array(X)
        mask = ~np.isnan(X_array).any(axis=1)
        
        if np.any(mask):
            self.pca.fit(X_array[mask])
        else:
            raise ValueError("No rows without NaNs found in this column group. PCA cannot fit.")
        return self
    
    def transform(self, X):
        '''
        Transforms only the rows without missing values 
        and the rows with missing values are filled with NaN.
        '''
        X_array = np.array(X)
        mask = ~np.isnan(X_array).any(axis=1)
        

        n_outputs = self.pca.n_components_
        

        output = np.full((X_array.shape[0], n_outputs), np.nan)
        
        if np.any(mask):
            output[mask] = self.pca.transform(X_array[mask])
            
        return output