import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from rapidfuzz import process, fuzz
import pickle


# --- Custom Transformer Pipes ---
class ColumnRenamer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_mapping):
        self.columns_mapping = columns_mapping

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.rename(columns=self.columns_mapping)
    
    
class RemoveFeature(BaseEstimator, TransformerMixin):
    def __init__(self, drop_features):
        self.drop_features = drop_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_dropped = X.drop(columns=self.drop_features)
        return X_dropped
    
    
class RemoveMissing(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        # We hard-code the columns to ignore
        self.exclude_cols_ = ['status', 'side_sensor']

    def fit(self, X, y=None):
        # Columns to be imputed
        self.impute_cols_ = [c for c in X.columns if c not in self.exclude_cols_]
        # Drop rows with missing values in these columns for fitting
        X_clean = X[self.impute_cols_].dropna()
        self.imputer_ = KNNImputer(n_neighbors=self.n_neighbors)
        self.imputer_.fit(X_clean)

        # Save which rows survived for training data
        self.train_index_ = X_clean.index
        return self

    def fit_transform(self, X, y=None, **fit_params):
        # Fit on the entire X, then return only the "clean" subset from training
        self.fit(X, y)
        return X.loc[self.train_index_].reset_index(drop=True)

    def transform(self, X, y=None):
        # Apply KNN imputation to the impute_cols only, ignoring the excluded ones
        X = X.copy()
        X_impute = X[self.impute_cols_]
        X_imputed = self.imputer_.transform(X_impute)
        X[self.impute_cols_] = X_imputed
        return X.reset_index(drop=True)

    
class RemoveOutlier(BaseEstimator, TransformerMixin):
    def __init__(self, column, threshold):
        self.column = column
        self.threshold = threshold

    def fit(self, X, y=None):
        # Identify inlier rows
        X_inliers = X[X[self.column] < self.threshold]
        self.inlier_index_ = X_inliers.index
        return self

    def fit_transform(self, X, y=None):
        # Return only the inlier subset for training
        self.fit(X, y)
        return X.loc[self.inlier_index_].reset_index(drop=True)

    def transform(self, X, y=None):
        # For test, do not remove any rows
        return X.reset_index(drop=True)
    
    
class RemoveDuplicates(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Identify unique rows in training
        X_unique = X.drop_duplicates()
        self.unique_index_ = X_unique.index
        return self

    def fit_transform(self, X, y=None):
        # After fitting, return only the unique rows for the training set
        self.fit(X, y)
        return X.loc[self.unique_index_].reset_index(drop=True)

    def transform(self, X, y=None):
        # For test (or any subsequent transform), leave duplicates as is
        return X.reset_index(drop=True)
    

class StatusCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, col="status", threshold=80):
        self.col = col
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        # Lowercase the column values.
        X[self.col] = X[self.col].fillna("subcooled").str.lower()
        # Map the entries using fuzzy matching.
        X[self.col] = X[self.col].apply(self.map_category)
        return X

    def map_category(self, entry):
        valid_categories = ["subcooled", "superheated"]
        best_match, score, _ = process.extractOne(entry, valid_categories, scorer=fuzz.ratio)
        if score >= self.threshold:
            return best_match
        else:
            # Assign a default most frequent category
            return "subcooled"   
    
    
class PressureLogTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        if "pressure" in X.columns:
            X["pressure"] = np.log(X['pressure']+0.001)
        return X

    
class CustomStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, exclude_columns=None):
        if exclude_columns is None:
            exclude_columns = []
        self.exclude_columns = exclude_columns
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        # Identify columns to scale: all columns except those in exclude_columns.
        self.std_cols_ = X.drop(columns=self.exclude_columns).columns
        self.scaler.fit(X[self.std_cols_])
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X[self.std_cols_] = self.scaler.transform(X[self.std_cols_])
        return X
    
    
class CustomOHETransformer(BaseEstimator, TransformerMixin):
    def __init__(self, side_dummy_prefix='side'):
        # Default mappings 
        self.status_map = {'subcooled': 0, 'superheated': 1}
        self.side_map = {1: 'back', 2: 'front', 3: 'left', 4: 'top', 5: 'right'}
        self.side_dummy_prefix = side_dummy_prefix

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        # Map status column
        X['status'] = X['status'].map(self.status_map)
        # Map side_sensor column and then perform one-hot encoding
        X['side_sensor'] = X['side_sensor'].map(self.side_map)
        X = pd.get_dummies(X, columns=['side_sensor'], prefix=self.side_dummy_prefix, dtype=int)
        return X
    

# --- Pipeline Variables ---
columns_mapping = {
    'Unnamed: 0': 'Unnamed: 0',
    'Tank Failure Pressure (bar)': 'tfp',
    'Liquid Ratio': 'lr',
    'Tank Width (m)': 'width_tank',
    'Tank Length (m)': 'length_tank',
    'Tank Height (m)': 'height_tank',
    'BLEVE Height (m)': 'height_bleve',
    'Vapour Height (m)': 'height_vapour', 
    'Vapour Temperature (K)': 'temp_vapour', 
    'Liquid Temperature (K)': 'temp_liquid',
    'Obstacle Distance to BLEVE (m)': 'distance',
    'Obstacle Width (m)': 'width_obstacle',
    'Obstacle Height (m)': 'height_obstacle',
    'Obstacle Thickness (m)': 'thickness_obstacle',
    'Obstacle Angle': 'angle_obstacle',
    'Status': 'status',
    'Liquid Critical Pressure (bar)': 'crit_pressure_liquid',
    'Liquid Boiling Temperature (K)': 'boil_temp_liquid',
    'Liquid Critical Temperature (K)': 'crit_temp_liquid',
    'Sensor ID': 'id_sensor',
    'Sensor Position Side': 'side_sensor', 
    'Sensor Position x': 'x_sensor',
    'Sensor Position y': 'y_sensor', 
    'Sensor Position z': 'z_sensor', 
    'Target Pressure (bar)': 'pressure'
}
drop_features = ['id_sensor','distance','height_vapour',
                 'crit_temp_liquid','crit_pressure_liquid','boil_temp_liquid'] 
exclude_columns_for_scaling = ['status', 'side_sensor', 'pressure']


# --- Full Pipeline ---
full_pipe = Pipeline([
    ('column_renamer', ColumnRenamer(columns_mapping=columns_mapping)),
    ('remove_features', RemoveFeature(drop_features=drop_features)),
    ('remove_missing', RemoveMissing(n_neighbors=5)),
    ('remove_duplicates', RemoveDuplicates()),
    ('remove_outlier', RemoveOutlier(column='tfp', threshold=1000)),
    ('status_cleaner', StatusCleaner(col='status', threshold=80)),
    ('pressure_log', PressureLogTransformer()),
    ('scaler', CustomStandardScaler(exclude_columns=exclude_columns_for_scaling)),
    ('ohe_transformer', CustomOHETransformer())
])
    

# --- Pickling Pipeline ---
pickle.dump(full_pipe, open('data/pipeline.pkl', 'wb'))
    
    
    
    
    
    
    
