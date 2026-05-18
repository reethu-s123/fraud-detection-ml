"""
Data Preprocessing Module
Handles scaling, splitting, and imbalance treatment.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from typing import Tuple, Optional, Dict
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocess transaction data for ML modeling."""
    
    def __init__(self, scaler_type: str = 'standard', random_state: int = 42):
        """
        Initialize preprocessor.
        
        Args:
            scaler_type: 'standard', 'robust', or 'minmax'
            random_state: Random seed
        """
        self.random_state = random_state
        self.scaler = self._get_scaler(scaler_type)
        self.feature_names = None
        self.n_features = None
        
    def _get_scaler(self, scaler_type: str):
        """Get appropriate scaler."""
        scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler()
        }
        return scalers.get(scaler_type, StandardScaler())
    
    def fit_transform_features(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fit scaler and transform features.
        
        Args:
            X: Feature dataframe
            
        Returns:
            Scaled features as numpy array
        """
        self.feature_names = X.columns.tolist()
        self.n_features = X.shape[1]
        
        X_scaled = self.scaler.fit_transform(X)
        logger.info(f"✓ Fitted scaler on {X.shape[0]} samples with {X.shape[1]} features")
        
        return X_scaled
    
    def transform_features(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform features using fitted scaler.
        
        Args:
            X: Feature dataframe
            
        Returns:
            Scaled features as numpy array
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_transform_features first.")
        
        X_scaled = self.scaler.transform(X)
        return X_scaled
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   test_size: float = 0.2) -> Tuple:
        """
        Split data into train and test sets.
        
        Args:
            X: Features
            y: Target variable
            test_size: Proportion of test set
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y  # Ensure same fraud rate in both sets
        )
        
        logger.info(f"✓ Split data: {len(X_train)} train, {len(X_test)} test")
        logger.info(f"  Train fraud rate: {y_train.mean()*100:.2f}%")
        logger.info(f"  Test fraud rate: {y_test.mean()*100:.2f}%")
        
        return X_train, X_test, y_train, y_test
    
    def handle_imbalance(self, X_train: np.ndarray, y_train: np.ndarray,
                        strategy: str = 'smote',
                        sampling_ratio: float = 0.3) -> Tuple:
        """
        Handle class imbalance using various techniques.
        
        Args:
            X_train: Training features
            y_train: Training target
            strategy: 'smote', 'oversample', 'undersample', or 'hybrid'
            sampling_ratio: SMOTE sampling ratio (0-1)
            
        Returns:
            Balanced (X_train_balanced, y_train_balanced)
        """
        initial_fraud_count = (y_train == 1).sum()
        initial_normal_count = (y_train == 0).sum()
        
        if strategy == 'smote':
            sampler = SMOTE(
                sampling_strategy=sampling_ratio,
                random_state=self.random_state,
                k_neighbors=5
            )
            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
            method_name = "SMOTE"
            
        elif strategy == 'oversample':
            sampler = RandomOverSampler(random_state=self.random_state)
            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
            method_name = "RandomOverSampler"
            
        elif strategy == 'undersample':
            sampler = RandomUnderSampler(
                sampling_strategy=sampling_ratio,
                random_state=self.random_state
            )
            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
            method_name = "RandomUnderSampler"
            
        elif strategy == 'hybrid':
            # Combine SMOTE and undersampling
            sampler = ImbPipeline([
                ('smote', SMOTE(sampling_strategy=0.5, random_state=self.random_state)),
                ('undersample', RandomUnderSampler(sampling_strategy=0.7, 
                                                   random_state=self.random_state))
            ])
            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
            method_name = "Hybrid (SMOTE + Undersample)"
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        final_fraud_count = (y_resampled == 1).sum()
        final_normal_count = (y_resampled == 0).sum()
        
        logger.info(f"✓ Handled imbalance using {method_name}")
        logger.info(f"  Before: {initial_fraud_count} fraud, {initial_normal_count} normal")
        logger.info(f"  After: {final_fraud_count} fraud, {final_normal_count} normal")
        logger.info(f"  Fraud rate: {initial_fraud_count/len(y_train)*100:.2f}% → {final_fraud_count/len(y_resampled)*100:.2f}%")
        
        return X_resampled, y_resampled
    
    def save_preprocessor(self, path: str) -> None:
        """Save fitted scaler to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, path)
        logger.info(f"✓ Saved preprocessor to {path}")
    
    def load_preprocessor(self, path: str) -> None:
        """Load fitted scaler from disk."""
        self.scaler = joblib.load(path)
        logger.info(f"✓ Loaded preprocessor from {path}")
