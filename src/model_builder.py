"""
Model Building Module
Builds and trains multiple ML models for fraud detection.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from typing import Dict, Any, Optional, Tuple
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelBuilder:
    """Build and train fraud detection models."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize model builder.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        
    def build_logistic_regression(self, max_iter: int = 1000, 
                                 C: float = 1.0) -> LogisticRegression:
        """
        Build Logistic Regression model.
        
        Args:
            max_iter: Maximum iterations
            C: Inverse regularization strength
            
        Returns:
            Untrained LogisticRegression model
        """
        model = LogisticRegression(
            max_iter=max_iter,
            C=C,
            class_weight='balanced',  # Handle imbalance
            random_state=self.random_state,
            n_jobs=-1,
            solver='lbfgs'
        )
        logger.info("✓ Built Logistic Regression model")
        return model
    
    def build_random_forest(self, n_estimators: int = 100,
                          max_depth: int = 15,
                          min_samples_split: int = 10,
                          min_samples_leaf: int = 4) -> RandomForestClassifier:
        """
        Build Random Forest model.
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split
            min_samples_leaf: Minimum samples per leaf
            
        Returns:
            Untrained RandomForestClassifier model
        """
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight='balanced',  # Handle imbalance
            random_state=self.random_state,
            n_jobs=-1,
            oob_score=True  # Out-of-bag scoring
        )
        logger.info("✓ Built Random Forest model")
        return model
    
    def build_xgboost(self, n_estimators: int = 100,
                     max_depth: int = 7,
                     learning_rate: float = 0.1,
                     subsample: float = 0.8,
                     colsample_bytree: float = 0.8) -> xgb.XGBClassifier:
        """
        Build XGBoost model.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate (eta)
            subsample: Row sampling ratio
            colsample_bytree: Feature sampling ratio
            
        Returns:
            Untrained XGBClassifier model
        """
        # scale_pos_weight = negative_samples / positive_samples
        # This is approximated based on typical imbalance
        scale_pos_weight = 200  # Adjust based on actual fraud rate
        
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            scale_pos_weight=scale_pos_weight,  # Handle imbalance
            random_state=self.random_state,
            tree_method='hist',  # Faster histogram-based
            verbosity=0,
            n_jobs=-1
        )
        logger.info("✓ Built XGBoost model")
        return model
    
    def build_isolation_forest(self, contamination: float = 0.001,
                             n_estimators: int = 100,
                             max_samples: str = 'auto') -> IsolationForest:
        """
        Build Isolation Forest for anomaly detection.
        
        Args:
            contamination: Expected proportion of outliers
            n_estimators: Number of base estimators
            max_samples: Number of samples to draw
            
        Returns:
            Untrained IsolationForest model
        """
        model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=self.random_state,
            n_jobs=-1
        )
        logger.info("✓ Built Isolation Forest model")
        return model
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """
        Train all models on training data.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Dictionary of trained models
        """
        logger.info("\nTraining all models...")
        
        # Logistic Regression
        logger.info("Training Logistic Regression...")
        self.models['logistic_regression'] = self.build_logistic_regression()
        self.models['logistic_regression'].fit(X_train, y_train)
        
        # Random Forest
        logger.info("Training Random Forest...")
        self.models['random_forest'] = self.build_random_forest()
        self.models['random_forest'].fit(X_train, y_train)
        
        # XGBoost
        logger.info("Training XGBoost...")
        self.models['xgboost'] = self.build_xgboost()
        self.models['xgboost'].fit(X_train, y_train)
        
        # Isolation Forest (unsupervised, so no y_train needed)
        logger.info("Training Isolation Forest...")
        self.models['isolation_forest'] = self.build_isolation_forest()
        self.models['isolation_forest'].fit(X_train)
        
        logger.info("✓ All models trained successfully")
        return self.models
    
    def hyperparameter_tune(self, X_train: np.ndarray, y_train: np.ndarray,
                          model_name: str = 'xgboost',
                          cv_folds: int = 5) -> Any:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training target
            model_name: Model to tune
            cv_folds: Cross-validation folds
            
        Returns:
            Best model
        """
        param_grids = {
            'xgboost': {
                'max_depth': [5, 7, 10],
                'learning_rate': [0.01, 0.1, 0.3],
                'n_estimators': [50, 100, 200]
            },
            'random_forest': {
                'max_depth': [10, 15, 20],
                'n_estimators': [50, 100, 200],
                'min_samples_split': [5, 10, 20]
            },
            'logistic_regression': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'max_iter': [1000, 5000]
            }
        }
        
        if model_name not in param_grids:
            raise ValueError(f"Unknown model: {model_name}")
        
        if model_name == 'xgboost':
            base_model = self.build_xgboost()
        elif model_name == 'random_forest':
            base_model = self.build_random_forest()
        elif model_name == 'logistic_regression':
            base_model = self.build_logistic_regression()
        
        logger.info(f"\nTuning {model_name} hyperparameters...")
        
        grid_search = GridSearchCV(
            base_model,
            param_grids[model_name],
            cv=cv_folds,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"✓ Best parameters: {grid_search.best_params_}")
        logger.info(f"✓ Best ROC-AUC: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def get_model(self, name: str):
        """Get trained model by name."""
        if name not in self.models:
            raise ValueError(f"Model not found: {name}. Available: {list(self.models.keys())}")
        return self.models[name]
    
    def get_all_models(self) -> Dict:
        """Get all trained models."""
        return self.models
    
    def save_model(self, model_name: str, path: str) -> None:
        """
        Save trained model to disk.
        
        Args:
            model_name: Name of model to save
            path: Path to save model
        """
        if model_name not in self.models:
            raise ValueError(f"Model not found: {model_name}")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.models[model_name], path)
        logger.info(f"✓ Saved {model_name} to {path}")
    
    def load_model(self, model_name: str, path: str) -> None:
        """
        Load trained model from disk.
        
        Args:
            model_name: Name to assign to loaded model
            path: Path to model file
        """
        self.models[model_name] = joblib.load(path)
        logger.info(f"✓ Loaded model from {path}")
