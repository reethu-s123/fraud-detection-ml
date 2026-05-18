"""
Data Loading and Validation Module
Handles ingestion and validation of transaction data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and validate credit card transaction data."""
    
    def __init__(self, data_path: str, random_state: int = 42):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to CSV file containing transaction data
            random_state: Random seed for reproducibility
        """
        self.data_path = Path(data_path)
        self.random_state = random_state
        self.df = None
        self.feature_names = None
        self.target_name = 'Class'
        
    def load(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """
        Load transaction data from CSV.
        
        Args:
            nrows: Maximum number of rows to load (for testing)
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If data format is invalid
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        try:
            logger.info(f"Loading data from {self.data_path}...")
            self.df = pd.read_csv(self.data_path, nrows=nrows)
            logger.info(f"Loaded {len(self.df)} transactions")
            
            # Validate required columns
            self._validate_schema()
            
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _validate_schema(self) -> None:
        """Validate that required columns exist."""
        required_cols = {self.target_name, 'Time', 'Amount'}
        missing_cols = required_cols - set(self.df.columns)
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        logger.info("✓ Schema validation passed")
    
    def get_statistics(self) -> Dict:
        """Get comprehensive data statistics."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        fraud_count = (self.df[self.target_name] == 1).sum()
        normal_count = (self.df[self.target_name] == 0).sum()
        fraud_rate = fraud_count / len(self.df) * 100
        
        stats = {
            'total_transactions': len(self.df),
            'fraud_transactions': fraud_count,
            'normal_transactions': normal_count,
            'fraud_rate_percent': fraud_rate,
            'feature_count': self.df.shape[1] - 1,  # Exclude target
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2,
            'missing_values': self.df.isnull().sum().sum(),
            'amount_stats': {
                'mean': self.df['Amount'].mean(),
                'median': self.df['Amount'].median(),
                'min': self.df['Amount'].min(),
                'max': self.df['Amount'].max(),
                'std': self.df['Amount'].std(),
            }
        }
        
        return stats
    
    def print_summary(self) -> None:
        """Print formatted data summary."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)
        print(f"Total Transactions: {stats['total_transactions']:,}")
        print(f"Fraudulent: {stats['fraud_transactions']:,} ({stats['fraud_rate_percent']:.2f}%)")
        print(f"Normal: {stats['normal_transactions']:,} ({100-stats['fraud_rate_percent']:.2f}%)")
        print(f"Features: {stats['feature_count']}")
        print(f"Memory Usage: {stats['memory_usage_mb']:.2f} MB")
        print(f"Missing Values: {stats['missing_values']}")
        print(f"\nAmount Statistics:")
        print(f"  Mean: ${stats['amount_stats']['mean']:.2f}")
        print(f"  Median: ${stats['amount_stats']['median']:.2f}")
        print(f"  Range: ${stats['amount_stats']['min']:.2f} - ${stats['amount_stats']['max']:.2f}")
        print(f"  Std Dev: ${stats['amount_stats']['std']:.2f}")
        print("="*60 + "\n")
    
    def split_features_target(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Split features and target variable."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        self.feature_names = [col for col in self.df.columns 
                             if col not in [self.target_name, 'Time']]
        
        X = self.df[self.feature_names]
        y = self.df[self.target_name]
        
        logger.info(f"Extracted {len(self.feature_names)} features")
        return X, y
    
    def get_feature_names(self) -> list:
        """Get list of feature column names."""
        if self.feature_names is None:
            self.split_features_target()
        return self.feature_names
