"""
Real-time Fraud Detection Interface
Provides scoring interface for incoming transactions.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class FraudDetector:
    """
    Real-time fraud detection interface.
    Scores transactions and returns fraud risk assessment.
    """
    
    def __init__(self, model: Any, preprocessor: Any,
                feature_names: list, threshold: float = 0.5):
        """
        Initialize fraud detector.
        
        Args:
            model: Trained ML model
            preprocessor: Fitted data preprocessor
            feature_names: List of feature names
            threshold: Decision threshold for fraud (0-1)
        """
        self.model = model
        self.preprocessor = preprocessor
        self.feature_names = feature_names
        self.threshold = threshold
        
    def score_transaction(self, transaction: Dict[str, float]) -> Dict:
        """
        Score a single transaction.
        
        Args:
            transaction: Dictionary with feature values
            
        Returns:
            Dictionary with fraud score and classification
        """
        try:
            # Validate transaction
            self._validate_transaction(transaction)
            
            # Extract features in order
            features = np.array([transaction.get(f, 0) for f in self.feature_names]).reshape(1, -1)
            
            # Scale features
            features_scaled = self.preprocessor.transform_features(features)
            
            # Get prediction
            fraud_score = self.model.predict_proba(features_scaled)[0, 1]
            fraud_flag = fraud_score >= self.threshold
            
            risk_level = self._assess_risk(fraud_score)
            
            result = {
                'fraud_score': float(fraud_score),
                'is_fraud': bool(fraud_flag),
                'risk_level': risk_level,
                'threshold_used': self.threshold,
                'amount': transaction.get('Amount', 0),
                'confidence': float(abs(fraud_score - 0.5) * 2)  # 0-1 confidence
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error scoring transaction: {e}")
            raise
    
    def score_batch(self, transactions: List[Dict]) -> List[Dict]:
        """
        Score multiple transactions.
        
        Args:
            transactions: List of transaction dictionaries
            
        Returns:
            List of scoring results
        """
        results = []
        for transaction in transactions:
            result = self.score_transaction(transaction)
            results.append(result)
        
        logger.info(f"Scored {len(transactions)} transactions")
        return results
    
    def score_dataframe(self, df) -> np.ndarray:
        """
        Score all transactions in a DataFrame.
        
        Args:
            df: DataFrame with feature columns
            
        Returns:
            Array of fraud scores
        """
        features = df[self.feature_names].values
        features_scaled = self.preprocessor.transform_features(features)
        fraud_scores = self.model.predict_proba(features_scaled)[:, 1]
        
        return fraud_scores
    
    def _validate_transaction(self, transaction: Dict) -> None:
        """
        Validate transaction has required features.
        
        Args:
            transaction: Transaction dictionary
            
        Raises:
            ValueError: If required features missing
        """
        if not isinstance(transaction, dict):
            raise ValueError("Transaction must be a dictionary")
        
        if 'Amount' not in transaction:
            raise ValueError("Transaction must contain 'Amount' field")
    
    def _assess_risk(self, fraud_score: float) -> str:
        """
        Categorize fraud score into risk levels.
        
        Args:
            fraud_score: Fraud probability (0-1)
            
        Returns:
            Risk level string
        """
        if fraud_score < 0.2:
            return "LOW"
        elif fraud_score < 0.5:
            return "MEDIUM"
        elif fraud_score < 0.8:
            return "HIGH"
        else:
            return "VERY_HIGH"
    
    def set_threshold(self, threshold: float) -> None:
        """
        Update decision threshold.
        
        Args:
            threshold: New threshold (0-1)
            
        Raises:
            ValueError: If threshold not in [0, 1]
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        
        self.threshold = threshold
        logger.info(f"Updated threshold to {threshold}")
    
    def get_threshold(self) -> float:
        """Get current decision threshold."""
        return self.threshold
