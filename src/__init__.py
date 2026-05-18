"""
Credit Card Fraud Detection System
A production-ready ML system for detecting fraudulent transactions in real-time.
"""

__version__ = "1.0.0"
__author__ = "Fraud Detection Team"

from src.data_loader import DataLoader
from src.preprocessor import DataPreprocessor
from src.model_builder import ModelBuilder
from src.evaluator import ModelEvaluator
from src.fraud_detector import FraudDetector

__all__ = [
    'DataLoader',
    'DataPreprocessor',
    'ModelBuilder',
    'ModelEvaluator',
    'FraudDetector',
]
