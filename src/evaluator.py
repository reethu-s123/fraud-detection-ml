"""
Model Evaluation Module
Evaluates fraud detection models using appropriate metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, f1_score, precision_score, recall_score,
    roc_auc_score, fbeta_score
)
from sklearn.ensemble import RandomForestClassifier, IsolationForest
import xgboost as xgb
from typing import Dict, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate fraud detection models."""
    
    def __init__(self, cost_false_negative: float = 500,
                cost_false_positive: float = 1):
        """
        Initialize evaluator.
        
        Args:
            cost_false_negative: Cost of missing fraud (business impact)
            cost_false_positive: Cost of false alarm (customer inconvenience)
        """
        self.cost_fn = cost_false_negative
        self.cost_fp = cost_false_positive
        self.results = {}
    
    def evaluate(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                model_name: str = 'model', threshold: float = 0.5) -> Dict:
        """
        Evaluate model on test set.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of model
            threshold: Decision threshold for fraud classification
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"\nEvaluating {model_name}...")
        
        # Get predictions
        if isinstance(model, IsolationForest):
            # Isolation Forest returns -1 for anomalies, 1 for normal
            predictions_raw = model.predict(X_test)
            y_pred = (predictions_raw == -1).astype(int)  # Convert to 0/1
            y_pred_proba = model.score_samples(X_test)
            y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())
        else:
            # For probabilistic models
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Compute metrics
        metrics = self._compute_metrics(y_test, y_pred, y_pred_proba, model_name)
        
        self.results[model_name] = metrics
        return metrics
    
    def _compute_metrics(self, y_test: np.ndarray, y_pred: np.ndarray,
                        y_pred_proba: np.ndarray,
                        model_name: str) -> Dict:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            y_test: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            model_name: Name of model
            
        Returns:
            Dictionary of metrics
        """
        # Confusion Matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # Standard Metrics
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        f2 = fbeta_score(y_test, y_pred, beta=2, zero_division=0)  # Emphasize recall
        
        # ROC-AUC
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Precision-Recall AUC
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall_vals, precision_vals)
        
        # Cost-based metrics
        cost_matrix_cost = (fn * self.cost_fn) + (fp * self.cost_fp)
        
        # Rates
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        metrics = {
            'model_name': model_name,
            'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'f2_score': f2,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'false_positive_rate': fpr,
            'false_negative_rate': fnr,
            'true_positive_rate': tpr,
            'total_cost': cost_matrix_cost,
            'predictions_positive': np.sum(y_pred),
            'test_set_size': len(y_test),
        }
        
        self._print_metrics(metrics)
        return metrics
    
    def _print_metrics(self, metrics: Dict) -> None:
        """Print formatted metrics."""
        cm = metrics['confusion_matrix']
        print(f"\n{'='*60}")
        print(f"{metrics['model_name'].upper()}")
        print(f"{'='*60}")
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives (TN): {cm['tn']:,}")
        print(f"  False Positives (FP): {cm['fp']:,}")
        print(f"  False Negatives (FN): {cm['fn']:,}")
        print(f"  True Positives (TP): {cm['tp']:,}")
        print(f"\nPerformance Metrics:")
        print(f"  Precision: {metrics['precision']:.4f} (of flagged, how many are actual fraud)")
        print(f"  Recall: {metrics['recall']:.4f} (of actual fraud, how many detected)")
        print(f"  F1-Score: {metrics['f1_score']:.4f} (balanced precision-recall)")
        print(f"  F2-Score: {metrics['f2_score']:.4f} (emphasizes recall)")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f} (discrimination ability)")
        print(f"  PR-AUC: {metrics['pr_auc']:.4f} (precision-recall curve)")
        print(f"\nError Rates:")
        print(f"  False Positive Rate: {metrics['false_positive_rate']:.4f}")
        print(f"  False Negative Rate: {metrics['false_negative_rate']:.4f}")
        print(f"  True Positive Rate: {metrics['true_positive_rate']:.4f}")
        print(f"\nCost Analysis (Cost_FN={self.cost_fn}, Cost_FP={self.cost_fp}):")
        print(f"  Total Cost: ${metrics['total_cost']:,.2f}")
        print(f"  Avg Cost per Transaction: ${metrics['total_cost']/metrics['test_set_size']:.4f}")
        print(f"\nPredictions:")
        print(f"  Flagged as Fraud: {metrics['predictions_positive']:,}")
        print(f"  Test Set Size: {metrics['test_set_size']:,}")
        print(f"{'='*60}\n")
    
    def compare_models(self, results: Optional[Dict] = None) -> pd.DataFrame:
        """
        Compare multiple models side by side.
        
        Args:
            results: Results dictionary (uses self.results if None)
            
        Returns:
            DataFrame comparing all models
        """
        if results is None:
            results = self.results
        
        comparison_data = []
        for model_name, metrics in results.items():
            comparison_data.append({
                'Model': model_name,
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1': metrics['f1_score'],
                'F2': metrics['f2_score'],
                'ROC-AUC': metrics['roc_auc'],
                'PR-AUC': metrics['pr_auc'],
                'FPR': metrics['false_positive_rate'],
                'FNR': metrics['false_negative_rate'],
                'Total Cost': metrics['total_cost'],
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        print("\n" + "="*100)
        print("MODEL COMPARISON")
        print("="*100)
        print(df_comparison.to_string(index=False))
        print("="*100 + "\n")
        
        return df_comparison
    
    def get_feature_importance(self, model: Any, feature_names: list,
                             top_n: int = 10) -> pd.DataFrame:
        """
        Extract feature importance from tree-based models.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            top_n: Number of top features to return
            
        Returns:
            DataFrame of feature importance
        """
        if isinstance(model, (RandomForestClassifier, xgb.XGBClassifier)):
            importances = model.feature_importances_
        else:
            logger.warning(f"Model type {type(model)} doesn't support feature_importances")
            return None
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(top_n)
        
        print(f"\nTop {top_n} Important Features:")
        print(importance_df.to_string(index=False))
        
        return importance_df
