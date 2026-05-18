"""
Main Training Pipeline
Orchestrates the complete ML workflow: data loading, preprocessing,
model training, and evaluation.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import DataLoader
from src.preprocessor import DataPreprocessor
from src.model_builder import ModelBuilder
from src.evaluator import ModelEvaluator
from src.fraud_detector import FraudDetector
from src.utils import setup_logging

# Setup logging
logger = setup_logging(log_level='INFO')


def main():
    """
    Execute complete training pipeline.
    """
    print("\n" + "="*80)
    print("CREDIT CARD FRAUD DETECTION - TRAINING PIPELINE")
    print("="*80 + "\n")
    
    # Paths
    data_path = "data/raw/creditcard.csv"
    preprocessor_path = "data/models/preprocessor.pkl"
    
    # ============================================================================
    # 1. LOAD DATA
    # ============================================================================
    print("\n[1/6] LOADING DATA...")
    try:
        data_loader = DataLoader(data_path)
        df = data_loader.load()
        data_loader.print_summary()
    except FileNotFoundError:
        logger.error(f"Data file not found at {data_path}")
        logger.error("Please download the dataset from: https://www.kaggle.com/mlg-ulb/creditcardfraud")
        logger.error("And place creditcard.csv in data/raw/ directory")
        return
    
    # ============================================================================
    # 2. EXTRACT FEATURES AND TARGET
    # ============================================================================
    print("\n[2/6] EXTRACTING FEATURES AND TARGET...")
    X, y = data_loader.split_features_target()
    feature_names = data_loader.get_feature_names()
    logger.info(f"Features: {len(feature_names)}, Target: {y.name}")
    
    # ============================================================================
    # 3. PREPROCESS DATA
    # ============================================================================
    print("\n[3/6] PREPROCESSING DATA...")
    preprocessor = DataPreprocessor(scaler_type='standard')
    X_scaled = preprocessor.fit_transform_features(X)
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(X_scaled, y.values)
    
    # Handle imbalance on training set only
    X_train_balanced, y_train_balanced = preprocessor.handle_imbalance(
        X_train, y_train,
        strategy='smote',
        sampling_ratio=0.3
    )
    
    # Save preprocessor for inference
    preprocessor.save_preprocessor(preprocessor_path)
    
    # ============================================================================
    # 4. BUILD MODELS
    # ============================================================================
    print("\n[4/6] BUILDING MODELS...")
    model_builder = ModelBuilder(random_state=42)
    models = model_builder.train_all_models(X_train_balanced, y_train_balanced)
    
    for model_name in models:
        model_builder.save_model(model_name, f"data/models/{model_name}.pkl")
    
    # ============================================================================
    # 5. EVALUATE MODELS
    # ============================================================================
    print("\n[5/6] EVALUATING MODELS...")
    evaluator = ModelEvaluator(cost_false_negative=500, cost_false_positive=1)
    
    results = {}
    for model_name, model in models.items():
        results[model_name] = evaluator.evaluate(
            model, X_test, y_test,
            model_name=model_name.replace('_', ' ').title(),
            threshold=0.5
        )
    
    # Compare models
    comparison_df = evaluator.compare_models(results)
    
    # Feature importance for tree-based models
    logger.info("\nExtracting feature importance...")
    rf_model = model_builder.get_model('random_forest')
    evaluator.get_feature_importance(rf_model, feature_names, top_n=15)
    
    xgb_model = model_builder.get_model('xgboost')
    evaluator.get_feature_importance(xgb_model, feature_names, top_n=15)
    
    # ============================================================================
    # 6. TEST FRAUD DETECTOR
    # ============================================================================
    print("\n[6/6] TESTING FRAUD DETECTOR...")
    
    # Load preprocessor
    preprocessor.load_preprocessor(preprocessor_path)
    
    # Use best model (XGBoost)
    best_model = xgb_model
    
    # Create fraud detector
    detector = FraudDetector(
        model=best_model,
        preprocessor=preprocessor,
        feature_names=feature_names,
        threshold=0.5
    )
    
    # Test on a few samples
    print("\nSample Predictions on Test Set:")
    print("="*80)
    
    normal_indices = y_test[y_test == 0].index.tolist()[:3]
    fraud_indices = y_test[y_test == 1].index.tolist()[:3]
    
    test_indices = normal_indices + fraud_indices
    
    for idx in test_indices:
        row = df.loc[idx]
        transaction = {col: row[col] for col in feature_names}
        transaction['Amount'] = row['Amount']
        
        result = detector.score_transaction(transaction)
        true_label = "FRAUD" if y_test.loc[idx] == 1 else "NORMAL"
        pred_label = "FRAUD" if result['is_fraud'] else "NORMAL"
        match = "✓" if (true_label == "FRAUD") == result['is_fraud'] else "✗"
        
        print(f"{match} Amount: ${row['Amount']:.2f} | "
              f"True: {true_label} | "
              f"Predicted: {pred_label} | "
              f"Score: {result['fraud_score']:.4f} | "
              f"Risk: {result['risk_level']}")
    
    print("="*80)
    
    # ============================================================================
    # SUMMARY
    # ============================================================================
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\n✓ Data loaded and preprocessed")
    print(f"✓ 4 models trained (Logistic Regression, Random Forest, XGBoost, Isolation Forest)")
    print(f"✓ Models evaluated with fraud-specific metrics")
    print(f"✓ Best model: XGBoost (ROC-AUC: {results['xgboost']['roc_auc']:.4f})")
    print(f"\nArtifacts saved to data/models/:")
    print(f"  - logistic_regression.pkl")
    print(f"  - random_forest.pkl")
    print(f"  - xgboost.pkl")
    print(f"  - isolation_forest.pkl")
    print(f"  - preprocessor.pkl")
    print(f"\nNext steps:")
    print(f"  1. Review model performance in comparison above")
    print(f"  2. Run: python api/app.py (to start API server)")
    print(f"  3. Test predictions via HTTP endpoints")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
