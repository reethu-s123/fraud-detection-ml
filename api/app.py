"""
REST API for Credit Card Fraud Detection
Provides endpoints for real-time transaction scoring.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List

from flask import Flask, request, jsonify
from flask_cors import CORS

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fraud_detector import FraudDetector
from src.preprocessor import DataPreprocessor
from src.utils import setup_logging
import joblib

# Setup
logger = setup_logging(log_level='INFO')
app = Flask(__name__)
CORS(app)

# Global state
fraud_detector = None
model_info = None


def load_model_and_preprocessor():
    """
    Load trained model and preprocessor.
    """
    global fraud_detector, model_info
    
    try:
        # Load model and preprocessor
        model = joblib.load('data/models/xgboost.pkl')
        preprocessor = DataPreprocessor(scaler_type='standard')
        preprocessor.load_preprocessor('data/models/preprocessor.pkl')
        
        # Feature names (from training)
        feature_names = preprocessor.feature_names
        
        if feature_names is None:
            logger.warning("Feature names not found in preprocessor. Using default feature names.")
            feature_names = [f'V{i}' for i in range(1, 29)] + ['Amount']
        
        # Create fraud detector
        fraud_detector = FraudDetector(
            model=model,
            preprocessor=preprocessor,
            feature_names=feature_names,
            threshold=0.5
        )
        
        model_info = {
            'model_name': 'XGBoost Classifier',
            'feature_count': len(feature_names),
            'features': feature_names,
            'decision_threshold': 0.5,
            'status': 'ready'
        }
        
        logger.info("✓ Model loaded successfully")
        return True
        
    except FileNotFoundError as e:
        logger.error(f"Model files not found: {e}")
        logger.error("Please run: python train_pipeline.py")
        return False
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False


@app.before_request
def before_request():
    """Check model is loaded."""
    if fraud_detector is None and request.endpoint not in ['health', 'info']:
        return jsonify({'error': 'Model not loaded'}), 503


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy' if fraud_detector else 'model_not_loaded',
        'model_loaded': fraud_detector is not None
    }), 200


@app.route('/info', methods=['GET'])
def info():
    """Get model information."""
    if fraud_detector is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    return jsonify(model_info), 200


@app.route('/predict', methods=['POST'])
def predict_single():
    """
    Score a single transaction.
    
    Request body:
    {
        "transaction": {
            "Time": <float>,
            "Amount": <float>,
            "V1": <float>, ... "V28": <float>
        }
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'transaction' not in data:
            return jsonify({'error': 'Missing transaction data'}), 400
        
        transaction = data['transaction']
        
        # Validate
        if 'Amount' not in transaction:
            return jsonify({'error': 'Missing Amount field'}), 400
        
        # Score
        result = fraud_detector.score_transaction(transaction)
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Score multiple transactions.
    
    Request body:
    {
        "transactions": [
            {"Amount": <float>, "V1": <float>, ...},
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'transactions' not in data:
            return jsonify({'error': 'Missing transactions data'}), 400
        
        transactions = data['transactions']
        
        if not isinstance(transactions, list):
            return jsonify({'error': 'Transactions must be a list'}), 400
        
        if len(transactions) == 0:
            return jsonify({'error': 'Empty transactions list'}), 400
        
        # Score all
        results = fraud_detector.score_batch(transactions)
        
        # Summary
        fraud_count = sum(1 for r in results if r['is_fraud'])
        
        return jsonify({
            'count': len(results),
            'frauds_detected': fraud_count,
            'fraud_rate': fraud_count / len(results),
            'results': results
        }), 200
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/set_threshold', methods=['POST'])
def set_threshold():
    """
    Update decision threshold.
    
    Request body:
    {
        "threshold": <float between 0 and 1>
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'threshold' not in data:
            return jsonify({'error': 'Missing threshold value'}), 400
        
        threshold = data['threshold']
        
        if not isinstance(threshold, (int, float)):
            return jsonify({'error': 'Threshold must be a number'}), 400
        
        if not 0 <= threshold <= 1:
            return jsonify({'error': 'Threshold must be between 0 and 1'}), 400
        
        fraud_detector.set_threshold(threshold)
        model_info['decision_threshold'] = threshold
        
        return jsonify({
            'threshold': threshold,
            'message': 'Threshold updated successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error setting threshold: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/model_info', methods=['GET'])
def get_model_info():
    """Get detailed model information."""
    if fraud_detector is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    return jsonify(model_info), 200


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    logger.info("\n" + "="*80)
    logger.info("STARTING FRAUD DETECTION API")
    logger.info("="*80)
    
    # Load model
    if load_model_and_preprocessor():
        logger.info("✓ API ready to serve predictions")
        logger.info("\nStarting Flask server...")
        logger.info("API available at: http://localhost:5000")
        logger.info("\nEndpoints:")
        logger.info("  POST   /predict         - Score single transaction")
        logger.info("  POST   /predict_batch   - Score multiple transactions")
        logger.info("  POST   /set_threshold   - Update decision threshold")
        logger.info("  GET    /model_info      - Get model information")
        logger.info("  GET    /health          - Health check")
        logger.info("\n" + "="*80 + "\n")
        
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        logger.error("Failed to load model. Cannot start API.")
        sys.exit(1)
