import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib
import logging
from typing import Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class FraudKNNModel:
    """K-Nearest Neighbors model for fraud detection with enhanced features."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = {
            'n_neighbors': 5,
            'weights': 'uniform',
            'algorithm': 'auto',
            'metric': 'euclidean',
            'n_jobs': -1
        }
        self.model = None
        self.training_time = None
        self.inference_times = []
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Trains the KNN model with error handling."""
        logger.info("Training KNN model...")
        
        if X_train.empty or y_train.empty:
            raise ValueError("Training data is empty")
        
        start_time = time.time()
        try:
            self.model = KNeighborsClassifier(
                n_neighbors=self.config['n_neighbors'],
                weights=self.config['weights'],
                algorithm=self.config['algorithm'],
                metric=self.config['metric'],
                n_jobs=self.config['n_jobs']
            )
            self.model.fit(X_train, y_train)
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise
        
        self.training_time = time.time() - start_time
        logger.info(f"Training completed in {self.training_time:.2f} seconds")
        return {'training_time': self.training_time}
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Makes predictions on new data with inference time logging."""
        if self.model is None:
            raise ValueError("Model is not trained yet")
        
        start_time = time.time()
        predictions = self.model.predict(X)
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns prediction probabilities."""
        if self.model is None:
            raise ValueError("Model is not trained yet")
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluates model performance with comprehensive metrics."""
        logger.info("Evaluating KNN model...")
        
        if self.model is None:
            raise ValueError("Model is not trained yet")
        
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        accuracy = self.model.score(X_test, y_test)
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        avg_inference_time = np.mean(self.inference_times) if self.inference_times else 0
        
        results = {
            'accuracy': accuracy,
            'auc_roc': auc_roc,
            'precision': class_report['1']['precision'],
            'recall': class_report['1']['recall'],
            'f1_score': class_report['1']['f1-score'],
            'confusion_matrix': cm,
            'avg_inference_time_ms': avg_inference_time * 1000,
            'training_time_s': self.training_time
        }
        
        logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}, AUC-ROC: {auc_roc:.4f}")
        return results
    
    def plot_results(self, X_test: pd.DataFrame, y_test: pd.Series, save_path: str = None):
        """Plots model results, including prediction distribution."""
        if self.model is None:
            raise ValueError("Model is not trained yet")
        
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        axes[1].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
        axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend()
        
        # Prediction Distribution
        axes[2].hist(y_pred_proba[y_test == 0], bins=50, alpha=0.7, label='Non-Fraud', density=True)
        axes[2].hist(y_pred_proba[y_test == 1], bins=50, alpha=0.7, label='Fraud', density=True)
        axes[2].set_xlabel('Prediction Probability')
        axes[2].set_ylabel('Density')
        axes[2].set_title('Prediction Distribution')
        axes[2].legend()
        
        # Empty subplot
        axes[3].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Results plot saved to: {save_path}")
        plt.show()
    
    def save_model(self, file_path: str):
        """Saves the trained model with all relevant data."""
        if self.model is None:
            raise ValueError("Model is not trained yet")
        
        model_data = {
            'model': self.model,
            'config': self.config,
            'training_time': self.training_time
        }
        joblib.dump(model_data, file_path)
        logger.info(f"Model saved to: {file_path}")
    
    def load_model(self, file_path: str):
        """Loads a previously saved model."""
        model_data = joblib.load(file_path)
        self.model = model_data['model']
        self.config = model_data['config']
        self.training_time = model_data['training_time']
        logger.info(f"Model loaded from: {file_path}")
