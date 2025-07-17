import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, KFold
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib
import logging
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class FraudNeuralNetwork:
    """Neural Network model for fraud detection with enhanced features."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            'input_dim': None,
            'hidden_layers': [128, 64, 32],
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'early_stopping_patience': 10,
            'class_weight': {0: 1, 1: 10},  # Handle class imbalance
            'random_state': 42
        }
        self.model = None
        self.history = None
        self.training_time = None
        self.inference_times = []
        self.feature_importance = None
        self._validate_config()
        
        # Set random seeds
        np.random.seed(self.config['random_state'])
        tf.random.set_seed(self.config['random_state'])
    
    def _validate_config(self):
        """Validates configuration parameters."""
        if 'input_dim' in self.config and self.config['input_dim'] is not None and self.config['input_dim'] <= 0:
            raise ValueError("input_dim must be a positive integer or None")
        if not self.config['hidden_layers'] or any(units <= 0 for units in self.config['hidden_layers']):
            raise ValueError("hidden_layers must be a non-empty list of positive integers")
        if self.config['dropout_rate'] < 0 or self.config['dropout_rate'] > 1:
            raise ValueError("dropout_rate must be between 0 and 1")
        if self.config['learning_rate'] <= 0:
            raise ValueError("learning_rate must be positive")
        if self.config['batch_size'] <= 0:
            raise ValueError("batch_size must be positive")
        if self.config['epochs'] <= 0:
            raise ValueError("epochs must be positive")
        if self.config['early_stopping_patience'] <= 0:
            raise ValueError("early_stopping_patience must be positive")
    
    def build_model(self, input_dim: int) -> keras.Model:
        """Builds the neural network architecture."""
        self.config['input_dim'] = input_dim
        logger.info(f"Building neural network with input dimension {input_dim}")
        
        model = keras.Sequential([
            layers.Dense(self.config['hidden_layers'][0], activation='relu', input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(self.config['dropout_rate']),
            *[layer for i in range(1, len(self.config['hidden_layers'])) for layer in (
                layers.Dense(self.config['hidden_layers'][i], activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(self.config['dropout_rate'])
            )],
            layers.Dense(1, activation='sigmoid')
        ])
        
        optimizer = keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall')]
        )
        
        return model
    
    def create_callbacks(self) -> list:
        """Creates training callbacks."""
        Path('models').mkdir(exist_ok=True)
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                'models/best_neural_network.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        return callbacks_list
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict[str, Any]:
        """Trains the neural network model with error handling."""
        logger.info("Training Neural Network model...")
        
        if X_train.empty or y_train.empty:
            raise ValueError("Training data is empty")
        
        start_time = time.time()
        self.model = self.build_model(X_train.shape[1])
        
        if X_val is None or y_val is None:
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train, y_train,
                test_size=0.2,
                random_state=self.config['random_state'],
                stratify=y_train
            )
        else:
            X_train_split, y_train_split = X_train, y_train
        
        callbacks_list = self.create_callbacks()
        
        try:
            self.history = self.model.fit(
                X_train_split, y_train_split,
                validation_data=(X_val, y_val),
                batch_size=self.config['batch_size'],
                epochs=self.config['epochs'],
                callbacks=callbacks_list,
                class_weight=self.config['class_weight'],
                verbose=1
            )
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise
        
        self.training_time = time.time() - start_time
        logger.info(f"Training completed in {self.training_time:.2f} seconds")
        
        return {
            'training_time': self.training_time,
            'history': self.history.history,
            'best_epoch': np.argmin(self.history.history['val_loss']) + 1
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Makes predictions on new data with inference time logging."""
        if self.model is None:
            raise ValueError("Model is not trained yet")
        
        start_time = time.time()
        predictions = (self.model.predict(X, batch_size=self.config['batch_size']) > 0.5).astype(int).flatten()
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns prediction probabilities."""
        if self.model is None:
            raise ValueError("Model is not trained yet")
        return self.model.predict(X, batch_size=self.config['batch_size']).flatten()
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluates model performance with comprehensive metrics."""
        logger.info("Evaluating Neural Network model...")
        
        if self.model is None:
            raise ValueError("Model is not trained yet")
        
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        
        test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(X_test, y_test, verbose=0)
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        avg_inference_time = np.mean(self.inference_times) if self.inference_times else 0
        
        results = {
            'test_loss': test_loss,
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'auc_roc': auc_roc,
            'f1_score': class_report['1']['f1-score'],
            'confusion_matrix': cm,
            'avg_inference_time_ms': avg_inference_time * 1000,
            'training_time_s': self.training_time
        }
        
        logger.info(f"Evaluation completed. Accuracy: {test_accuracy:.4f}, AUC-ROC: {auc_roc:.4f}")
        return results
    
    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series, 
                             X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Performs hyperparameter tuning for learning rate and hidden layers."""
        logger.info("Performing hyperparameter tuning...")
        
        param_grid = {
            'learning_rate': [0.001, 0.0001, 0.00001],
            'hidden_layers': [[128, 64, 32], [64, 32], [256, 128, 64, 32]],
            'dropout_rate': [0.2, 0.3, 0.4]
        }
        
        best_score = float('inf')
        best_params = {}
        results = []
        
        for lr in param_grid['learning_rate']:
            for hl in param_grid['hidden_layers']:
                for dr in param_grid['dropout_rate']:
                    self.config['learning_rate'] = lr
                    self.config['hidden_layers'] = hl
                    self.config['dropout_rate'] = dr
                    
                    try:
                        self.model = self.build_model(X_train.shape[1])
                        history = self.model.fit(
                            X_train, y_train,
                            validation_data=(X_val, y_val),
                            batch_size=self.config['batch_size'],
                            epochs=self.config['epochs'],
                            callbacks=self.create_callbacks(),
                            class_weight=self.config['class_weight'],
                            verbose=0
                        )
                        val_loss = min(history.history['val_loss'])
                        results.append({
                            'learning_rate': lr,
                            'hidden_layers': hl,
                            'dropout_rate': dr,
                            'val_loss': val_loss
                        })
                        if val_loss < best_score:
                            best_score = val_loss
                            best_params = {'learning_rate': lr, 'hidden_layers': hl, 'dropout_rate': dr}
                    except Exception as e:
                        logger.warning(f"Tuning failed for lr={lr}, hl={hl}, dr={dr}: {e}")
        
        self.config.update(best_params)
        self.model = self.build_model(X_train.shape[1])
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best validation loss: {best_score:.4f}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'results': results
        }
    
    def cross_validation(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, Any]:
        """Performs k-fold cross-validation with F1 scoring."""
        logger.info(f"Performing {cv}-fold cross-validation...")
        
        kf = KFold(n_splits=cv, shuffle=True, random_state=self.config['random_state'])
        f1_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            try:
                self.model = self.build_model(X.shape[1])
                history = self.model.fit(
                    X_train_fold, y_train_fold,
                    validation_data=(X_val_fold, y_val_fold),
                    batch_size=self.config['batch_size'],
                    epochs=self.config['epochs'],
                    callbacks=self.create_callbacks(),
                    class_weight=self.config['class_weight'],
                    verbose=0
                )
                y_pred = (self.model.predict(X_val_fold) > 0.5).astype(int).flatten()
                class_report = classification_report(y_val_fold, y_pred, output_dict=True)
                f1_scores.append(class_report['1']['f1-score'])
            except Exception as e:
                logger.warning(f"Cross-validation fold {fold + 1} failed: {e}")
        
        results = {
            'cv_scores': np.array(f1_scores),
            'cv_mean': np.mean(f1_scores) if f1_scores else 0,
            'cv_std': np.std(f1_scores) if f1_scores else 0,
            'cv_min': np.min(f1_scores) if f1_scores else 0,
            'cv_max': np.max(f1_scores) if f1_scores else 0
        }
        
        logger.info(f"CV Results - Mean: {results['cv_mean']:.4f}, Std: {results['cv_std']:.4f}")
        return results
    
    def compute_feature_importance(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """Computes feature importance using permutation importance."""
        logger.info("Computing feature importance...")
        try:
            r = permutation_importance(
                estimator=lambda X: self.model.predict(X).flatten(),
                X=X_test,
                y=y_test,
                scoring='f1',
                n_repeats=10,
                random_state=self.config['random_state']
            )
            self.feature_importance = pd.DataFrame({
                'feature': X_test.columns,
                'importance': r.importances_mean
            }).sort_values('importance', ascending=False)
        except Exception as e:
            logger.error(f"Feature importance computation failed: {e}")
            raise
        return self.feature_importance
    
    def plot_training_history(self, save_path: str = None):
        """Plots training history."""
        if self.history is None:
            logger.warning("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        metrics = ['loss', 'accuracy', 'precision', 'recall']
        titles = ['Model Loss', 'Model Accuracy', 'Model Precision', 'Model Recall']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            axes[i].plot(self.history.history[metric], label=f'Training {metric.capitalize()}')
            axes[i].plot(self.history.history[f'val_{metric}'], label=f'Validation {metric.capitalize()}')
            axes[i].set_title(title)
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to: {save_path}")
        plt.show()
    
    def plot_results(self, X_test: pd.DataFrame, y_test: pd.Series, save_path: str = None):
        """Plots model results, including feature importance."""
        if self.model is None:
            raise ValueError("Model is not trained yet")
        
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        
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
        
        # Feature Importance
        if self.feature_importance is None:
            self.compute_feature_importance(X_test, y_test)
        top_features = self.feature_importance.head(10)
        axes[3].barh(range(len(top_features)), top_features['importance'])
        axes[3].set_yticks(range(len(top_features)))
        axes[3].set_yticklabels(top_features['feature'])
        axes[3].set_xlabel('Feature Importance')
        axes[3].set_title('Top 10 Feature Importance')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Results plot saved to: {save_path}")
        plt.show()
    
    def save_model(self, file_path: str):
        """Saves the trained model with metadata."""
        if self.model is None:
            raise ValueError("Model is not trained yet")
        
        self.model.save(file_path)
        metadata = {
            'config': self.config,
            'training_time': self.training_time,
            'history': self.history.history if self.history else None,
            'feature_importance': self.feature_importance
        }
        joblib.dump(metadata, file_path.replace('.h5', '_metadata.pkl'))
        logger.info(f"Model saved to: {file_path}")
    
    def load_model(self, file_path: str):
        """Loads a previously saved model."""
        try:
            self.model = keras.models.load_model(file_path)
            metadata = joblib.load(file_path.replace('.h5', '_metadata.pkl'))
            self.config = metadata['config']
            self.training_time = metadata['training_time']
            self.history = metadata['history']
            self.feature_importance = metadata.get('feature_importance')
            logger.info(f"Model loaded from: {file_path}")
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise

def main():
    """Main function to demonstrate the Neural Network model."""
    print("Neural Network Model for Fraud Detection")
    print("Team Member 4: Deep Learning Approach")
    print("\nLiterature Review Summary:")
    print("- Neural networks achieve 92-95% accuracy on credit card fraud datasets (PyFi, 2023)")
    print("- Effective for capturing complex non-linear patterns in fraud data")
    print("- Class weighting and regularization prevent overfitting in imbalanced datasets")
    print("- Batch normalization and dropout improve training stability")
    print("- Suitable for cloud deployment with TensorFlow Serving or SageMaker")
    print("Note: Use in pipeline after preprocessing with FraudDataPreprocessor.")

if __name__ == "__main__":
    main()