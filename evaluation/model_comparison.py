"""
Model Comparison and Evaluation Module
Compares all team members' models using standard metrics and provides comprehensive analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import time
import json
import logging
from typing import Dict, Any, List, Tuple
from pathlib import Path
import joblib

logger = logging.getLogger(__name__)

class ModelComparator:
    """Compares multiple fraud detection models."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.comparison_data = None
        
    def add_model(self, name: str, model, model_type: str = "unknown"):
        """Adds a model to the comparison."""
        self.models[name] = {
            'model': model,
            'type': model_type
        }
        logger.info(f"Added model: {name} ({model_type})")
    
    def evaluate_model(self, name: str, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluates a single model."""
        if name not in self.models:
            raise ValueError(f"Model {name} not found")
        
        logger.info(f"Evaluating model: {name}")
        
        model_info = self.models[name]
        model = model_info['model']
        
        # Measure inference time
        start_time = time.time()
        
        # Make predictions
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred = model.predict(X_test)
            y_pred_proba = None
        
        inference_time = time.time() - start_time
        
        # Calculate metrics
        accuracy = np.mean(y_pred == y_test)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # AUC-ROC (if probabilities available)
        auc_roc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
        
        results = {
            'model_name': name,
            'model_type': model_info['type'],
            'accuracy': accuracy,
            'precision': class_report['1']['precision'],
            'recall': class_report['1']['recall'],
            'f1_score': class_report['1']['f1-score'],
            'auc_roc': auc_roc,
            'inference_time_ms': inference_time * 1000,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        self.results[name] = results
        logger.info(f"Model {name} evaluation completed")
        
        return results
    
    def evaluate_all_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluates all models."""
        logger.info("Evaluating all models...")
        
        all_results = {}
        
        for name in self.models.keys():
            try:
                results = self.evaluate_model(name, X_test, y_test)
                all_results[name] = results
            except Exception as e:
                logger.error(f"Error evaluating {name}: {e}")
                all_results[name] = {'error': str(e)}
        
        return all_results
    
    def create_comparison_dataframe(self) -> pd.DataFrame:
        """Creates a comparison dataframe of all models."""
        if not self.results:
            logger.warning("No results available for comparison")
            return pd.DataFrame()
        
        comparison_data = []
        
        for name, results in self.results.items():
            if 'error' not in results:
                comparison_data.append({
                    'Model': name,
                    'Type': results['model_type'],
                    'Accuracy': results['accuracy'],
                    'Precision': results['precision'],
                    'Recall': results['recall'],
                    'F1-Score': results['f1_score'],
                    'AUC-ROC': results['auc_roc'],
                    'Inference Time (ms)': results['inference_time_ms']
                })
        
        self.comparison_data = pd.DataFrame(comparison_data)
        return self.comparison_data
    
    def plot_comparison(self, save_path: str = None):
        """Plots comprehensive model comparison."""
        if self.comparison_data is None:
            self.create_comparison_dataframe()
        
        if self.comparison_data.empty:
            logger.warning("No comparison data available")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Accuracy Comparison
        axes[0, 0].bar(self.comparison_data['Model'], self.comparison_data['Accuracy'])
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. F1-Score Comparison
        axes[0, 1].bar(self.comparison_data['Model'], self.comparison_data['F1-Score'])
        axes[0, 1].set_title('Model F1-Score Comparison')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. AUC-ROC Comparison
        axes[0, 2].bar(self.comparison_data['Model'], self.comparison_data['AUC-ROC'])
        axes[0, 2].set_title('Model AUC-ROC Comparison')
        axes[0, 2].set_ylabel('AUC-ROC')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Inference Time Comparison
        axes[1, 0].bar(self.comparison_data['Model'], self.comparison_data['Inference Time (ms)'])
        axes[1, 0].set_title('Model Inference Time Comparison')
        axes[1, 0].set_ylabel('Inference Time (ms)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. Precision vs Recall
        axes[1, 1].scatter(self.comparison_data['Precision'], self.comparison_data['Recall'], 
                          s=100, alpha=0.7)
        for i, model in enumerate(self.comparison_data['Model']):
            axes[1, 1].annotate(model, (self.comparison_data['Precision'].iloc[i], 
                                       self.comparison_data['Recall'].iloc[i]))
        axes[1, 1].set_xlabel('Precision')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].set_title('Precision vs Recall')
        
        # 6. Performance Heatmap
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
        heatmap_data = self.comparison_data[metrics].T
        heatmap_data.columns = self.comparison_data['Model']
        
        sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', ax=axes[1, 2])
        axes[1, 2].set_title('Performance Metrics Heatmap')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to: {save_path}")
        
        plt.show()
    
    def plot_roc_curves(self, X_test: pd.DataFrame, y_test: pd.Series, save_path: str = None):
        """Plots ROC curves for all models."""
        plt.figure(figsize=(10, 8))
        
        for name, results in self.results.items():
            if 'error' not in results and results['probabilities'] is not None:
                fpr, tpr, _ = roc_curve(y_test, results['probabilities'])
                auc_score = results['auc_roc']
                plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curves plot saved to: {save_path}")
        
        plt.show()
    
    def generate_report(self, output_path: str = "reports/model_comparison_report.html"):
        """Generates a comprehensive HTML report."""
        if self.comparison_data is None:
            self.create_comparison_dataframe()
        
        if self.comparison_data.empty:
            logger.warning("No comparison data available for report")
            return
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Fraud Detection Model Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ font-weight: bold; color: #333; }}
                .best {{ background-color: #d4edda; }}
            </style>
        </head>
        <body>
            <h1>Fraud Detection Model Comparison Report</h1>
            <p><strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Model Performance Summary</h2>
            {self.comparison_data.to_html(index=False, classes='dataframe')}
            
            <h2>Best Performing Models</h2>
            <ul>
                <li><strong>Best Accuracy:</strong> {self.comparison_data.loc[self.comparison_data['Accuracy'].idxmax(), 'Model']} ({self.comparison_data['Accuracy'].max():.4f})</li>
                <li><strong>Best F1-Score:</strong> {self.comparison_data.loc[self.comparison_data['F1-Score'].idxmax(), 'Model']} ({self.comparison_data['F1-Score'].max():.4f})</li>
                <li><strong>Best AUC-ROC:</strong> {self.comparison_data.loc[self.comparison_data['AUC-ROC'].idxmax(), 'Model']} ({self.comparison_data['AUC-ROC'].max():.4f})</li>
                <li><strong>Fastest Inference:</strong> {self.comparison_data.loc[self.comparison_data['Inference Time (ms)'].idxmin(), 'Model']} ({self.comparison_data['Inference Time (ms)'].min():.2f} ms)</li>
            </ul>
            
            <h2>Detailed Results</h2>
        """
        
        # Add detailed results for each model
        for name, results in self.results.items():
            if 'error' not in results:
                html_content += f"""
                <h3>{name} ({results['model_type']})</h3>
                <table>
                    <tr><td>Accuracy</td><td>{results['accuracy']:.4f}</td></tr>
                    <tr><td>Precision</td><td>{results['precision']:.4f}</td></tr>
                    <tr><td>Recall</td><td>{results['recall']:.4f}</td></tr>
                    <tr><td>F1-Score</td><td>{results['f1_score']:.4f}</td></tr>
                    <tr><td>AUC-ROC</td><td>{results['auc_roc']:.4f}</td></tr>
                    <tr><td>Inference Time</td><td>{results['inference_time_ms']:.2f} ms</td></tr>
                </table>
                """
        
        html_content += """
        </body>
        </html>
        """
        
        # Save report
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Comparison report saved to: {output_path}")
    
    def save_results(self, file_path: str):
        """Saves comparison results to file."""
        results_data = {
            'comparison_data': self.comparison_data.to_dict() if self.comparison_data is not None else None,
            'results': {name: {k: v for k, v in res.items() if k not in ['predictions', 'probabilities']} 
                       for name, res in self.results.items()}
        }
        
        with open(file_path, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {file_path}")
    
    def load_results(self, file_path: str):
        """Loads comparison results from file."""
        with open(file_path, 'r') as f:
            results_data = json.load(f)
        
        if results_data['comparison_data']:
            self.comparison_data = pd.DataFrame(results_data['comparison_data'])
        self.results = results_data['results']
        
        logger.info(f"Results loaded from: {file_path}")

def main():
    """Main function to demonstrate model comparison."""
    print("Model Comparison and Evaluation Module")
    print("Compares all team members' models using standard metrics")
    print("\nFeatures:")
    print("- Comprehensive model evaluation")
    print("- Performance comparison across multiple metrics")
    print("- Visualization of results")
    print("- HTML report generation")
    print("- Latency and throughput analysis")

if __name__ == "__main__":
    main() 