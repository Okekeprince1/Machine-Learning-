"""
Main Pipeline for Fraud Detection Project
Orchestrates the complete workflow using existing train/test datasets.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import time
import pandas as pd
from typing import Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Import project modules
from data_processing.download_data import DataLoader
from data_processing.preprocessor import FraudDataPreprocessor
from models.logistic_regression_model import FraudLogisticRegression
from models.neural_network_model import FraudNeuralNetwork
from models.knn_model import FraudKNNModel
from models.decision_tree_model import FraudDecisionTreeModel
from evaluation.model_comparison import ModelComparator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fraud_detection_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FraudDetectionPipeline:
    """Main pipeline for fraud detection project."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            'data_dir': 'data',
            'models_dir': 'models',
            'reports_dir': 'reports',
            'load_data': True,
            'preprocess_data': True,
            'train_models': True,
            'evaluate_models': True,
            'deploy_models': False,
            'cloud_provider': 'aws'
        }
        
        # Create directories
        for dir_name in [self.config['data_dir'], self.config['models_dir'], self.config['reports_dir']]:
            Path(dir_name).mkdir(exist_ok=True)
        
        # Initialize components
        self.data_loader = None
        self.preprocessor = None
        self.models = {}
        self.comparator = ModelComparator()
        
    def load_datasets(self) -> bool:
        """Loads the existing fraud detection datasets."""
        logger.info("Step 1: Loading existing datasets...")
        
        try:
            self.data_loader = DataLoader(self.config['data_dir'])
            
            # Load datasets
            train_data, test_data = self.data_loader.load_datasets()
            
            # Validate data
            metadata = self.data_loader.validate_data(train_data, test_data)
            if metadata["valid"]:
                # Save metadata
                self.data_loader.save_metadata(metadata)
                
                # Prepare data for training
                X_train, X_test, y_train, y_test = self.data_loader.prepare_data_for_training(train_data, test_data)
                
                # Save prepared data
                X_train.to_csv(f"{self.config['data_dir']}/X_train.csv", index=False)
                X_test.to_csv(f"{self.config['data_dir']}/X_test.csv", index=False)
                y_train.to_csv(f"{self.config['data_dir']}/y_train.csv", index=False)
                y_test.to_csv(f"{self.config['data_dir']}/y_test.csv", index=False)
                
                logger.info("✅ Datasets loaded and validated successfully!")
                return True
            else:
                logger.error(f"❌ Dataset validation failed: {metadata.get('error')}")
                return False
                
        except Exception as e:
            logger.error(f"Error in dataset loading: {e}")
            return False
    
    def preprocess_data(self) -> bool:
        """Preprocesses the dataset."""
        logger.info("Step 2: Preprocessing data...")
        
        try:
            # Initialize preprocessor
            self.preprocessor = FraudDataPreprocessor()
            
            # Run preprocessing pipeline
            X_train, X_test, y_train, y_test = self.preprocessor.preprocess_pipeline(
                train_file="data/fraudTrain.csv",
                test_file="data/fraudTest.csv",
                save_preprocessor=True
            )
            
            # Save processed data
            X_train.to_csv(f"{self.config['data_dir']}/X_train_processed.csv", index=False)
            X_test.to_csv(f"{self.config['data_dir']}/X_test_processed.csv", index=False)
            y_train.to_csv(f"{self.config['data_dir']}/y_train_processed.csv", index=False)
            y_test.to_csv(f"{self.config['data_dir']}/y_test_processed.csv", index=False)
            
            logger.info("✅ Data preprocessing completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {e}")
            return False
    
    def train_models(self) -> bool:
        """Trains all fraud detection models."""
        logger.info("Step 3: Training models...")
        
        try:
            # Load processed data
            X_train = pd.read_csv(f"{self.config['data_dir']}/X_train_processed.csv")
            X_test = pd.read_csv(f"{self.config['data_dir']}/X_test_processed.csv")
            y_train = pd.read_csv(f"{self.config['data_dir']}/y_train_processed.csv").iloc[:, 0]
            y_test = pd.read_csv(f"{self.config['data_dir']}/y_test_processed.csv").iloc[:, 0]
            
            # Train Logistic Regression (Team Member 1)
            logger.info("Training Logistic Regression model...")
            lr_model = FraudLogisticRegression()
            lr_model.train(X_train, y_train)
            lr_results = lr_model.evaluate(X_test, y_test)
            lr_model.save_model(f"{self.config['models_dir']}/logistic_regression_model.pkl")
            self.models['logistic_regression'] = lr_model
            
            # Train Neural Network (Team Member 2)
            logger.info("Training Neural Network model...")
            nn_model = FraudNeuralNetwork()
            nn_model.train(X_train, y_train)
            nn_results = nn_model.evaluate(X_test, y_test)
            nn_model.save_model(f"{self.config['models_dir']}/neural_network_model.h5")
            self.models['neural_network'] = nn_model
            
            # Train KNN Model (New)
            logger.info("Training KNN model...")
            knn_model = FraudKNNModel()
            knn_model.train(X_train, y_train)
            knn_results = knn_model.evaluate(X_test, y_test)
            knn_model.save_model(f"{self.config['models_dir']}/knn_model.pkl")
            self.models['knn'] = knn_model
            
            # Train Decision Tree Model (New)
            logger.info("Training Decision Tree model...")
            dt_model = FraudDecisionTreeModel()
            dt_model.train(X_train, y_train)
            dt_results = dt_model.evaluate(X_test, y_test)
            dt_model.save_model(f"{self.config['models_dir']}/decision_tree_model.pkl")
            self.models['decision_tree'] = dt_model
            
            logger.info("✅ All models trained successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            return False
    
    def evaluate_models(self) -> bool:
        """Evaluates and compares all models."""
        logger.info("Step 4: Evaluating models...")
        
        try:
            # Load test data
            X_test = pd.read_csv(f"{self.config['data_dir']}/X_test_processed.csv")
            y_test = pd.read_csv(f"{self.config['data_dir']}/y_test_processed.csv").iloc[:, 0]
            
            # Add models to comparator
            for name, model in self.models.items():
                self.comparator.add_model(name, model.model, name)
            
            # Evaluate all models
            results = self.comparator.evaluate_all_models(X_test, y_test)
            
            # Create comparison dataframe
            comparison_df = self.comparator.create_comparison_dataframe()
            
            # Save results
            comparison_df.to_csv(f"{self.config['reports_dir']}/model_comparison.csv", index=False)
            self.comparator.save_results(f"{self.config['reports_dir']}/evaluation_results.json")
            
            # Generate plots
            self.comparator.plot_comparison(f"{self.config['reports_dir']}/model_comparison.png")
            self.comparator.plot_roc_curves(X_test, y_test, f"{self.config['reports_dir']}/roc_curves.png")
            
            # Generate HTML report
            self.comparator.generate_report(f"{self.config['reports_dir']}/model_comparison_report.html")
            
            # Print summary
            print("\n" + "="*50)
            print("MODEL EVALUATION SUMMARY")
            print("="*50)
            print(comparison_df.to_string(index=False))
            print("\n" + "="*50)
            
            logger.info("✅ Model evaluation completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error in model evaluation: {e}")
            return False
    
    def deploy_models(self) -> bool:
        """Deploys models to cloud environment."""
        if not self.config['deploy_models']:
            logger.info("Model deployment skipped (disabled in config)")
            return True
        
        logger.info("Step 5: Deploying models to cloud...")
        
        try:
            from cloud.deployment import CloudDeployer
            
            deployer = CloudDeployer()
            success = deployer.deploy_complete_pipeline(self.config['cloud_provider'])
            
            if success:
                logger.info("✅ Model deployment completed successfully!")
            else:
                logger.error("❌ Model deployment failed")
            
            return success
            
        except Exception as e:
            logger.error(f"Error in model deployment: {e}")
            return False
    
    def run_pipeline(self) -> bool:
        """Runs the complete fraud detection pipeline."""
        logger.info("Starting Fraud Detection Pipeline")
        logger.info("="*50)
        
        start_time = time.time()
        
        # Step 1: Load datasets
        if self.config['load_data']:
            if not self.load_datasets():
                logger.error("Pipeline failed at dataset loading step")
                return False
        
        # Step 2: Preprocess data
        if self.config['preprocess_data']:
            if not self.preprocess_data():
                logger.error("Pipeline failed at data preprocessing step")
                return False
        
        # Step 3: Train models
        if self.config['train_models']:
            if not self.train_models():
                logger.error("Pipeline failed at model training step")
                return False
        
        # Step 4: Evaluate models
        if self.config['evaluate_models']:
            if not self.evaluate_models():
                logger.error("Pipeline failed at model evaluation step")
                return False
        
        # Step 5: Deploy models
        if self.config['deploy_models']:
            if not self.deploy_models():
                logger.error("Pipeline failed at model deployment step")
                return False
        
        total_time = time.time() - start_time
        
        logger.info("="*50)
        logger.info(f"Pipeline completed successfully in {total_time:.2f} seconds!")
        logger.info("="*50)
        
        return True

def main():
    """Main function to run the fraud detection pipeline."""
    parser = argparse.ArgumentParser(description="Fraud Detection Pipeline")
    parser.add_argument("--skip-load", action="store_true", help="Skip dataset loading")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip data preprocessing")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training")
    parser.add_argument("--skip-evaluate", action="store_true", help="Skip model evaluation")
    parser.add_argument("--deploy", action="store_true", help="Deploy models to cloud")
    parser.add_argument("--cloud-provider", choices=["aws", "gcp"], default="aws", help="Cloud provider for deployment")
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        'data_dir': 'data',
        'models_dir': 'models',
        'reports_dir': 'reports',
        'load_data': not args.skip_load,
        'preprocess_data': not args.skip_preprocess,
        'train_models': not args.skip_train,
        'evaluate_models': not args.skip_evaluate,
        'deploy_models': args.deploy,
        'cloud_provider': args.cloud_provider
    }
    
    # Run pipeline
    pipeline = FraudDetectionPipeline(config)
    success = pipeline.run_pipeline()
    
    if success:
        print("\n🎉 Fraud Detection Pipeline completed successfully!")
        print("📊 Check the 'reports' directory for detailed results")
        print("🤖 Models are saved in the 'models' directory")
        if args.deploy:
            print("☁️  Models have been deployed to the cloud")
    else:
        print("\n❌ Fraud Detection Pipeline failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 