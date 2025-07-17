"""
Data Loading Module for Fraud Detection Project
Loads existing train and test datasets from local files and performs initial validation.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Tuple, Dict, Any, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Loads and validates fraud detection datasets from local files.
    """
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.train_file = str(self.data_dir) + "/" + "fraudTrain.csv"
        self.test_file = str(self.data_dir) + "/" + "fraudTest.csv"

    def load_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the train and test datasets from local files.
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (train_data, test_data)
        """
        try:
            logger.info("Loading fraud detection datasets from local files...")
            train_path = Path(self.train_file)
            test_path = Path(self.test_file)
            if not train_path.exists():
                raise FileNotFoundError(f"Training file not found: {self.train_file}")
            if not test_path.exists():
                raise FileNotFoundError(f"Test file not found: {self.test_file}")
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            logger.info(f"Training data shape: {train_data.shape}")
            logger.info(f"Test data shape: {test_data.shape}")
            return train_data, test_data
        except Exception as e:
            logger.error(f"Error loading datasets: {e}")
            raise

    def convert_numpy_types(self, obj: Any) -> Any:
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        else:
            return obj

    def validate_data(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validates the loaded datasets and returns basic statistics.
        Returns:
            dict: Dataset statistics and validation results
        """
        try:
            # Basic validation
            validation_results: Dict[str, Any] = {
                "valid": True,
                "train_shape": train_data.shape,
                "test_shape": test_data.shape,
                "train_columns": list(train_data.columns),
                "test_columns": list(test_data.columns),
                "train_missing_values": train_data.isnull().sum().to_dict(),
                "test_missing_values": test_data.isnull().sum().to_dict(),
                "train_data_types": {col: str(dtype) for col, dtype in train_data.dtypes.to_dict().items()},
                "test_data_types": {col: str(dtype) for col, dtype in test_data.dtypes.to_dict().items()},
                "train_target_distribution": train_data['Class'].value_counts().to_dict() if 'Class' in train_data.columns else None,
                "test_target_distribution": test_data['Class'].value_counts().to_dict() if 'Class' in test_data.columns else None,
                "train_memory_usage_mb": float(train_data.memory_usage(deep=True).sum() / 1024 / 1024),
                "test_memory_usage_mb": float(test_data.memory_usage(deep=True).sum() / 1024 / 1024)
            }

            # Convert numpy types to native Python types
            validation_results = self.convert_numpy_types(validation_results)

            logger.info(f"Training data validation completed. Shape: {train_data.shape}")
            logger.info(f"Test data validation completed. Shape: {test_data.shape}")

            train_dist = validation_results.get('train_target_distribution')
            test_dist = validation_results.get('test_target_distribution')

            if train_dist:
                logger.info(f"Training target distribution: {train_dist}")
            if test_dist:
                logger.info(f"Test target distribution: {test_dist}")

            return validation_results

        except Exception as e:
            logger.error(f"Error validating data: {e}")
            return {"valid": False, "error": str(e)}

    def save_metadata(self, metadata: Dict[str, Any]) -> None:
        """Saves dataset metadata for future reference."""
        import json

        metadata_file = self.data_dir / "dataset_metadata.json"
        self.data_dir.mkdir(exist_ok=True)

        # Ensure metadata is JSON serializable
        metadata = self.convert_numpy_types(metadata)

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Metadata saved to: {metadata_file}")

    def prepare_data_for_training(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepares data for training by separating features and targets.
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: (X_train, X_test, y_train, y_test)
        """
        try:
            # Separate features and target for training data
            if 'is_fraud' in train_data.columns:
                X_train = train_data.drop('is_fraud', axis=1)
                y_train = train_data['is_fraud']
            else:
                raise ValueError("'is_fraud' column not found in training data")

            # Separate features and target for test data
            if 'is_fraud' in test_data.columns:
                X_test = test_data.drop('is_fraud', axis=1)
                y_test = test_data['is_fraud']
            else:
                raise ValueError("'is_fraud' column not found in test data")

            logger.info(f"Prepared training features: {X_train.shape}")
            logger.info(f"Prepared test features: {X_test.shape}")
            logger.info(f"Training target shape: {y_train.shape}")
            logger.info(f"Test target shape: {y_test.shape}")

            return X_train, X_test, y_train, y_test

        except Exception as e:
            logger.error(f"Error preparing data for training: {e}")
            raise

def main() -> None:
    """Main function to load and validate the existing datasets."""
    loader = DataLoader()

    try:
        # Load datasets
        train_data, test_data = loader.load_datasets()

        # Validate data
        metadata = loader.validate_data(train_data, test_data)

        if metadata.get("valid", False):
            # Save metadata
            loader.save_metadata(metadata)

            # Prepare data for training
            X_train, X_test, y_train, y_test = loader.prepare_data_for_training(train_data, test_data)

            # Save prepared data
            X_train.to_csv('data/X_train.csv', index=False)
            X_test.to_csv('data/X_test.csv', index=False)
            y_train.to_csv('data/y_train.csv', index=False)
            y_test.to_csv('data/y_test.csv', index=False)

            print("✅ Datasets loaded and validated successfully!")
            print(f"📊 Training data shape: {train_data.shape}")
            print(f"📊 Test data shape: {test_data.shape}")

            train_dist = metadata.get('train_target_distribution')
            test_dist = metadata.get('test_target_distribution')

            if train_dist:
                print(f"🎯 Training target distribution: {train_dist}")
            if test_dist:
                print(f"🎯 Test target distribution: {test_dist}")
        else:
            error_msg = metadata.get('error', 'Unknown error')
            print(f"❌ Dataset validation failed: {error_msg}")

    except Exception as e:
        print(f"❌ Error loading datasets: {e}")

if __name__ == "__main__":
    main() 