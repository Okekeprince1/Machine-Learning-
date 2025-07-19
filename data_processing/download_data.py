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
