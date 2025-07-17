import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
import logging
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import joblib

logger = logging.getLogger(__name__)

class FraudDataPreprocessor:
    """Handles preprocessing of fraud detection data for a specific dataset."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {
            'scaler_type': 'robust',  # 'standard' or 'robust'
            'handle_imbalance': True,
            'feature_engineering': True,
            'imbalance_method': 'undersample'  # 'undersample' or 'oversample'
        }
        self.scaler = None
        self.num_imputer = None
        self.cat_imputer = None
        self.encoder = None
        self.feature_columns = None
        self.target_column = 'is_fraud'
        
    def load_existing_data(self, train_file: str, test_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads existing train and test datasets."""
        try:
            logger.info(f"Loading training data from: {train_file}")
            train_data = pd.read_csv(train_file)
            
            logger.info(f"Loading test data from: {test_file}")
            test_data = pd.read_csv(test_file)
            
            logger.info(f"Training data shape: {train_data.shape}")
            logger.info(f"Test data shape: {test_data.shape}")
            
            return train_data, test_data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def analyze_data(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Performs comprehensive data analysis on both datasets."""
        analysis = {
            'train_shape': train_data.shape,
            'test_shape': test_data.shape,
            'train_missing_values': train_data.isnull().sum().to_dict(),
            'test_missing_values': test_data.isnull().sum().to_dict(),
            'train_duplicates': train_data.duplicated().sum(),
            'test_duplicates': test_data.duplicated().sum(),
            'train_target_distribution': train_data[self.target_column].value_counts().to_dict(),
            'test_target_distribution': test_data[self.target_column].value_counts().to_dict(),
            'numerical_features': train_data.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_features': train_data.select_dtypes(exclude=[np.number]).columns.tolist(),
            'train_memory_usage_mb': train_data.memory_usage(deep=True).sum() / 1024 / 1024,
            'test_memory_usage_mb': test_data.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        logger.info(f"Data analysis completed.")
        logger.info(f"Train imbalance ratio: {analysis['train_target_distribution'][1] / analysis['train_target_distribution'][0]:.4f}")
        
        return analysis
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes duplicate rows."""
        initial_count = len(df)
        df = df.drop_duplicates()
        removed_count = initial_count - len(df)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} duplicate rows")
        return df
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Performs feature engineering tailored to the fraud dataset."""
        if not self.config['feature_engineering']:
            return df
        
        logger.info("Performing feature engineering...")
        
        # Convert datetime columns
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
        df['dob'] = pd.to_datetime(df['dob'])
        
        # Create new features
        df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.total_seconds() / (365.25 * 24 * 3600)
        df['hour'] = df['trans_date_trans_time'].dt.hour
        df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
        df['amount_log'] = np.log1p(df['amt'])
        df['amount_sqrt'] = np.sqrt(df['amt'])
        
        logger.info("Feature engineering completed")
        return df
    
    def handle_imbalanced_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Handles imbalanced data using specified method."""
        if not self.config['handle_imbalance']:
            return X, y
        
        logger.info(f"Handling imbalanced data using {self.config['imbalance_method']}")
        
        df = pd.concat([X, y], axis=1)
        majority_class = df[df[self.target_column] == 0]
        minority_class = df[df[self.target_column] == 1]
        
        if self.config['imbalance_method'] == 'undersample':
            majority_downsampled = resample(
                majority_class,
                replace=False,
                n_samples=len(minority_class),
                random_state=42
            )
            df_balanced = pd.concat([majority_downsampled, minority_class])
        elif self.config['imbalance_method'] == 'oversample':
            minority_upsampled = resample(
                minority_class,
                replace=True,
                n_samples=len(majority_class),
                random_state=42
            )
            df_balanced = pd.concat([majority_class, minority_upsampled])
        else:
            logger.warning(f"Unknown method: {self.config['imbalance_method']}. Returning original data.")
            return X, y
        
        df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
        X_balanced = df_balanced.drop(self.target_column, axis=1)
        y_balanced = df_balanced[self.target_column]
        
        logger.info(f"Balanced dataset shape: {X_balanced.shape}")
        return X_balanced, y_balanced
    
    def preprocess_pipeline(self, train_file: str = "fraudTrain.csv", test_file: str = "fraudTest.csv", save_preprocessor: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Complete preprocessing pipeline for the fraud detection dataset."""
        logger.info("Starting preprocessing pipeline...")
        
        # Load data
        train_data, test_data = self.load_existing_data(train_file, test_file)
        
        # Analyze data
        analysis = self.analyze_data(train_data, test_data)
        
        # Identify features
        numerical_features = [col for col in analysis['numerical_features'] if col != self.target_column]
        categorical_features = analysis['categorical_features']
        
        # Handle missing values
        self.num_imputer = SimpleImputer(strategy='median')
        self.cat_imputer = SimpleImputer(strategy='most_frequent')
        
        train_data[numerical_features] = self.num_imputer.fit_transform(train_data[numerical_features])
        test_data[numerical_features] = self.num_imputer.transform(test_data[numerical_features])
        
        train_data[categorical_features] = self.cat_imputer.fit_transform(train_data[categorical_features])
        test_data[categorical_features] = self.cat_imputer.transform(test_data[categorical_features])
        
        # Remove duplicates
        train_data = self.remove_duplicates(train_data)
        test_data = self.remove_duplicates(test_data)
        
        # Feature engineering
        train_data = self.feature_engineering(train_data)
        test_data = self.feature_engineering(test_data)
        
        # Drop unnecessary columns
        columns_to_drop = ['trans_num', 'first', 'last', 'street', 'city', 'zip', 'dob', 'trans_date_trans_time', 'cc_num']
        train_data = train_data.drop(columns=[col for col in columns_to_drop if col in train_data.columns])
        test_data = test_data.drop(columns=[col for col in columns_to_drop if col in test_data.columns])
        
        # Update numerical features
        new_numerical_features = ['age', 'hour', 'day_of_week', 'amount_log', 'amount_sqrt']
        all_numerical_features = [col for col in numerical_features + new_numerical_features if col in train_data.columns and col != self.target_column]
        categorical_features = [col for col in categorical_features if col in train_data.columns and col not in columns_to_drop]
        
        # Scale numerical features
        self.scaler = RobustScaler() if self.config['scaler_type'] == 'robust' else StandardScaler()
        train_data[all_numerical_features] = self.scaler.fit_transform(train_data[all_numerical_features])
        test_data[all_numerical_features] = self.scaler.transform(test_data[all_numerical_features])
        
        # Encode categorical features
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        train_encoded = self.encoder.fit_transform(train_data[categorical_features])
        test_encoded = self.encoder.transform(test_data[categorical_features])
        
        train_encoded_df = pd.DataFrame(train_encoded, columns=self.encoder.get_feature_names_out(categorical_features), index=train_data.index)
        test_encoded_df = pd.DataFrame(test_encoded, columns=self.encoder.get_feature_names_out(categorical_features), index=test_data.index)
        
        # Combine features
        X_train = pd.concat([train_data[all_numerical_features], train_encoded_df], axis=1)
        X_test = pd.concat([test_data[all_numerical_features], test_encoded_df], axis=1)
        y_train = train_data[self.target_column]
        y_test = test_data[self.target_column]
        
        # Handle imbalance
        X_train, y_train = self.handle_imbalanced_data(X_train, y_train)
        
        # Store feature columns
        self.feature_columns = X_train.columns.tolist()
        
        # Save preprocessor
        if save_preprocessor:
            Path('models').mkdir(exist_ok=True)
            self.save_preprocessor('models/preprocessor.pkl')
        
        logger.info("Preprocessing pipeline completed successfully!")
        return X_train, X_test, y_train, y_test
    
    def save_preprocessor(self, file_path: str):
        """Saves the fitted preprocessor."""
        preprocessor_data = {
            'scaler': self.scaler,
            'num_imputer': self.num_imputer,
            'cat_imputer': self.cat_imputer,
            'encoder': self.encoder,
            'feature_columns': self.feature_columns,
            'config': self.config
        }
        joblib.dump(preprocessor_data, file_path)
        logger.info(f"Preprocessor saved to: {file_path}")
    
    def load_preprocessor(self, file_path: str):
        """Loads a previously saved preprocessor."""
        preprocessor_data = joblib.load(file_path)
        self.scaler = preprocessor_data['scaler']
        self.num_imputer = preprocessor_data['num_imputer']
        self.cat_imputer = preprocessor_data['cat_imputer']
        self.encoder = preprocessor_data['encoder']
        self.feature_columns = preprocessor_data['feature_columns']
        self.config = preprocessor_data['config']
        logger.info(f"Preprocessor loaded from: {file_path}")

def main():
    """Main function to run the preprocessing pipeline."""
    preprocessor = FraudDataPreprocessor()
    try:
        X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(
            train_file="fraudTrain.csv",  # Adjust this path as needed
            test_file="fraudTest.csv"      # Adjust this path as needed
        )
        print("✅ Preprocessing completed successfully!")
        print(f"📊 Training set shape: {X_train.shape}")
        print(f"📊 Test set shape: {X_test.shape}")
        
        Path('data').mkdir(exist_ok=True)
        X_train.to_csv('data/X_train.csv', index=False)
        X_test.to_csv('data/X_test.csv', index=False)
        y_train.to_csv('data/y_train.csv', index=False)
        y_test.to_csv('data/y_test.csv', index=False)
        
        print("💾 Processed data saved to data/ directory")
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")

if __name__ == "__main__":
    main()