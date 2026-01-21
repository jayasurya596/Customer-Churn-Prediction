"""
Data Preprocessing Module for Customer Churn Prediction
Handles data loading, cleaning, encoding, and splitting
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

class DataPreprocessor:
    def __init__(self, data_path='data/telecom_churn.csv'):
        """Initialize the data preprocessor"""
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
    def load_data(self):
        """Load the customer churn dataset"""
        print("Loading dataset...")
        self.data = pd.read_csv(self.data_path)
        print(f"Dataset loaded: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        return self.data
    
    def explore_data(self):
        """Display basic information about the dataset"""
        print("\n=== Dataset Information ===")
        print(self.data.info())
        print("\n=== First Few Rows ===")
        print(self.data.head())
        print("\n=== Statistical Summary ===")
        print(self.data.describe())
        print("\n=== Missing Values ===")
        print(self.data.isnull().sum())
        print("\n=== Churn Distribution ===")
        print(self.data['Churn'].value_counts())
        
    def clean_data(self):
        """Clean the dataset"""
        print("\nCleaning data...")
        
        # Remove duplicates
        initial_rows = len(self.data)
        self.data = self.data.drop_duplicates()
        print(f"Removed {initial_rows - len(self.data)} duplicate rows")
        
        # Handle TotalCharges - convert to numeric and handle missing values
        if 'TotalCharges' in self.data.columns:
            self.data['TotalCharges'] = pd.to_numeric(self.data['TotalCharges'], errors='coerce')
            # Fill missing TotalCharges with median
            self.data['TotalCharges'].fillna(self.data['TotalCharges'].median(), inplace=True)
        
        # Drop customerID if present (not useful for prediction)
        if 'customerID' in self.data.columns:
            self.data = self.data.drop('customerID', axis=1)
            print("Dropped customerID column")
        
        # Handle any remaining missing values
        missing_before = self.data.isnull().sum().sum()
        if missing_before > 0:
            # Fill numerical columns with median
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if self.data[col].isnull().sum() > 0:
                    self.data[col].fillna(self.data[col].median(), inplace=True)
            
            # Fill categorical columns with mode
            categorical_cols = self.data.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if self.data[col].isnull().sum() > 0:
                    self.data[col].fillna(self.data[col].mode()[0], inplace=True)
            
            print(f"Handled {missing_before} missing values")
        
        print("Data cleaning completed")
        
    def encode_features(self):
        """Encode categorical variables"""
        print("\nEncoding categorical features...")
        
        # Separate target variable
        if 'Churn' in self.data.columns:
            # Encode target variable (Yes/No to 1/0)
            self.data['Churn'] = self.data['Churn'].map({'Yes': 1, 'No': 0})
            if self.data['Churn'].isnull().any():
                # If mapping didn't work, try label encoding
                le = LabelEncoder()
                self.data['Churn'] = le.fit_transform(self.data['Churn'].astype(str))
        
        # Get categorical columns (excluding target)
        categorical_cols = self.data.select_dtypes(include=['object']).columns.tolist()
        
        # Binary categorical variables - use label encoding
        binary_cols = []
        for col in categorical_cols:
            if self.data[col].nunique() == 2:
                binary_cols.append(col)
                le = LabelEncoder()
                self.data[col] = le.fit_transform(self.data[col])
                self.label_encoders[col] = le
        
        # Multi-class categorical variables - use one-hot encoding
        multi_class_cols = [col for col in categorical_cols if col not in binary_cols]
        if multi_class_cols:
            self.data = pd.get_dummies(self.data, columns=multi_class_cols, drop_first=True)
            print(f"One-hot encoded columns: {multi_class_cols}")
        
        print(f"Label encoded columns: {binary_cols}")
        print(f"Final dataset shape: {self.data.shape}")
        
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        print(f"\nSplitting data (test size: {test_size})...")
        
        # Separate features and target
        X = self.data.drop('Churn', axis=1)
        y = self.data['Churn']
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        print(f"Churn rate in training set: {self.y_train.mean():.2%}")
        print(f"Churn rate in test set: {self.y_test.mean():.2%}")
        
    def scale_features(self):
        """Scale numerical features"""
        print("\nScaling features...")
        
        # Fit scaler on training data and transform both train and test
        self.X_train = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.feature_names,
            index=self.X_train.index
        )
        
        self.X_test = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.feature_names,
            index=self.X_test.index
        )
        
        print("Feature scaling completed")
        
    def get_processed_data(self):
        """Return processed train and test data"""
        return self.X_train, self.X_test, self.y_train, self.y_test, self.feature_names
    
    def process_pipeline(self):
        """Execute the complete preprocessing pipeline"""
        print("=" * 60)
        print("STARTING DATA PREPROCESSING PIPELINE")
        print("=" * 60)
        
        self.load_data()
        self.explore_data()
        self.clean_data()
        self.encode_features()
        self.split_data()
        self.scale_features()
        
        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
        return self.get_processed_data()


def load_and_clean_data(data_path='data/telecom_churn.csv'):
    """
    Convenience function to load and preprocess data
    Returns: X_train, X_test, y_train, y_test, feature_names
    """
    preprocessor = DataPreprocessor(data_path)
    return preprocessor.process_pipeline()


if __name__ == "__main__":
    # Test the preprocessing pipeline
    X_train, X_test, y_train, y_test, feature_names = load_and_clean_data()
    print(f"\nFinal shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"Number of features: {len(feature_names)}")
