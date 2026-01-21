"""
Model Training Module for Customer Churn Prediction
Trains XGBoost and Random Forest models with hyperparameter tuning
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

class ChurnModelTrainer:
    def __init__(self, X_train, X_test, y_train, y_test, feature_names):
        """Initialize the model trainer"""
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names
        
        self.xgb_model = None
        self.rf_model = None
        
        self.models_dir = 'models'
        self.viz_dir = 'visualizations'
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)
        
    def handle_class_imbalance(self, method='smote'):
        """Handle class imbalance using SMOTE or class weights"""
        print("\n=== Handling Class Imbalance ===")
        print(f"Original class distribution:")
        print(f"Class 0: {(self.y_train == 0).sum()} ({(self.y_train == 0).sum() / len(self.y_train):.2%})")
        print(f"Class 1: {(self.y_train == 1).sum()} ({(self.y_train == 1).sum() / len(self.y_train):.2%})")
        
        if method == 'smote':
            print("\nApplying SMOTE...")
            smote = SMOTE(random_state=42)
            self.X_train_balanced, self.y_train_balanced = smote.fit_resample(self.X_train, self.y_train)
            
            print(f"After SMOTE:")
            print(f"Class 0: {(self.y_train_balanced == 0).sum()}")
            print(f"Class 1: {(self.y_train_balanced == 1).sum()}")
            
            return self.X_train_balanced, self.y_train_balanced
        else:
            # Will use class_weight='balanced' in model training
            return self.X_train, self.y_train
    
    def train_xgboost(self, use_smote=True, tune_hyperparameters=True):
        """Train XGBoost model"""
        print("\n" + "=" * 60)
        print("TRAINING XGBOOST MODEL")
        print("=" * 60)
        
        # Handle class imbalance
        if use_smote:
            X_train, y_train = self.handle_class_imbalance(method='smote')
        else:
            X_train, y_train = self.X_train, self.y_train
        
        if tune_hyperparameters:
            print("\nPerforming hyperparameter tuning...")
            
            # Define parameter grid
            param_grid = {
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'n_estimators': [100, 200],
                'min_child_weight': [1, 3],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
            
            # Create base model
            xgb_base = XGBClassifier(random_state=42, eval_metric='logloss')
            
            # Grid search
            grid_search = GridSearchCV(
                xgb_base, param_grid, cv=3, scoring='roc_auc', 
                n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            print(f"\nBest parameters: {grid_search.best_params_}")
            print(f"Best cross-validation ROC-AUC: {grid_search.best_score_:.4f}")
            
            self.xgb_model = grid_search.best_estimator_
        else:
            print("\nTraining with default parameters...")
            self.xgb_model = XGBClassifier(
                max_depth=5,
                learning_rate=0.1,
                n_estimators=200,
                random_state=42,
                eval_metric='logloss'
            )
            self.xgb_model.fit(X_train, y_train)
        
        # Cross-validation
        print("\nPerforming 5-fold cross-validation...")
        cv_scores = cross_val_score(self.xgb_model, X_train, y_train, 
                                     cv=5, scoring='roc_auc', n_jobs=-1)
        print(f"Cross-validation ROC-AUC scores: {cv_scores}")
        print(f"Mean CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Save model
        model_path = f'{self.models_dir}/xgboost_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.xgb_model, f)
        print(f"\nModel saved to: {model_path}")
        
        return self.xgb_model
    
    def train_random_forest(self, use_smote=True, tune_hyperparameters=True):
        """Train Random Forest model"""
        print("\n" + "=" * 60)
        print("TRAINING RANDOM FOREST MODEL")
        print("=" * 60)
        
        # Handle class imbalance
        if use_smote:
            X_train, y_train = self.handle_class_imbalance(method='smote')
        else:
            X_train, y_train = self.X_train, self.y_train
        
        if tune_hyperparameters:
            print("\nPerforming hyperparameter tuning...")
            
            # Define parameter grid
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
            
            # Create base model
            rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
            
            # Grid search
            grid_search = GridSearchCV(
                rf_base, param_grid, cv=3, scoring='roc_auc', 
                n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            print(f"\nBest parameters: {grid_search.best_params_}")
            print(f"Best cross-validation ROC-AUC: {grid_search.best_score_:.4f}")
            
            self.rf_model = grid_search.best_estimator_
        else:
            print("\nTraining with default parameters...")
            self.rf_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            self.rf_model.fit(X_train, y_train)
        
        # Cross-validation
        print("\nPerforming 5-fold cross-validation...")
        cv_scores = cross_val_score(self.rf_model, X_train, y_train, 
                                     cv=5, scoring='roc_auc', n_jobs=-1)
        print(f"Cross-validation ROC-AUC scores: {cv_scores}")
        print(f"Mean CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Save model
        model_path = f'{self.models_dir}/random_forest_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.rf_model, f)
        print(f"\nModel saved to: {model_path}")
        
        return self.rf_model
    
    def get_feature_importance(self, top_n=15):
        """Extract and visualize feature importance from both models"""
        print("\n" + "=" * 60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 60)
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        
        # XGBoost feature importance
        if self.xgb_model is not None:
            xgb_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.xgb_model.feature_importances_
            }).sort_values('importance', ascending=False).head(top_n)
            
            print("\nTop features from XGBoost:")
            print(xgb_importance)
            
            sns.barplot(data=xgb_importance, x='importance', y='feature', ax=axes[0], palette='viridis')
            axes[0].set_title('XGBoost - Top Feature Importance', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Importance Score')
            axes[0].set_ylabel('Features')
        
        # Random Forest feature importance
        if self.rf_model is not None:
            rf_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.rf_model.feature_importances_
            }).sort_values('importance', ascending=False).head(top_n)
            
            print("\nTop features from Random Forest:")
            print(rf_importance)
            
            sns.barplot(data=rf_importance, x='importance', y='feature', ax=axes[1], palette='magma')
            axes[1].set_title('Random Forest - Top Feature Importance', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Importance Score')
            axes[1].set_ylabel('Features')
        
        plt.tight_layout()
        plt.savefig(f'{self.viz_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
        print(f"\nFeature importance plot saved to: {self.viz_dir}/feature_importance.png")
        plt.close()
        
        return xgb_importance if self.xgb_model else None, rf_importance if self.rf_model else None
    
    def train_all_models(self, use_smote=True, tune_hyperparameters=False):
        """Train both models"""
        print("\n" + "=" * 60)
        print("STARTING MODEL TRAINING PIPELINE")
        print("=" * 60)
        
        # Train XGBoost
        self.train_xgboost(use_smote=use_smote, tune_hyperparameters=tune_hyperparameters)
        
        # Train Random Forest
        self.train_random_forest(use_smote=use_smote, tune_hyperparameters=tune_hyperparameters)
        
        # Get feature importance
        self.get_feature_importance()
        
        print("\n" + "=" * 60)
        print("MODEL TRAINING COMPLETED")
        print("=" * 60)
        
        return self.xgb_model, self.rf_model


def train_models(X_train, X_test, y_train, y_test, feature_names, 
                 use_smote=True, tune_hyperparameters=False):
    """
    Convenience function to train models
    """
    trainer = ChurnModelTrainer(X_train, X_test, y_train, y_test, feature_names)
    return trainer.train_all_models(use_smote=use_smote, tune_hyperparameters=tune_hyperparameters)


if __name__ == "__main__":
    # Test model training
    from data_preprocessing import load_and_clean_data
    
    X_train, X_test, y_train, y_test, feature_names = load_and_clean_data()
    xgb_model, rf_model = train_models(X_train, X_test, y_train, y_test, feature_names)
