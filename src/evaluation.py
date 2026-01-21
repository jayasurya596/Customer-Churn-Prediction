"""
Model Evaluation Module for Customer Churn Prediction
Evaluates models using ROC-AUC, F1-score, confusion matrix, and ROC curves
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, f1_score, classification_report, 
    confusion_matrix, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import os

class ChurnModelEvaluator:
    def __init__(self, models, X_test, y_test, model_names=None):
        """Initialize the evaluator"""
        self.models = models if isinstance(models, list) else [models]
        self.model_names = model_names if model_names else [f'Model_{i+1}' for i in range(len(self.models))]
        self.X_test = X_test
        self.y_test = y_test
        
        self.viz_dir = 'visualizations'
        os.makedirs(self.viz_dir, exist_ok=True)
        
        self.results = {}
        
    def evaluate_single_model(self, model, model_name):
        """Evaluate a single model"""
        print(f"\n{'=' * 60}")
        print(f"EVALUATING {model_name.upper()}")
        print('=' * 60)
        
        # Make predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        f1_weighted = f1_score(self.y_test, y_pred, average='weighted')
        f1_macro = f1_score(self.y_test, y_pred, average='macro')
        f1_binary = f1_score(self.y_test, y_pred)
        
        print(f"\n{model_name} Performance Metrics:")
        print(f"{'─' * 40}")
        print(f"ROC-AUC Score:        {roc_auc:.4f}")
        print(f"F1-Score (Binary):    {f1_binary:.4f}")
        print(f"F1-Score (Weighted):  {f1_weighted:.4f}")
        print(f"F1-Score (Macro):     {f1_macro:.4f}")
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(self.y_test, y_pred, target_names=['No Churn', 'Churn']))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        # Store results
        self.results[model_name] = {
            'roc_auc': roc_auc,
            'f1_binary': f1_binary,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': cm
        }
        
        return self.results[model_name]
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        print(f"\n{'=' * 60}")
        print("GENERATING CONFUSION MATRICES")
        print('=' * 60)
        
        n_models = len(self.models)
        fig, axes = plt.subplots(1, n_models, figsize=(7*n_models, 6))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, results) in enumerate(self.results.items()):
            cm = results['confusion_matrix']
            
            # Create heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       cbar_kws={'label': 'Count'},
                       xticklabels=['No Churn', 'Churn'],
                       yticklabels=['No Churn', 'Churn'])
            
            axes[idx].set_title(f'{model_name}\nConfusion Matrix', fontsize=14, fontweight='bold')
            axes[idx].set_ylabel('Actual', fontsize=12)
            axes[idx].set_xlabel('Predicted', fontsize=12)
            
            # Add accuracy text
            accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
            axes[idx].text(0.5, -0.15, f'Accuracy: {accuracy:.2%}', 
                          ha='center', transform=axes[idx].transAxes, fontsize=11)
        
        plt.tight_layout()
        plt.savefig(f'{self.viz_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {self.viz_dir}/confusion_matrices.png")
        plt.close()
    
    def plot_roc_curves(self):
        """Plot ROC curves for all models"""
        print(f"\n{'=' * 60}")
        print("GENERATING ROC CURVES")
        print('=' * 60)
        
        plt.figure(figsize=(10, 8))
        
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
        
        for idx, (model_name, results) in enumerate(self.results.items()):
            fpr, tpr, _ = roc_curve(self.y_test, results['y_pred_proba'])
            roc_auc = results['roc_auc']
            
            plt.plot(fpr, tpr, color=colors[idx % len(colors)], lw=2,
                    label=f'{model_name} (AUC = {roc_auc:.4f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.5000)')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.viz_dir}/roc_curves.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {self.viz_dir}/roc_curves.png")
        plt.close()
    
    def compare_models(self):
        """Create a comparison table of model performance"""
        print(f"\n{'=' * 60}")
        print("MODEL PERFORMANCE COMPARISON")
        print('=' * 60)
        
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'ROC-AUC': results['roc_auc'],
                'F1-Score (Binary)': results['f1_binary'],
                'F1-Score (Weighted)': results['f1_weighted'],
                'F1-Score (Macro)': results['f1_macro']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\n", comparison_df.to_string(index=False))
        
        # Visualize comparison
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # ROC-AUC comparison
        comparison_df.plot(x='Model', y='ROC-AUC', kind='bar', ax=axes[0], 
                          color='#3498db', legend=False)
        axes[0].set_title('ROC-AUC Score Comparison', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('ROC-AUC Score', fontsize=12)
        axes[0].set_xlabel('Model', fontsize=12)
        axes[0].set_ylim([0, 1])
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].axhline(y=0.5, color='r', linestyle='--', label='Random Classifier')
        axes[0].legend()
        
        # Add value labels on bars
        for container in axes[0].containers:
            axes[0].bar_label(container, fmt='%.4f')
        
        # F1-Score comparison
        f1_data = comparison_df[['Model', 'F1-Score (Binary)', 'F1-Score (Weighted)', 'F1-Score (Macro)']]
        f1_data.set_index('Model').plot(kind='bar', ax=axes[1], width=0.8)
        axes[1].set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('F1-Score', fontsize=12)
        axes[1].set_xlabel('Model', fontsize=12)
        axes[1].set_ylim([0, 1])
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].legend(loc='lower right')
        
        plt.tight_layout()
        plt.savefig(f'{self.viz_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved: {self.viz_dir}/model_comparison.png")
        plt.close()
        
        return comparison_df
    
    def evaluate_all_models(self):
        """Evaluate all models"""
        print("\n" + "=" * 60)
        print("STARTING MODEL EVALUATION")
        print("=" * 60)
        
        # Evaluate each model
        for model, model_name in zip(self.models, self.model_names):
            self.evaluate_single_model(model, model_name)
        
        # Generate visualizations
        self.plot_confusion_matrices()
        self.plot_roc_curves()
        comparison_df = self.compare_models()
        
        print("\n" + "=" * 60)
        print("MODEL EVALUATION COMPLETED")
        print("=" * 60)
        
        return self.results, comparison_df


def evaluate_models(models, X_test, y_test, model_names=None):
    """
    Convenience function to evaluate models
    """
    evaluator = ChurnModelEvaluator(models, X_test, y_test, model_names)
    return evaluator.evaluate_all_models()


if __name__ == "__main__":
    # Test evaluation
    from data_preprocessing import load_and_clean_data
    from model_training import train_models
    
    X_train, X_test, y_train, y_test, feature_names = load_and_clean_data()
    xgb_model, rf_model = train_models(X_train, X_test, y_train, y_test, feature_names)
    
    results, comparison = evaluate_models(
        [xgb_model, rf_model], 
        X_test, y_test, 
        ['XGBoost', 'Random Forest']
    )
