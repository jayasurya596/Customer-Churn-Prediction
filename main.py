"""
Main Pipeline for Customer Churn Prediction
Orchestrates the entire workflow from data loading to business impact analysis
"""

import os
import sys
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from src.data_preprocessing import DataPreprocessor
from src.eda import perform_eda
from src.model_training import ChurnModelTrainer
from src.evaluation import evaluate_models
from src.business_impact import analyze_business_impact

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def main():
    """Main execution pipeline"""
    print_header("CUSTOMER CHURN PREDICTION - COMPLETE PIPELINE")
    
    # Configuration
    DATA_PATH = 'data/telecom_churn.csv'
    USE_SMOTE = True
    TUNE_HYPERPARAMETERS = False  # Set to True for better results (takes longer)
    
    # Check if data file exists
    if not os.path.exists(DATA_PATH):
        print(f"\n❌ ERROR: Dataset not found at {DATA_PATH}")
        print("\nPlease download the Telco Customer Churn dataset and place it in the 'data' folder.")
        print("You can download it from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
        print("\nAlternatively, you can use any customer churn dataset with a 'Churn' column.")
        return
    
    # ========================================
    # STEP 1: DATA PREPROCESSING
    # ========================================
    print_header("STEP 1: DATA PREPROCESSING")
    
    preprocessor = DataPreprocessor(DATA_PATH)
    X_train, X_test, y_train, y_test, feature_names = preprocessor.process_pipeline()
    
    print(f"\n✅ Data preprocessing completed successfully!")
    print(f"   - Training samples: {X_train.shape[0]}")
    print(f"   - Test samples: {X_test.shape[0]}")
    print(f"   - Number of features: {len(feature_names)}")
    
    # ========================================
    # STEP 2: EXPLORATORY DATA ANALYSIS
    # ========================================
    print_header("STEP 2: EXPLORATORY DATA ANALYSIS")
    
    # Load raw data for EDA
    raw_data = pd.read_csv(DATA_PATH)
    perform_eda(raw_data)
    
    print(f"\n✅ EDA completed successfully!")
    print(f"   - Visualizations saved to 'visualizations/' folder")
    
    # ========================================
    # STEP 3: MODEL TRAINING
    # ========================================
    print_header("STEP 3: MODEL TRAINING")
    
    trainer = ChurnModelTrainer(X_train, X_test, y_train, y_test, feature_names)
    xgb_model, rf_model = trainer.train_all_models(
        use_smote=USE_SMOTE, 
        tune_hyperparameters=TUNE_HYPERPARAMETERS
    )
    
    print(f"\n✅ Model training completed successfully!")
    print(f"   - XGBoost model saved to 'models/xgboost_model.pkl'")
    print(f"   - Random Forest model saved to 'models/random_forest_model.pkl'")
    print(f"   - Feature importance plots saved to 'visualizations/'")
    
    # ========================================
    # STEP 4: MODEL EVALUATION
    # ========================================
    print_header("STEP 4: MODEL EVALUATION")
    
    results, comparison_df = evaluate_models(
        [xgb_model, rf_model], 
        X_test, 
        y_test, 
        ['XGBoost', 'Random Forest']
    )
    
    print(f"\n✅ Model evaluation completed successfully!")
    print(f"   - Confusion matrices saved to 'visualizations/confusion_matrices.png'")
    print(f"   - ROC curves saved to 'visualizations/roc_curves.png'")
    print(f"   - Model comparison saved to 'visualizations/model_comparison.png'")
    
    # ========================================
    # STEP 5: BUSINESS IMPACT ANALYSIS
    # ========================================
    print_header("STEP 5: BUSINESS IMPACT ANALYSIS")
    
    # Use the better performing model (based on ROC-AUC)
    if results['XGBoost']['roc_auc'] >= results['Random Forest']['roc_auc']:
        best_model_name = 'XGBoost'
        best_model = xgb_model
        best_results = results['XGBoost']
    else:
        best_model_name = 'Random Forest'
        best_model = rf_model
        best_results = results['Random Forest']
    
    print(f"\nUsing {best_model_name} for business impact analysis (ROC-AUC: {best_results['roc_auc']:.4f})")
    
    impact_results, recommendations = analyze_business_impact(
        y_test,
        best_results['y_pred'],
        best_results['y_pred_proba'],
        best_model_name,
        clv=2000,  # Average Customer Lifetime Value
        cac=500,   # Customer Acquisition Cost
        retention_cost=100,  # Cost per retention campaign
        success_rate=0.7  # 70% retention campaign success rate
    )
    
    print(f"\n✅ Business impact analysis completed successfully!")
    print(f"   - Business impact visualizations saved to 'visualizations/business_impact_analysis.png'")
    
    # ========================================
    # FINAL SUMMARY
    # ========================================
    print_header("FINAL SUMMARY")
    
    print("\n📊 MODEL PERFORMANCE:")
    print(f"   XGBoost ROC-AUC: {results['XGBoost']['roc_auc']:.4f}")
    print(f"   Random Forest ROC-AUC: {results['Random Forest']['roc_auc']:.4f}")
    print(f"   XGBoost F1-Score: {results['XGBoost']['f1_binary']:.4f}")
    print(f"   Random Forest F1-Score: {results['Random Forest']['f1_binary']:.4f}")
    
    print("\n💰 BUSINESS IMPACT:")
    print(f"   Net Savings: ${impact_results['net_savings']:,.2f}")
    print(f"   Savings per Customer: ${impact_results['savings_per_customer']:,.2f}")
    print(f"   Customers Successfully Retained: {int(impact_results['retained_customers'])}")
    
    print("\n📁 OUTPUT FILES:")
    print("   All visualizations saved to: visualizations/")
    print("   All models saved to: models/")
    
    print("\n" + "=" * 70)
    print("  ✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    
    print("\n📌 NEXT STEPS:")
    print("   1. Review visualizations in the 'visualizations/' folder")
    print("   2. Examine feature importance to understand churn drivers")
    print("   3. Implement retention campaigns for high-risk customers")
    print("   4. Monitor model performance and retrain periodically")
    
    return {
        'models': {'xgboost': xgb_model, 'random_forest': rf_model},
        'results': results,
        'comparison': comparison_df,
        'impact': impact_results,
        'recommendations': recommendations
    }

if __name__ == "__main__":
    try:
        pipeline_results = main()
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
