# Customer Churn Prediction Model

A comprehensive machine learning solution for predicting customer churn in the telecom industry using **XGBoost** and **Random Forest** classifiers. This project includes exploratory data analysis, model training, evaluation, feature importance analysis, and business impact assessment.

## 🎯 Project Objective

Predict customers likely to leave a service and identify the key factors driving churn to enable targeted retention strategies and reduce customer acquisition costs.

## 📊 Dataset

This project uses the **Telco Customer Churn** dataset, which includes:
- **Customer Demographics**: Gender, senior citizen status, partner, dependents
- **Services Subscribed**: Phone service, internet service, online security, tech support, streaming services
- **Account Information**: Tenure, contract type, payment method, monthly charges, total charges
- **Churn Label**: Whether the customer churned (Yes/No)

### Download Dataset

You can download the dataset from [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

Place the downloaded `WA_Fn-UseC_-Telco-Customer-Churn.csv` file in the `data/` folder and rename it to `telecom_churn.csv`.

## 🚀 Features

- ✅ **Data Preprocessing**: Automated data cleaning, encoding, and feature scaling
- ✅ **Exploratory Data Analysis**: Comprehensive visualizations and statistical insights
- ✅ **Class Imbalance Handling**: SMOTE (Synthetic Minority Over-sampling Technique)
- ✅ **Model Training**: XGBoost and Random Forest with hyperparameter tuning
- ✅ **Feature Importance**: Identify top factors causing churn
- ✅ **Model Evaluation**: ROC-AUC, F1-Score, Confusion Matrix, ROC Curves
- ✅ **Business Impact Analysis**: Cost savings, ROI, and actionable recommendations

## 🛠️ Tech Stack

- **Python 3.8+**
- **XGBoost**: Gradient boosting classifier
- **Random Forest**: Ensemble learning method
- **scikit-learn**: Machine learning utilities
- **pandas & numpy**: Data manipulation
- **matplotlib & seaborn**: Data visualization
- **imbalanced-learn**: SMOTE for class imbalance

## 📦 Installation

### 1. Clone or Navigate to the Project Directory

```bash
cd "Customer Churn Prediction"
```

### 2. Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
```

**Activate the virtual environment:**

- **Windows:**
  ```bash
  venv\Scripts\activate
  ```

- **macOS/Linux:**
  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 📂 Project Structure

```
Customer Churn Prediction/
├── data/
│   └── telecom_churn.csv          # Dataset (download separately)
├── src/
│   ├── data_preprocessing.py      # Data loading and cleaning
│   ├── eda.py                     # Exploratory data analysis
│   ├── model_training.py          # Model training with XGBoost & RF
│   ├── evaluation.py              # Model evaluation metrics
│   └── business_impact.py         # Business impact analysis
├── models/
│   ├── xgboost_model.pkl          # Trained XGBoost model
│   └── random_forest_model.pkl    # Trained Random Forest model
├── visualizations/                # Generated plots and charts
├── notebooks/                     # Jupyter notebooks (optional)
├── requirements.txt               # Python dependencies
├── main.py                        # Main pipeline script
└── README.md                      # This file
```

## 🎮 Usage

### Run the Complete Pipeline

Execute the entire workflow from data preprocessing to business impact analysis:

```bash
python main.py
```

This will:
1. Load and clean the customer data
2. Perform exploratory data analysis
3. Train XGBoost and Random Forest models
4. Evaluate models using ROC-AUC and F1-score
5. Identify top features causing churn
6. Calculate business impact and cost savings

### Run Individual Modules

You can also run individual components:

**Data Preprocessing:**
```bash
python src/data_preprocessing.py
```

**Exploratory Data Analysis:**
```bash
python src/eda.py
```

**Model Training:**
```bash
python src/model_training.py
```

**Model Evaluation:**
```bash
python src/evaluation.py
```

**Business Impact Analysis:**
```bash
python src/business_impact.py
```

## 📈 Model Performance

### XGBoost
- **ROC-AUC Score**: ~0.84
- **F1-Score**: ~0.75
- **Precision**: ~0.73
- **Recall**: ~0.77

### Random Forest
- **ROC-AUC Score**: ~0.82
- **F1-Score**: ~0.73
- **Precision**: ~0.71
- **Recall**: ~0.75

*Note: Actual performance may vary based on hyperparameter tuning and data preprocessing.*

## 🔍 Key Findings

### Top Features Causing Churn

1. **Contract Type**: Month-to-month contracts have higher churn rates
2. **Tenure**: Customers with shorter tenure are more likely to churn
3. **Monthly Charges**: Higher monthly charges correlate with increased churn
4. **Internet Service**: Fiber optic users show higher churn rates
5. **Payment Method**: Electronic check users have higher churn probability
6. **Tech Support**: Lack of tech support increases churn likelihood
7. **Online Security**: Customers without online security churn more
8. **Total Charges**: Lower total charges indicate higher churn risk

## 💼 Business Impact

### Cost Savings Analysis

Based on the business impact analysis with the following assumptions:
- **Average Customer Lifetime Value (CLV)**: $2,000
- **Customer Acquisition Cost (CAC)**: $500
- **Retention Cost per Customer**: $100
- **Retention Campaign Success Rate**: 70%

### Estimated Savings

- **Net Savings**: ~$150,000 - $200,000 (varies by dataset size)
- **Savings per Customer**: ~$75 - $100
- **ROI**: 300% - 400%

### Key Recommendations

1. **Implement Targeted Retention Campaigns**: Focus on high-risk customers identified by the model
2. **Improve Contract Offerings**: Encourage long-term contracts with incentives
3. **Enhance Customer Support**: Provide better tech support and online security
4. **Optimize Pricing Strategy**: Review pricing for high-churn segments
5. **Monitor Key Metrics**: Track tenure, contract type, and service usage patterns

## 📊 Visualizations

All visualizations are automatically saved to the `visualizations/` folder:

- `churn_distribution.png` - Churn rate distribution
- `numerical_features_analysis.png` - Analysis of numerical features
- `categorical_features_analysis.png` - Analysis of categorical features
- `correlation_heatmap.png` - Feature correlation matrix
- `feature_importance.png` - Top features from both models
- `confusion_matrices.png` - Confusion matrices for both models
- `roc_curves.png` - ROC curves comparison
- `model_comparison.png` - Side-by-side model performance
- `business_impact_analysis.png` - Business impact visualizations

## 🔧 Configuration

You can customize the pipeline by modifying parameters in `main.py`:

```python
# Configuration
DATA_PATH = 'data/telecom_churn.csv'
USE_SMOTE = True  # Handle class imbalance
TUNE_HYPERPARAMETERS = False  # Set to True for better results (takes longer)

# Business parameters in business_impact.py
clv = 2000  # Customer Lifetime Value
cac = 500   # Customer Acquisition Cost
retention_cost = 100  # Cost per retention campaign
success_rate = 0.7  # Retention campaign success rate
```

## 🧪 Testing

To verify the installation and setup:

```bash
# Test data preprocessing
python -c "from src.data_preprocessing import load_and_clean_data; X_train, X_test, y_train, y_test, features = load_and_clean_data(); print(f'Success! Train: {X_train.shape}, Test: {X_test.shape}')"
```

## 📝 Future Improvements

- [ ] Add deep learning models (Neural Networks, LSTM)
- [ ] Implement real-time prediction API
- [ ] Add customer segmentation analysis
- [ ] Create interactive dashboard with Streamlit/Dash
- [ ] Add A/B testing framework for retention strategies
- [ ] Implement automated model retraining pipeline
- [ ] Add explainability with SHAP values

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open-source and available under the MIT License.

## 👨‍💻 Author

Created as part of a machine learning project focused on customer retention and business analytics.

## 📧 Contact

For questions or feedback, please open an issue in the repository.

---

**⭐ If you find this project helpful, please consider giving it a star!**
