"""
Exploratory Data Analysis Module for Customer Churn Prediction
Generates visualizations and statistical insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class ChurnEDA:
    def __init__(self, data):
        """Initialize EDA with dataset"""
        self.data = data
        self.viz_dir = 'visualizations'
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # Set style for better visualizations
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        
    def analyze_churn_distribution(self):
        """Analyze and visualize churn distribution"""
        print("\n=== Churn Distribution Analysis ===")
        
        # Map churn values if needed
        churn_col = self.data['Churn']
        if churn_col.dtype == 'object':
            churn_counts = churn_col.value_counts()
        else:
            churn_counts = churn_col.map({0: 'No', 1: 'Yes'}).value_counts()
        
        print(churn_counts)
        print(f"\nChurn Rate: {(churn_counts.get('Yes', churn_counts.get(1, 0)) / len(self.data)):.2%}")
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar chart
        if churn_col.dtype == 'object':
            churn_col.value_counts().plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'])
        else:
            churn_col.map({0: 'No', 1: 'Yes'}).value_counts().plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'])
        axes[0].set_title('Churn Distribution (Bar Chart)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Churn')
        axes[0].set_ylabel('Count')
        axes[0].tick_params(axis='x', rotation=0)
        
        # Pie chart
        if churn_col.dtype == 'object':
            churn_col.value_counts().plot(kind='pie', ax=axes[1], autopct='%1.1f%%', 
                                          colors=['#2ecc71', '#e74c3c'], startangle=90)
        else:
            churn_col.map({0: 'No', 1: 'Yes'}).value_counts().plot(kind='pie', ax=axes[1], 
                                                                     autopct='%1.1f%%', 
                                                                     colors=['#2ecc71', '#e74c3c'], 
                                                                     startangle=90)
        axes[1].set_title('Churn Distribution (Pie Chart)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('')
        
        plt.tight_layout()
        plt.savefig(f'{self.viz_dir}/churn_distribution.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {self.viz_dir}/churn_distribution.png")
        plt.close()
        
    def analyze_numerical_features(self):
        """Analyze numerical features"""
        print("\n=== Numerical Features Analysis ===")
        
        # Make a copy to avoid modifying original data
        data_copy = self.data.copy()
        
        # Convert TotalCharges to numeric if it exists
        if 'TotalCharges' in data_copy.columns:
            data_copy['TotalCharges'] = pd.to_numeric(data_copy['TotalCharges'], errors='coerce')
        
        # Get numerical columns
        numerical_cols = data_copy.select_dtypes(include=[np.number]).columns.tolist()
        if 'Churn' in numerical_cols:
            numerical_cols.remove('Churn')
        
        # Common numerical features in telecom churn dataset
        key_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        available_features = [col for col in key_features if col in data_copy.columns]
        
        if not available_features:
            available_features = numerical_cols[:3]  # Take first 3 numerical features
        
        if available_features:
            fig, axes = plt.subplots(len(available_features), 2, figsize=(14, 5*len(available_features)))
            if len(available_features) == 1:
                axes = axes.reshape(1, -1)
            
            for idx, feature in enumerate(available_features):
                # Distribution plot
                data_copy[feature].dropna().hist(bins=30, ax=axes[idx, 0], color='#3498db', edgecolor='black')
                axes[idx, 0].set_title(f'{feature} Distribution', fontsize=12, fontweight='bold')
                axes[idx, 0].set_xlabel(feature)
                axes[idx, 0].set_ylabel('Frequency')
                
                # Box plot by churn
                churn_col = data_copy['Churn']
                if churn_col.dtype == 'object':
                    # Create a clean dataframe for boxplot
                    plot_data = data_copy[[feature, 'Churn']].dropna()
                    plot_data.boxplot(column=feature, by='Churn', ax=axes[idx, 1])
                else:
                    temp_df = data_copy[[feature, 'Churn']].dropna().copy()
                    temp_df['Churn_Label'] = temp_df['Churn'].map({0: 'No', 1: 'Yes'})
                    temp_df.boxplot(column=feature, by='Churn_Label', ax=axes[idx, 1])
                
                axes[idx, 1].set_title(f'{feature} by Churn Status', fontsize=12, fontweight='bold')
                axes[idx, 1].set_xlabel('Churn')
                axes[idx, 1].set_ylabel(feature)
                plt.sca(axes[idx, 1])
                plt.xticks(rotation=0)
            
            plt.tight_layout()
            plt.savefig(f'{self.viz_dir}/numerical_features_analysis.png', dpi=300, bbox_inches='tight')
            print(f"Saved: {self.viz_dir}/numerical_features_analysis.png")
            plt.close()
            
    def analyze_categorical_features(self):
        """Analyze categorical features"""
        print("\n=== Categorical Features Analysis ===")
        
        # Get categorical columns
        categorical_cols = self.data.select_dtypes(include=['object']).columns.tolist()
        if 'Churn' in categorical_cols:
            categorical_cols.remove('Churn')
        
        # Common categorical features
        key_features = ['Contract', 'PaymentMethod', 'InternetService', 'gender']
        available_features = [col for col in key_features if col in categorical_cols]
        
        if not available_features:
            available_features = categorical_cols[:4]  # Take first 4 categorical features
        
        if available_features:
            n_features = len(available_features)
            n_cols = 2
            n_rows = (n_features + 1) // 2
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5*n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            axes = axes.flatten()
            
            for idx, feature in enumerate(available_features):
                # Create crosstab
                churn_col = self.data['Churn']
                if churn_col.dtype == 'object':
                    ct = pd.crosstab(self.data[feature], self.data['Churn'], normalize='index') * 100
                else:
                    temp_df = self.data.copy()
                    temp_df['Churn_Label'] = temp_df['Churn'].map({0: 'No', 1: 'Yes'})
                    ct = pd.crosstab(self.data[feature], temp_df['Churn_Label'], normalize='index') * 100
                
                ct.plot(kind='bar', ax=axes[idx], color=['#2ecc71', '#e74c3c'])
                axes[idx].set_title(f'Churn Rate by {feature}', fontsize=12, fontweight='bold')
                axes[idx].set_xlabel(feature)
                axes[idx].set_ylabel('Percentage (%)')
                axes[idx].legend(title='Churn', loc='best')
                axes[idx].tick_params(axis='x', rotation=45)
            
            # Hide unused subplots
            for idx in range(len(available_features), len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'{self.viz_dir}/categorical_features_analysis.png', dpi=300, bbox_inches='tight')
            print(f"Saved: {self.viz_dir}/categorical_features_analysis.png")
            plt.close()
            
    def correlation_analysis(self):
        """Analyze feature correlations"""
        print("\n=== Correlation Analysis ===")
        
        # Get numerical columns only
        numerical_data = self.data.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        corr_matrix = numerical_data.corr()
        
        # Create heatmap
        plt.figure(figsize=(14, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f'{self.viz_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {self.viz_dir}/correlation_heatmap.png")
        plt.close()
        
        # Print top correlations with Churn
        if 'Churn' in corr_matrix.columns:
            print("\nTop correlations with Churn:")
            churn_corr = corr_matrix['Churn'].sort_values(ascending=False)
            print(churn_corr[churn_corr.index != 'Churn'].head(10))
            
    def generate_summary_statistics(self):
        """Generate and save summary statistics"""
        print("\n=== Summary Statistics ===")
        
        # Overall statistics
        print("\nDataset Shape:", self.data.shape)
        print("\nData Types:")
        print(self.data.dtypes.value_counts())
        
        # Numerical summary
        print("\nNumerical Features Summary:")
        print(self.data.describe())
        
        # Categorical summary
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print("\nCategorical Features Summary:")
            for col in categorical_cols:
                print(f"\n{col}:")
                print(self.data[col].value_counts())
                
    def run_full_eda(self):
        """Execute complete EDA pipeline"""
        print("=" * 60)
        print("STARTING EXPLORATORY DATA ANALYSIS")
        print("=" * 60)
        
        self.generate_summary_statistics()
        self.analyze_churn_distribution()
        self.analyze_numerical_features()
        self.analyze_categorical_features()
        self.correlation_analysis()
        
        print("\n" + "=" * 60)
        print("EDA COMPLETED - All visualizations saved to 'visualizations/' folder")
        print("=" * 60)


def perform_eda(data):
    """
    Convenience function to perform EDA
    """
    eda = ChurnEDA(data)
    eda.run_full_eda()
    return eda


if __name__ == "__main__":
    # Test EDA with sample data
    from data_preprocessing import load_and_clean_data
    
    # Load raw data for EDA
    import pandas as pd
    data = pd.read_csv('data/telecom_churn.csv')
    
    # Perform EDA
    perform_eda(data)
