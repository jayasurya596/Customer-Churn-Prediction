"""
Business Impact Analysis Module for Customer Churn Prediction
Calculates cost savings, ROI, and provides actionable recommendations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class BusinessImpactAnalyzer:
    def __init__(self, y_test, y_pred, y_pred_proba, model_name='Model'):
        """Initialize business impact analyzer"""
        self.y_test = y_test
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        self.model_name = model_name
        
        self.viz_dir = 'visualizations'
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # Business parameters (can be customized)
        self.avg_customer_lifetime_value = 2000  # Average CLV in dollars
        self.customer_acquisition_cost = 500  # Cost to acquire new customer
        self.retention_cost_per_customer = 100  # Cost to retain a customer
        self.retention_success_rate = 0.7  # 70% success rate for retention campaigns
        
    def set_business_parameters(self, clv=None, cac=None, retention_cost=None, success_rate=None):
        """Set custom business parameters"""
        if clv is not None:
            self.avg_customer_lifetime_value = clv
        if cac is not None:
            self.customer_acquisition_cost = cac
        if retention_cost is not None:
            self.retention_cost_per_customer = retention_cost
        if success_rate is not None:
            self.retention_success_rate = success_rate
    
    def calculate_confusion_matrix_costs(self):
        """Calculate costs based on confusion matrix"""
        print("\n" + "=" * 60)
        print("BUSINESS IMPACT ANALYSIS")
        print("=" * 60)
        
        print(f"\nBusiness Parameters:")
        print(f"{'─' * 40}")
        print(f"Average Customer Lifetime Value: ${self.avg_customer_lifetime_value:,.2f}")
        print(f"Customer Acquisition Cost: ${self.customer_acquisition_cost:,.2f}")
        print(f"Retention Cost per Customer: ${self.retention_cost_per_customer:,.2f}")
        print(f"Retention Campaign Success Rate: {self.retention_success_rate:.0%}")
        
        # Calculate confusion matrix components
        true_negatives = np.sum((self.y_test == 0) & (self.y_pred == 0))
        false_positives = np.sum((self.y_test == 0) & (self.y_pred == 1))
        false_negatives = np.sum((self.y_test == 1) & (self.y_pred == 0))
        true_positives = np.sum((self.y_test == 1) & (self.y_pred == 1))
        
        print(f"\nConfusion Matrix Breakdown:")
        print(f"{'─' * 40}")
        print(f"True Negatives (Correctly identified non-churners): {true_negatives}")
        print(f"False Positives (Incorrectly predicted as churners): {false_positives}")
        print(f"False Negatives (Missed churners): {false_negatives}")
        print(f"True Positives (Correctly identified churners): {true_positives}")
        
        # Calculate costs
        # Cost of false positives: wasted retention efforts
        fp_cost = false_positives * self.retention_cost_per_customer
        
        # Cost of false negatives: lost customers
        fn_cost = false_negatives * self.avg_customer_lifetime_value
        
        # Savings from true positives: retained customers
        # Only a percentage of retention campaigns succeed
        retained_customers = true_positives * self.retention_success_rate
        tp_savings = retained_customers * (self.avg_customer_lifetime_value - self.retention_cost_per_customer)
        
        # Cost of retention campaigns for true positives
        tp_cost = true_positives * self.retention_cost_per_customer
        
        print(f"\nCost Analysis:")
        print(f"{'─' * 40}")
        print(f"Cost of False Positives (wasted retention): ${fp_cost:,.2f}")
        print(f"Cost of False Negatives (lost customers): ${fn_cost:,.2f}")
        print(f"Cost of Retention Campaigns (TP): ${tp_cost:,.2f}")
        print(f"Savings from Retained Customers (TP): ${tp_savings:,.2f}")
        
        # Total cost without model (all churners lost)
        total_churners = true_positives + false_negatives
        cost_without_model = total_churners * self.avg_customer_lifetime_value
        
        # Total cost with model
        cost_with_model = fp_cost + fn_cost + tp_cost - tp_savings
        
        # Net savings
        net_savings = cost_without_model - cost_with_model
        
        print(f"\nOverall Impact:")
        print(f"{'─' * 40}")
        print(f"Cost WITHOUT Model (all churners lost): ${cost_without_model:,.2f}")
        print(f"Cost WITH Model: ${cost_with_model:,.2f}")
        print(f"NET SAVINGS: ${net_savings:,.2f}")
        print(f"ROI: {(net_savings / (tp_cost + fp_cost)) * 100:.2f}%" if (tp_cost + fp_cost) > 0 else "N/A")
        
        # Calculate per-customer metrics
        total_customers = len(self.y_test)
        savings_per_customer = net_savings / total_customers
        
        print(f"\nPer-Customer Metrics:")
        print(f"{'─' * 40}")
        print(f"Total Customers Analyzed: {total_customers}")
        print(f"Savings per Customer: ${savings_per_customer:,.2f}")
        
        return {
            'fp_cost': fp_cost,
            'fn_cost': fn_cost,
            'tp_cost': tp_cost,
            'tp_savings': tp_savings,
            'cost_without_model': cost_without_model,
            'cost_with_model': cost_with_model,
            'net_savings': net_savings,
            'total_customers': total_customers,
            'savings_per_customer': savings_per_customer,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'true_negatives': true_negatives,
            'retained_customers': retained_customers
        }
    
    def visualize_business_impact(self, impact_results):
        """Create visualizations for business impact"""
        print(f"\n{'=' * 60}")
        print("GENERATING BUSINESS IMPACT VISUALIZATIONS")
        print('=' * 60)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Cost Comparison
        cost_data = pd.DataFrame({
            'Scenario': ['Without Model', 'With Model'],
            'Cost': [impact_results['cost_without_model'], impact_results['cost_with_model']]
        })
        
        bars = axes[0, 0].bar(cost_data['Scenario'], cost_data['Cost'], 
                              color=['#e74c3c', '#2ecc71'], width=0.6)
        axes[0, 0].set_title('Cost Comparison: With vs Without Model', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Total Cost ($)', fontsize=12)
        axes[0, 0].set_xlabel('Scenario', fontsize=12)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'${height:,.0f}',
                           ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Add savings annotation
        axes[0, 0].annotate(f'Savings: ${impact_results["net_savings"]:,.0f}',
                           xy=(0.5, impact_results['cost_with_model']),
                           xytext=(0.5, (impact_results['cost_without_model'] + impact_results['cost_with_model'])/2),
                           ha='center', fontsize=12, fontweight='bold', color='green',
                           arrowprops=dict(arrowstyle='<->', color='green', lw=2))
        
        # 2. Cost Breakdown
        cost_breakdown = pd.DataFrame({
            'Category': ['False Positives\n(Wasted Retention)', 
                        'False Negatives\n(Lost Customers)', 
                        'Retention Campaigns\n(True Positives)'],
            'Cost': [impact_results['fp_cost'], 
                    impact_results['fn_cost'], 
                    impact_results['tp_cost']]
        })
        
        colors = ['#f39c12', '#e74c3c', '#3498db']
        bars = axes[0, 1].bar(range(len(cost_breakdown)), cost_breakdown['Cost'], 
                              color=colors, width=0.6)
        axes[0, 1].set_title('Cost Breakdown by Category', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Cost ($)', fontsize=12)
        axes[0, 1].set_xticks(range(len(cost_breakdown)))
        axes[0, 1].set_xticklabels(cost_breakdown['Category'], fontsize=10)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'${height:,.0f}',
                           ha='center', va='bottom', fontsize=10)
        
        # 3. Customer Outcomes
        outcomes = pd.DataFrame({
            'Outcome': ['Retained\n(True Positives)', 'Missed\n(False Negatives)', 
                       'Correctly Identified\n(True Negatives)', 'False Alarms\n(False Positives)'],
            'Count': [impact_results['retained_customers'], 
                     impact_results['false_negatives'],
                     impact_results['true_negatives'],
                     impact_results['false_positives']]
        })
        
        colors = ['#2ecc71', '#e74c3c', '#95a5a6', '#f39c12']
        wedges, texts, autotexts = axes[1, 0].pie(outcomes['Count'], labels=outcomes['Outcome'],
                                                    autopct='%1.1f%%', colors=colors, startangle=90)
        axes[1, 0].set_title('Customer Outcomes Distribution', fontsize=14, fontweight='bold')
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        # 4. ROI and Savings Summary
        axes[1, 1].axis('off')
        
        summary_text = f"""
        BUSINESS IMPACT SUMMARY
        {'─' * 50}
        
        Total Customers Analyzed: {impact_results['total_customers']:,}
        
        Customers Correctly Identified as Churners: {impact_results['true_positives']}
        Customers Successfully Retained: {int(impact_results['retained_customers'])}
        
        Cost Without Model: ${impact_results['cost_without_model']:,.2f}
        Cost With Model: ${impact_results['cost_with_model']:,.2f}
        
        NET SAVINGS: ${impact_results['net_savings']:,.2f}
        
        Savings per Customer: ${impact_results['savings_per_customer']:,.2f}
        
        {'─' * 50}
        
        RECOMMENDATION:
        Implementing this churn prediction model can save
        approximately ${impact_results['net_savings']:,.2f} by enabling
        targeted retention campaigns for at-risk customers.
        """
        
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                       verticalalignment='center', bbox=dict(boxstyle='round', 
                       facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(f'{self.viz_dir}/business_impact_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {self.viz_dir}/business_impact_analysis.png")
        plt.close()
    
    def generate_recommendations(self, impact_results):
        """Generate actionable business recommendations"""
        print(f"\n{'=' * 60}")
        print("ACTIONABLE BUSINESS RECOMMENDATIONS")
        print('=' * 60)
        
        recommendations = []
        
        # Recommendation 1: Prioritize high-risk customers
        recommendations.append({
            'priority': 'HIGH',
            'recommendation': 'Implement Targeted Retention Campaigns',
            'details': f"Focus retention efforts on the {impact_results['true_positives']} customers "
                      f"identified as high-risk churners. This can potentially retain "
                      f"{int(impact_results['retained_customers'])} customers."
        })
        
        # Recommendation 2: Optimize retention budget
        if impact_results['fp_cost'] > impact_results['tp_cost'] * 0.3:
            recommendations.append({
                'priority': 'MEDIUM',
                'recommendation': 'Refine Model Threshold to Reduce False Positives',
                'details': f"Current false positive cost is ${impact_results['fp_cost']:,.2f}. "
                          f"Consider adjusting the prediction threshold to reduce wasted retention efforts."
            })
        
        # Recommendation 3: Focus on high-value customers
        recommendations.append({
            'priority': 'HIGH',
            'recommendation': 'Prioritize High-Value Customers',
            'details': f"With an average CLV of ${self.avg_customer_lifetime_value:,.2f}, focus retention "
                      f"campaigns on customers with higher-than-average lifetime value to maximize ROI."
        })
        
        # Recommendation 4: Improve retention strategies
        if self.retention_success_rate < 0.8:
            recommendations.append({
                'priority': 'MEDIUM',
                'recommendation': 'Improve Retention Campaign Effectiveness',
                'details': f"Current retention success rate is {self.retention_success_rate:.0%}. "
                          f"Improving this to 80% could increase savings significantly."
            })
        
        # Recommendation 5: Monitor key churn drivers
        recommendations.append({
            'priority': 'HIGH',
            'recommendation': 'Address Root Causes of Churn',
            'details': "Analyze feature importance results to identify and address the key factors "
                      "driving customer churn (e.g., contract type, tenure, service quality)."
        })
        
        # Print recommendations
        for idx, rec in enumerate(recommendations, 1):
            print(f"\n{idx}. [{rec['priority']}] {rec['recommendation']}")
            print(f"   {rec['details']}")
        
        return recommendations
    
    def run_full_analysis(self):
        """Execute complete business impact analysis"""
        print("\n" + "=" * 60)
        print("STARTING BUSINESS IMPACT ANALYSIS")
        print("=" * 60)
        
        impact_results = self.calculate_confusion_matrix_costs()
        self.visualize_business_impact(impact_results)
        recommendations = self.generate_recommendations(impact_results)
        
        print("\n" + "=" * 60)
        print("BUSINESS IMPACT ANALYSIS COMPLETED")
        print("=" * 60)
        
        return impact_results, recommendations


def analyze_business_impact(y_test, y_pred, y_pred_proba, model_name='Model',
                            clv=2000, cac=500, retention_cost=100, success_rate=0.7):
    """
    Convenience function to perform business impact analysis
    """
    analyzer = BusinessImpactAnalyzer(y_test, y_pred, y_pred_proba, model_name)
    analyzer.set_business_parameters(clv, cac, retention_cost, success_rate)
    return analyzer.run_full_analysis()


if __name__ == "__main__":
    # Test business impact analysis
    from data_preprocessing import load_and_clean_data
    from model_training import train_models
    
    X_train, X_test, y_train, y_test, feature_names = load_and_clean_data()
    xgb_model, rf_model = train_models(X_train, X_test, y_train, y_test, feature_names)
    
    y_pred = xgb_model.predict(X_test)
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    
    impact_results, recommendations = analyze_business_impact(
        y_test, y_pred, y_pred_proba, 'XGBoost'
    )
