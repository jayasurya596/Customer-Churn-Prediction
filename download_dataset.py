"""
Script to download the Telco Customer Churn dataset
"""

import os
import urllib.request
import pandas as pd

def download_dataset():
    """Download the Telco Customer Churn dataset"""
    
    print("=" * 60)
    print("DOWNLOADING TELCO CUSTOMER CHURN DATASET")
    print("=" * 60)
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Dataset URL (from a public source)
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    output_path = "data/telecom_churn.csv"
    
    try:
        print(f"\nDownloading dataset from:\n{url}")
        print(f"\nSaving to: {output_path}")
        
        # Download the file
        urllib.request.urlretrieve(url, output_path)
        
        print("\n✅ Dataset downloaded successfully!")
        
        # Verify the download
        df = pd.read_csv(output_path)
        print(f"\nDataset Info:")
        print(f"  - Rows: {df.shape[0]}")
        print(f"  - Columns: {df.shape[1]}")
        print(f"  - Churn column present: {'Churn' in df.columns}")
        
        print("\nFirst few rows:")
        print(df.head())
        
        print("\n" + "=" * 60)
        print("DATASET READY FOR USE!")
        print("=" * 60)
        print("\nYou can now run: python main.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {str(e)}")
        print("\nAlternative: Download manually from:")
        print("https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
        print("\nPlace the file in the 'data/' folder and rename it to 'telecom_churn.csv'")
        return False

if __name__ == "__main__":
    download_dataset()
