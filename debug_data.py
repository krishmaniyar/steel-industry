#!/usr/bin/env python3
"""
Debug script to diagnose issues with the transformed data
"""

import pandas as pd
import numpy as np
import joblib

def diagnose_data():
    print("=" * 60)
    print("DATA TRANSFORMATION DIAGNOSIS")
    print("=" * 60)
    
    try:
        print("\n1. Loading data...")
        df = pd.read_csv('transformed_data.csv')
        print(f"Data loaded: {df.shape}")
        
        try:
            feature_names = joblib.load('feature_names.pkl')
            print(f"Feature names loaded: {len(feature_names)} features")
        except:
            print("Could not load feature_names.pkl, using all columns except 'target'")
            feature_names = [col for col in df.columns if col != 'target']
        
        print("\n2. Checking target variable...")
        if 'target' in df.columns:
            target = df['target']
            print(f"Target column found: {target.name}")
            print(f"  Data type: {target.dtype}")
            print(f"  Unique values: {sorted(target.unique())}")
            print(f"  Value counts: {target.value_counts().to_dict()}")
        else:
            print("No 'target' column found!")
            print(f"Available columns: {list(df.columns)}")
        
        print("\n3. Checking features...")
        X = df[feature_names]
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Feature data types:")
        dtype_counts = X.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"  {dtype}: {count} columns")
        
        string_cols = X.select_dtypes(include=['object']).columns.tolist()
        if string_cols:
            print(f"\nISSUE FOUND: String columns in features!")
            print(f"String columns: {string_cols}")
            
            for col in string_cols[:3]:
                print(f"\n{col} sample values:")
                print(X[col].value_counts().head())
        else:
            print("\nNo string columns found in features")
        
        print("\n4. Checking for missing values...")
        missing_features = X.isnull().sum()
        if missing_features.sum() > 0:
            print("Missing values found:")
            print(missing_features[missing_features > 0])
        else:
            print("No missing values in features")
        
        print("\n5. Checking for infinite values...")
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            inf_count = np.isinf(X[numeric_cols]).sum().sum()
            if inf_count > 0:
                print(f"Found {inf_count} infinite values")
                inf_cols = X[numeric_cols].columns[np.isinf(X[numeric_cols]).any()]
                print(f"Columns with infinite values: {inf_cols.tolist()}")
            else:
                print("No infinite values found")
        
        print("\n" + "=" * 60)
        print("DIAGNOSIS SUMMARY")
        print("=" * 60)
        
        issues = []
        if 'target' not in df.columns:
            issues.append("Missing 'target' column")
        
        if string_cols:
            issues.append(f"String columns in features: {string_cols}")
        
        if missing_features.sum() > 0:
            issues.append("Missing values in features")
        
        if len(numeric_cols) > 0 and np.isinf(X[numeric_cols]).sum().sum() > 0:
            issues.append("Infinite values in features")
        
        if issues:
            print("ISSUES FOUND:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")
            
            print("\nRECOMMENDED FIXES:")
            if string_cols:
                print("  - Re-run the data transformation notebook (03_data_transformation.ipynb)")
                print("  - Check categorical encoding section")
            
        else:
            print("No issues found! Data should be ready for modeling.")
        
        return issues
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Make sure you've run the previous notebooks in order:")
        print("  1. 01_data_ingestion.ipynb")
        print("  2. 02_data_validation.ipynb") 
        print("  3. 03_data_transformation.ipynb")
        return ["Missing data files"]
    
    except Exception as e:
        print(f"Error: {e}")
        return [f"Error: {e}"]

if __name__ == "__main__":
    issues = diagnose_data()
    
    if issues:
        print(f"\nFound {len(issues)} issue(s). Please fix before training models.")
        exit(1)
    else:
        print("\nData looks good! You can proceed with model training.")
        exit(0)
