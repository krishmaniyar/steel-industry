#!/usr/bin/env python3
"""
Model Training Script for Steel Industry Load Type Prediction
This script properly handles the Load_Type column issue and trains multiple ML models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

def load_and_prepare_data():
    """Load and prepare data for model training"""
    print("=== LOADING AND PREPARING DATA ===")
    
    try:
        df = pd.read_csv('transformed_data.csv')
        print(f"Data loaded successfully: {df.shape}")
    except FileNotFoundError:
        print("Error: transformed_data.csv not found. Please run 03_data_transformation.ipynb first.")
        raise
    
    try:
        encoders = joblib.load('encoders.pkl')
        transformation_metadata = joblib.load('transformation_metadata.pkl')
        print("Additional artifacts loaded successfully")
        print(f"Target classes: {transformation_metadata['target_classes']}")
    except Exception as e:
        print(f"Warning: Could not load additional artifacts: {e}")
        transformation_metadata = {}
    
    print("\n=== FIXING FEATURE SELECTION ===")
    
    numeric_columns = df.select_dtypes(include=[np.number, bool]).columns.tolist()
    print(f"Numeric and boolean columns available: {len(numeric_columns)}")
    
    features_to_exclude = ['target', 'Load_Type']
    feature_columns = [col for col in numeric_columns if col not in features_to_exclude]
    
    print(f"Features to exclude: {features_to_exclude}")
    print(f"Features to use: {len(feature_columns)}")
    
    features_df = df[feature_columns]
    string_columns = features_df.select_dtypes(include=['object']).columns
    if len(string_columns) > 0:
        print(f"ERROR: String columns still present: {list(string_columns)}")
        return None, None, None, None
    
    print("All features are numeric/boolean")
    
    X = df[feature_columns].copy()
    y = df['target'].copy()
    
    print(f"Final feature set: {X.shape}")
    print(f"Target set: {y.shape}")
    print(f"Target classes: {sorted(y.unique())}")
    
    print(f"\nData quality check:")
    print(f"Missing values in features: {X.isnull().sum().sum()}")
    print(f"Missing values in target: {y.isnull().sum()}")
    
    return X, y, feature_columns, transformation_metadata

def split_data(X, y):
    """Split data into training, validation, and test sets"""
    print("\n=== DATA SPLITTING ===")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Training set: {X_train_split.shape[0]:,} samples")
    print(f"Validation set: {X_val.shape[0]:,} samples")
    print(f"Test set: {X_test.shape[0]:,} samples")
    
    return X_train_split, X_val, X_test, y_train_split, y_val, y_test

def train_models(X_train, X_val, y_train, y_val):
    """Train multiple models and return results"""
    print("\n=== TRAINING MODELS ===")
    
    models = {
        'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Extra Trees': ExtraTreesClassifier(random_state=42, n_jobs=-1),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'SVM': SVC(random_state=42, probability=True),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'Neural Network': MLPClassifier(random_state=42, max_iter=500),
        'AdaBoost': AdaBoostClassifier(random_state=42)
    }
    
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')
        print("XGBoost added to model list")
    
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = lgb.LGBMClassifier(random_state=42, verbose=-1)
        print("LightGBM added to model list")
    
    print(f"\nTraining {len(models)} models...")
    
    model_results = []
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = datetime.now()
        
        try:
            model.fit(X_train, y_train)
            
            y_val_pred = model.predict(X_val)
            
            accuracy = accuracy_score(y_val, y_val_pred)
            precision = precision_score(y_val, y_val_pred, average='weighted')
            recall = recall_score(y_val, y_val_pred, average='weighted')
            f1 = f1_score(y_val, y_val_pred, average='weighted')
            
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'Model': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1_Score': f1,
                'CV_Mean': cv_mean,
                'CV_Std': cv_std,
                'Training_Time': training_time
            }
            
            model_results.append(result)
            trained_models[name] = model
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  CV Score: {cv_mean:.4f} (±{cv_std:.4f})")
            print(f"  Training time: {training_time:.2f}s")
            
        except Exception as e:
            print(f"  Error training {name}: {e}")
            continue
    
    print(f"\nSuccessfully trained {len(trained_models)} models")
    return model_results, trained_models

def evaluate_best_model(best_model_name, trained_models, X_train, X_test, y_train, y_test, transformation_metadata):
    """Evaluate the best model on test set"""
    print(f"\n=== EVALUATING BEST MODEL: {best_model_name} ===")
    
    best_model = trained_models[best_model_name]
    best_model.fit(X_train, y_train)
    
    y_test_pred = best_model.predict(X_test)
    y_test_pred_proba = best_model.predict_proba(X_test)
    
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='weighted')
    test_recall = recall_score(y_test, y_test_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    print(f"FINAL MODEL PERFORMANCE ON TEST SET:")
    print(f"Model: {best_model_name}")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1-Score: {test_f1:.4f}")
    
    class_names = ['Light_Load', 'Maximum_Load', 'Medium_Load']
    try:
        target_classes = transformation_metadata['target_classes']
        class_names = [k for k, v in sorted(target_classes.items(), key=lambda x: x[1])]
    except:
        pass
    
    print(f"\nDETAILED CLASSIFICATION REPORT:")
    report = classification_report(y_test, y_test_pred, target_names=class_names)
    print(report)
    
    return best_model, y_test_pred, y_test_pred_proba, test_accuracy, test_precision, test_recall, test_f1, class_names

def save_model_and_results(best_model, best_model_name, feature_columns, test_accuracy, test_precision, test_recall, test_f1, transformation_metadata):
    """Save the best model and results"""
    print(f"\n=== SAVING MODEL AND RESULTS ===")
    
    try:
        model_filename = f'best_model_{best_model_name.replace(" ", "_").lower()}.pkl'
        joblib.dump(best_model, model_filename)
        print(f"Final model saved as: {model_filename}")
        
        model_metadata = {
            'model_name': best_model_name,
            'model_type': type(best_model).__name__,
            'training_date': datetime.now().isoformat(),
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1_score': test_f1,
            'feature_count': len(feature_columns),
            'feature_names': feature_columns,
            'target_classes': transformation_metadata.get('target_classes', {}),
            'model_parameters': best_model.get_params() if hasattr(best_model, 'get_params') else {}
        }
        
        joblib.dump(model_metadata, 'model_metadata.pkl')
        print("Model metadata saved as: model_metadata.pkl")
        
        return model_filename
        
    except Exception as e:
        print(f"Error saving model and results: {e}")
        return None

def main():
    """Main function to run the complete model training pipeline"""
    print("="*60)
    print("        STEEL INDUSTRY MODEL TRAINING")
    print("="*60)
    print(f"Started at: {datetime.now()}")
    print(f"XGBoost available: {XGBOOST_AVAILABLE}")
    print(f"LightGBM available: {LIGHTGBM_AVAILABLE}")
    
    try:
        X, y, feature_columns, transformation_metadata = load_and_prepare_data()
        if X is None:
            return
        
        X_train_split, X_val, X_test, y_train_split, y_val, y_test = split_data(X, y)
        
        model_results, trained_models = train_models(X_train_split, X_val, y_train_split, y_val)
        
        if len(model_results) == 0:
            print("No models were successfully trained!")
            return
        
        results_df = pd.DataFrame(model_results)
        results_df = results_df.sort_values('Accuracy', ascending=False)
        
        print(f"\n=== MODEL COMPARISON ===")
        print("Model Performance (sorted by Accuracy):")
        print(results_df.round(4))
        
        best_model_name = results_df.iloc[0]['Model']
        print(f"\nBest performing model: {best_model_name}")
        
        best_model, y_test_pred, y_test_pred_proba, test_accuracy, test_precision, test_recall, test_f1, class_names = evaluate_best_model(
            best_model_name, trained_models, X_train_split, X_test, y_train_split, y_test, transformation_metadata
        )
        
        model_filename = save_model_and_results(
            best_model, best_model_name, feature_columns, test_accuracy, test_precision, test_recall, test_f1, transformation_metadata
        )
        
        print(f"\n" + "="*60)
        print("        TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Best Model: {best_model_name}")
        print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"Model saved as: {model_filename}")
        print(f"Completed at: {datetime.now()}")
        
        if test_accuracy >= 0.9:
            readiness = "Excellent - Ready for Production"
        elif test_accuracy >= 0.8:
            readiness = "Good - Ready for Deployment"
        elif test_accuracy >= 0.7:
            readiness = "Acceptable - Consider Further Tuning"
        else:
            readiness = "Poor - Needs Significant Improvement"
        
        print(f"Status: {readiness}")
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
