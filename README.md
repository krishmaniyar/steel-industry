# Steel Industry Load Type Prediction

A comprehensive end-to-end machine learning pipeline for predicting steel industry load types. This project implements a complete ML workflow from data ingestion to model deployment, achieving **99.97% accuracy** on test data using gradient boosting algorithms.

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline Architecture](#pipeline-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Feature Engineering](#feature-engineering)
- [Model Comparison](#model-comparison)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🎯 Overview

This project predicts three types of load conditions in steel industry operations:
- **Light_Load** (51.58% of data)
- **Medium_Load** (27.67% of data)
- **Maximum_Load** (20.75% of data)

The pipeline processes time-series energy consumption data with 11 original features and transforms them into 36 engineered features for optimal model performance.

## 📊 Problem Statement

Steel industry operations require accurate prediction of load types to optimize energy consumption, reduce costs, and improve operational efficiency. This project develops a machine learning classification system that can predict load types based on various energy consumption metrics, power factors, and temporal features.

## 📁 Dataset

The dataset (`Steel_industry_data.csv`) contains **35,040 records** with the following features:

### Original Features
- `date`: Timestamp of the measurement
- `Usage_kWh`: Energy usage in kilowatt-hours
- `Lagging_Current_Reactive.Power_kVarh`: Lagging reactive power
- `Leading_Current_Reactive_Power_kVarh`: Leading reactive power
- `CO2(tCO2)`: Carbon dioxide emissions
- `Lagging_Current_Power_Factor`: Lagging power factor
- `Leading_Current_Power_Factor`: Leading power factor
- `NSM`: Number of seconds from midnight
- `WeekStatus`: Weekend or Weekday
- `Day_of_week`: Day of the week
- `Load_Type`: Target variable (Light_Load, Medium_Load, Maximum_Load)

## 📂 Project Structure

```
steel-industry/
├── 01_data_ingestion.ipynb          # Data loading and initial cleaning
├── 02_data_validation.ipynb         # Data quality validation
├── 03_data_transformation.ipynb     # Feature engineering and encoding
├── 04_model_trainer.ipynb           # Model training (notebook version)
├── model_training_script.py         # Model training (script version)
├── debug_data.py                    # Data diagnostic tool
├── requirements.txt                  # Full dependencies
├── requirements-minimal.txt          # Minimal dependencies
├── README.md                         # This file
│
├── Data Files (Generated):
├── Steel_industry_data.csv           # Raw dataset (input)
├── ingested_data.csv                 # Output from ingestion
├── validated_data.csv                # Output from validation
├── transformed_data.csv              # Output from transformation
│
└── Model Artifacts (Generated):
    ├── encoders.pkl                  # Categorical encoders
    ├── feature_names.pkl             # Feature names list
    ├── transformation_metadata.pkl    # Transformation metadata
    ├── best_model_lightgbm.pkl       # Best trained model
    ├── model_metadata.pkl            # Model metadata
    ├── feature_importance.csv        # Feature importance scores
    └── comprehensive_results.pkl          # Training results
```

## 🔄 Pipeline Architecture

The project follows a modular 4-stage pipeline:

### Stage 1: Data Ingestion (`01_data_ingestion.ipynb`)
- Loads raw CSV data
- Performs initial data exploration
- Basic data cleaning and preprocessing
- Extracts temporal features (year, month, day, hour, minute)
- Saves cleaned data as `ingested_data.csv`

**Output:** `ingested_data.csv` (35,040 rows × 16 columns)

### Stage 2: Data Validation (`02_data_validation.ipynb`)
- Validates data quality and integrity
- Checks for missing values, outliers, and anomalies
- Performs statistical analysis
- Validates data types and ranges
- Saves validated data as `validated_data.csv`

**Output:** `validated_data.csv` (35,040 rows × 16 columns)

### Stage 3: Data Transformation (`03_data_transformation.ipynb`)
- **Feature Engineering:**
  - Temporal features: day_of_year, week_of_year, quarter
  - Cyclical encoding: hour_sin, hour_cos, month_sin, month_cos
  - Derived features: Total_Reactive_Power, Power_Factor_Difference, Power_Efficiency
  - Lag features: Usage_kWh_lag1, Usage_kWh_lag2
  - Rolling statistics: Usage_kWh_rolling_mean_3, Usage_kWh_rolling_std_3
  - Usage intensity bins: Low, Medium, High
- **Categorical Encoding:**
  - One-hot encoding for Day_of_week
  - Label encoding for WeekStatus
- **Target Encoding:**
  - Encodes Load_Type to numeric target (0: Light_Load, 1: Maximum_Load, 2: Medium_Load)
- **Scaling:**
  - Standard scaling applied to numeric features
- Saves transformed data and artifacts

**Output:** 
- `transformed_data.csv` (35,040 rows × 38 columns)
- `encoders.pkl`, `feature_names.pkl`, `transformation_metadata.pkl`

### Stage 4: Model Training (`04_model_trainer.ipynb` or `model_training_script.py`)
- Loads transformed data
- Splits data: 64% train, 16% validation, 20% test (stratified)
- Trains 12 different ML models
- Evaluates models using cross-validation
- Selects best model based on validation accuracy
- Evaluates best model on test set
- Saves best model and metadata

**Output:**
- `best_model_<name>.pkl`
- `model_metadata.pkl`
- `feature_importance.csv` (if applicable)

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd steel-industry
```

2. **Create a virtual environment (recommended):**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**

For minimal installation (core functionality):
```bash
pip install --upgrade pip
pip install -r requirements-minimal.txt
```

For full installation (includes advanced ML libraries and visualization tools):
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** The full installation includes optional libraries like XGBoost, LightGBM, SHAP, and Optuna for enhanced model performance and interpretability.

## 💻 Usage

### Quick Start

1. **Run the pipeline notebooks in order:**

```bash
# Start Jupyter Notebook
jupyter notebook

# Then run in order:
# 1. 01_data_ingestion.ipynb
# 2. 02_data_validation.ipynb
# 3. 03_data_transformation.ipynb
```

2. **Verify data transformation (optional):**
```bash
python debug_data.py
```

3. **Train models:**

**Option A: Using the Python script (recommended for automation):**
```bash
python model_training_script.py
```

**Option B: Using the Jupyter notebook (recommended for exploration):**
```bash
jupyter notebook 04_model_trainer.ipynb
```

### Using the Trained Model

```python
import joblib
import pandas as pd
import numpy as np

# Load the trained model
model = joblib.load('best_model_lightgbm.pkl')
metadata = joblib.load('model_metadata.pkl')

# Load your transformed data (must have same features as training)
# For new data, apply the same transformations from 03_data_transformation.ipynb
X_new = pd.read_csv('transformed_data.csv')[metadata['feature_names']]

# Make predictions
predictions = model.predict(X_new)
prediction_proba = model.predict_proba(X_new)

# Map predictions back to class names
class_mapping = {0: 'Light_Load', 1: 'Maximum_Load', 2: 'Medium_Load'}
predicted_classes = [class_mapping[pred] for pred in predictions]

print(f"Predictions: {predicted_classes[:5]}")
print(f"Confidence scores: {prediction_proba[:5]}")
```

## 📈 Model Performance

### Best Model: LightGBM

**Test Set Performance:**
- **Accuracy:** 99.97%
- **Precision:** 99.97%
- **Recall:** 99.97%
- **F1-Score:** 99.97%

**Per-Class Performance:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Light_Load | 100.00% | 99.94% | 99.97% | 3,615 |
| Maximum_Load | 99.86% | 100.00% | 99.93% | 1,454 |
| Medium_Load | 100.00% | 100.00% | 100.00% | 1,939 |

## 🔬 Feature Engineering

The transformation pipeline creates **36 engineered features** from the original 11 features:

### Temporal Features
- `year`, `month`, `day`, `hour`, `minute`
- `day_of_year`, `week_of_year`, `quarter`
- `hour_sin`, `hour_cos` (cyclical encoding)
- `month_sin`, `month_cos` (cyclical encoding)

### Derived Features
- `Total_Reactive_Power`: Sum of lagging and leading reactive power
- `Power_Factor_Difference`: Difference between lagging and leading power factors
- `Power_Efficiency`: Calculated efficiency metric

### Lag Features
- `Usage_kWh_lag1`: Previous hour's usage
- `Usage_kWh_lag2`: Two hours ago usage

### Rolling Statistics
- `Usage_kWh_rolling_mean_3`: 3-hour rolling mean
- `Usage_kWh_rolling_std_3`: 3-hour rolling standard deviation

### Categorical Encodings
- `WeekStatus_encoded`: Binary encoding (Weekend/Weekday)
- `Day_of_week_*`: One-hot encoded day features (Monday through Sunday)
- `Usage_Intensity_*`: Binned usage intensity (Low, Medium, High)

## 📊 Model Comparison

The following table compares all 12 models trained on the validation set:

| Rank | Model | Accuracy | Precision | Recall | F1-Score | CV Mean | CV Std | Training Time (s) |
|------|-------|----------|-----------|--------|----------|---------|--------|-------------------|
| 🥇 | **LightGBM** | **0.9996** | **0.9996** | **0.9996** | **0.9996** | **0.9991** | 0.0003 | 1.56 |
| 🥈 | **XGBoost** | **0.9996** | **0.9996** | **0.9996** | **0.9996** | 0.9985 | 0.0003 | 3.40 |
| 🥉 | Random Forest | 0.9929 | 0.9929 | 0.9929 | 0.9929 | 0.9913 | 0.0011 | 1.64 |
| 4 | Decision Tree | 0.9907 | 0.9907 | 0.9907 | 0.9907 | 0.9895 | 0.0012 | 0.73 |
| 5 | Extra Trees | 0.9897 | 0.9896 | 0.9897 | 0.9896 | 0.9874 | 0.0019 | 1.09 |
| 6 | Gradient Boosting | 0.9888 | 0.9888 | 0.9888 | 0.9888 | 0.9859 | 0.0021 | 100.71 |
| 7 | Neural Network | 0.9838 | 0.9838 | 0.9838 | 0.9838 | 0.9819 | 0.0018 | 74.44 |
| 8 | SVM | 0.9121 | 0.9123 | 0.9121 | 0.9113 | 0.9045 | 0.0034 | 3.21 |
| 9 | K-Nearest Neighbors | 0.8901 | 0.8899 | 0.8901 | 0.8898 | 0.8858 | 0.0031 | 0.89 |
| 10 | Logistic Regression | 0.8334 | 0.8320 | 0.8334 | 0.8327 | 0.8297 | 0.0032 | 3.21 |
| 11 | AdaBoost | 0.7778 | 0.7752 | 0.7778 | 0.7646 | 0.7746 | 0.0045 | 0.78 |
| 12 | Naive Bayes | 0.7421 | 0.7768 | 0.7421 | 0.7201 | 0.7327 | 0.0052 | 0.74 |

### Key Observations

1. **Gradient Boosting Dominance:** LightGBM and XGBoost achieve near-perfect performance (99.96% accuracy)
2. **Ensemble Methods Excel:** Tree-based ensemble methods (Random Forest, Extra Trees, Gradient Boosting) all perform above 98%
3. **Speed vs. Performance:** LightGBM offers the best balance with 99.96% accuracy in just 1.56 seconds
4. **Linear Models Struggle:** Logistic Regression achieves only 83.34% accuracy, indicating non-linear relationships
5. **Neural Networks:** MLP achieves 98.38% but requires longer training time (74.44s)

## 📉 Results

### Model Selection
The **LightGBM** model was selected as the best model based on:
- Highest validation accuracy (99.96%)
- Highest cross-validation score (99.91% ± 0.03%)
- Fastest training time among top performers (1.56s)
- Excellent generalization on test set (99.97%)

### Feature Importance
Top 10 most important features (LightGBM):
1. Usage_kWh
2. Lagging_Current_Reactive.Power_kVarh
3. Leading_Current_Reactive_Power_kVarh
4. CO2(tCO2)
5. NSM
6. Power_Factor_Difference
7. Total_Reactive_Power
8. hour
9. Usage_kWh_rolling_mean_3
10. Leading_Current_Power_Factor

### Model Artifacts
After training, the following files are generated:
- `best_model_lightgbm.pkl`: Serialized LightGBM model
- `model_metadata.pkl`: Model metadata including:
  - Model parameters
  - Feature names
  - Training date
  - Performance metrics
  - Target class mapping
- `feature_importance.csv`: Feature importance scores (if available)

## 🔧 Troubleshooting

### Common Issues

1. **FileNotFoundError: transformed_data.csv not found**
   - **Solution:** Run notebooks 01, 02, and 03 in sequence before training

2. **String columns detected in features**
   - **Solution:** Re-run `03_data_transformation.ipynb` and ensure all categorical columns are properly encoded

3. **XGBoost/LightGBM not available**
   - **Solution:** Install using `pip install xgboost lightgbm` or use `requirements.txt`

4. **Memory errors during training**
   - **Solution:** Reduce batch size or use a machine with more RAM

5. **Inconsistent results**
   - **Solution:** Ensure random seeds are set (already configured in code with `random_state=42`)

### Data Validation

Use the diagnostic script to check data quality:
```bash
python debug_data.py
```

This will check for:
- Missing target column
- String columns in features
- Missing values
- Infinite values

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Dataset: Steel Industry Energy Consumption Data
- Libraries: pandas, scikit-learn, XGBoost, LightGBM, Jupyter
- Community: Open source contributors and the ML community

## 📧 Contact

For questions or issues, please open an issue on the GitHub repository.

---

**Last Updated:** 2025
**Version:** 1.0.0
**Status:** Production Ready ✅
