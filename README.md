# Salary Prediction ML Project

A production-ready machine learning project for predicting salaries based on employee characteristics.

## 📋 Project Overview

This project implements a complete end-to-end machine learning pipeline for salary prediction using employee data. It includes data loading, exploratory data analysis, preprocessing, multiple model training, evaluation, and model deployment capabilities.

## 🎯 Features

- **Comprehensive Data Analysis**: Statistical analysis, missing value detection, correlation analysis
- **Advanced Preprocessing**: Automatic handling of categorical and numerical features
- **Multiple Models**: Linear Regression (baseline), Random Forest, Gradient Boosting, and Voting Regressor
- **Performance Metrics**: R², RMSE, and MAE for comprehensive evaluation
- **Feature Importance**: Visualization and analysis of the most important features
- **Model Persistence**: Best model saved for production deployment

## 📊 Dataset

- **Target Variable**: `salary` (continuous)
- **Features**:
  - `education_level`: Education qualification (High School, Bachelor, Master, PhD)
  - `years_experience`: Years of professional experience
  - `location`: Work location (Urban, Suburban, Rural)
  - `job_title`: Job position (Manager, Director, Analyst, Engineer)
  - `age`: Employee age
  - `gender`: Gender (Male, Female)

**Dataset Statistics**:
- Total samples: 1,000
- No missing values
- Salary range: $33,510 - $193,016
- Mean salary: $105,558

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup



1. **Create virtual environment** (optional but recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Linux/Mac
   # .venv\Scripts\activate  # On Windows
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📦 Dependencies

- pandas >= 1.5.0
- numpy >= 1.23.0
- scikit-learn >= 1.2.0
- matplotlib >= 3.6.0
- seaborn >= 0.12.0
- joblib >= 1.2.0

## 🏃‍♂️ Usage

### Run the Complete Pipeline

Simply execute the main script:

```bash
python main.py
```

This will:
1. Load and inspect the dataset
2. Perform exploratory data analysis
3. Preprocess the data
4. Split into training and test sets
5. Train multiple models
6. Compare model performance
7. Save the best model
8. Generate feature importance plot

### Use the Trained Model

```python
import joblib
import pandas as pd

# Load the saved model
model = joblib.load('best_model.pkl')

# Prepare new data (same format as training data)
new_data = pd.DataFrame({
    'education_level': ['PhD'],
    'years_experience': [10],
    'location': ['Urban'],
    'job_title': ['Director'],
    'age': [35],
    'gender': ['Female']
})

# Make prediction
predicted_salary = model.predict(new_data)
print(f"Predicted Salary: ${predicted_salary[0]:,.2f}")
```

## 📈 Model Performance

| Model | Train R² | Test R² | RMSE ($) | MAE ($) |
|-------|----------|---------|----------|---------|
| **Linear Regression** | 0.8790 | **0.8702** | **10,295** | **8,158** |
| Voting Regressor | 0.9413 | 0.8602 | 10,683 | 8,566 |
| Random Forest | 0.9487 | 0.8465 | 11,196 | 9,111 |
| Gradient Boosting | 0.9661 | 0.8369 | 11,542 | 9,089 |

**Best Model**: Linear Regression (87% accuracy on test set)

## 📊 Feature Importance

Top features influencing salary predictions:

1. **Education Level (PhD)**: 38.25%
2. **Education Level (Master)**: 18.42%
3. **Years of Experience**: 14.06%
4. **Job Title (Director)**: 8.39%
5. **Education Level (High School)**: 5.23%

See `feature_importance.png` for the complete visualization.

## 📁 Project Structure

```
ibm project/
├── data/
│   └── salary_prediction_data.csv  # Dataset
├── main.py                          # Main pipeline script
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── best_model.pkl                   # Saved best model
├── preprocessor.pkl                 # Saved preprocessor
└── feature_importance.png           # Feature importance plot
```

## 🔧 Code Structure

The `main.py` file is organized into modular functions:

- `load_and_inspect_data()`: Data loading and initial inspection
- `perform_eda()`: Exploratory Data Analysis
- `preprocess_data()`: Data preprocessing and transformation
- `split_data()`: Train-test split
- `train_baseline_model()`: Linear Regression baseline
- `train_advanced_models()`: Advanced models (RF, GB, Voting)
- `compare_models()`: Performance comparison
- `save_best_model()`: Model persistence
- `plot_feature_importance()`: Feature importance visualization
- `main()`: Pipeline orchestration

## 🎯 Key Insights

1. **Education is the strongest predictor**: PhD holders earn significantly more
2. **Experience matters**: Strong positive correlation with salary (0.34)
3. **Job title impact**: Directors earn the most on average
4. **Age has minimal impact**: Weak correlation (-0.05)
5. **Model performance**: 87% accuracy indicates reliable predictions

## 🚀 Production Deployment

The saved model (`best_model.pkl`) can be deployed to:
- REST APIs (Flask, FastAPI)
- Cloud platforms (AWS, Azure, GCP)
- Containerized environments (Docker)
- Web applications
- Mobile applications



## 🤝 Contributing

This is a demonstration project. For improvements or issues, please feel free to modify and extend.
