# Predict-House-Price
Use of ML to predict House Price
Create and navigate to project directory
mkdir house-price-prediction
cd house-price-prediction

# Initialize git repository
git init

# Create initial README.md
echo "# House Price Prediction ML Workflow

This project implements a complete machine learning workflow to predict house prices using the Boston Housing dataset.

## Installation

Install the required dependencies:

\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Running the Models

### Decision Tree Regressor
\`\`\`bash
python train.py
\`\`\`

### Kernel Ridge Regressor
\`\`\`bash
python train2.py
\`\`\`

## Project Structure
- \`misc.py\`: Contains reusable functions for data loading, preprocessing, and model evaluation
- \`train.py\`: Trains and evaluates Decision Tree Regressor
- \`train2.py\`: Trains and evaluates Kernel Ridge Regressor
- \`requirements.txt\`: Project dependencies
- \`.github/workflows/\`: GitHub Actions configuration
" > README.md

# Create .gitignore
echo "__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
.env
*.egg-info/
.installed.cfg
*.egg
.DS_Store
*.log
.coverage
.pytest_cache/
.mypy_cache/
.ipynb_checkpoints/" > .gitignore

# Initial commit to main
git add .
git commit -m "Initial commit: Add README and project structure"
Branch: dtree
Now let's create the dtree branch with Decision Tree implementation:

bash
# Create and switch to dtree branch
git checkout -b dtree

# Create requirements.txt
echo "scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.1
seaborn==0.12.2" > requirements.txt

# Create misc.py with reusable functions
echo "import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_data():
    \"\"\"Load Boston Housing dataset\"\"\"
    data_url = \"http://lib.stat.cmu.edu/datasets/boston\"
    raw_df = pd.read_csv(data_url, sep=\"\\s+\", skiprows=22, header=None)
    
    # Split into data and target
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    
    # Feature names based on the original dataset
    feature_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target  # MEDV is our target variable
    return df

def preprocess_data(df, test_size=0.2, random_state=42):
    \"\"\"Preprocess data and split into train/test sets\"\"\"
    # Separate features and target
    X = df.drop('MEDV', axis=1)
    y = df['MEDV']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_model(model, X_train, y_train):
    \"\"\"Train a machine learning model\"\"\"
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, model_name=\"Model\"):
    \"\"\"Evaluate model performance and return metrics\"\"\"
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f\"{model_name} Performance:\")
    print(f\"Mean Squared Error (MSE): {mse:.4f}\")
    print(f\"R² Score: {r2:.4f}\")
    print(\"-\" * 50)
    
    return mse, r2, y_pred

def display_feature_importance(model, feature_names):
    \"\"\"Display feature importance for tree-based models\"\"\"
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(\"\\nFeature Importance:\")
        print(feature_imp)
        return feature_imp
    return None" > misc.py

# Create train.py for Decision Tree Regressor
echo "from sklearn.tree import DecisionTreeRegressor
from misc import load_data, preprocess_data, train_model, evaluate_model, display_feature_importance
import numpy as np

def main():
    # Load and preprocess data
    print(\"Loading Boston Housing dataset...\")
    df = load_data()
    print(f\"Dataset shape: {df.shape}\")
    print(f\"Features: {list(df.columns[:-1])}\")
    print(f\"Target variable: MEDV\")
    print(\"=\" * 60)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # Initialize and train Decision Tree model
    print(\"Training Decision Tree Regressor...\")
    dt_model = DecisionTreeRegressor(random_state=42, max_depth=6)
    dt_model = train_model(dt_model, X_train, y_train)
    
    # Evaluate model
    mse, r2, y_pred = evaluate_model(dt_model, X_test, y_test, \"Decision Tree Regressor\")
    
    # Display feature importance
    feature_names = df.columns[:-1]
    feature_imp = display_feature_importance(dt_model, feature_names)
    
    return mse, r2, dt_model

if __name__ == \"__main__\":
    mse, r2, model = main()" > train.py

# Commit dtree branch
git add .
git commit -m "Add Decision Tree implementation with reusable functions"
Merge dtree to main
bash
# Switch to main and merge dtree
git checkout main
git merge dtree -m "Merge dtree branch with Decision Tree implementation"
Branch: kernelridge
Now let's create the kernelridge branch with GitHub Actions:

bash
# Create and switch to kernelridge branch
git checkout -b kernelridge

# Create train2.py for Kernel Ridge Regressor
echo "from sklearn.kernel_ridge import KernelRidge
from misc import load_data, preprocess_data, train_model, evaluate_model
import numpy as np

def main():
    # Load and preprocess data
    print(\"Loading Boston Housing dataset...\")
    df = load_data()
    print(f\"Dataset shape: {df.shape}\")
    print(f\"Features: {list(df.columns[:-1])}\")
    print(f\"Target variable: MEDV\")
    print(\"=\" * 60)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # Initialize and train Kernel Ridge model
    print(\"Training Kernel Ridge Regressor...\")
    kr_model = KernelRridge(alpha=1.0, kernel='rbf', gamma=0.1)
    kr_model = train_model(kr_model, X_train, y_train)
    
    # Evaluate model
    mse, r2, y_pred = evaluate_model(kr_model, X_test, y_test, \"Kernel Ridge Regressor\")
    
    return mse, r2, kr_model

if __name__ == \"__main__\":
    mse, r2, model = main()" > train2.py

# Create GitHub Actions workflow directory and file
mkdir -p .github/workflows

echo "name: ML Pipeline CI

on:
  push:
    branches: [ kernelridge ]
  pull_request:
    branches: [ kernelridge ]

jobs:
  test-models:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        
    - name: Run Decision Tree Model
      run: |
        echo \"=== Testing Decision Tree Regressor ===\"
        python train.py
        echo \"Decision Tree test completed successfully\"
        
    - name: Run Kernel Ridge Model
      run: |
        echo \"=== Testing Kernel Ridge Regressor ===\"
        python train2.py
        echo \"Kernel Ridge test completed successfully\"
        
    - name: Verify imports
      run: |
        python -c \"
        from misc import load_data, preprocess_data, train_model, evaluate_model
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.kernel_ridge import KernelRidge
        import pandas as pd
        import numpy as np
        print('All imports successful!')\n
        df = load_data()\n
        print(f'Data loaded successfully: {df.shape}')\n
        \"" > .github/workflows/ml-pipeline.yml

# Commit kernelridge branch
git add .
git commit -m "Add Kernel Ridge implementation and GitHub Actions workflow"
Final Setup and 

Push
bash
# Switch back to main
git checkout main

# Create GitHub repository and push (assuming you've created repo on GitHub)
gh repo create house-price-prediction --public --push

# Push all branches
git push -u origin main
git push -u origin dtree
git push -u origin kernelridge
