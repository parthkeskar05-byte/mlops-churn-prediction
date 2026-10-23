import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings('ignore')

print("📥 Downloading Telco Churn dataset...")
# Load real-world dataset directly from GitHub
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

print("🧹 Cleaning and preprocessing data...")
# Drop customer ID (not useful for prediction)
df.drop('customerID', axis=1, inplace=True)

# Convert TotalCharges to numeric, dropping the few missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# Convert categorical target 'Churn' to binary (Yes: 1, No: 0)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# One-hot encode the rest of the categorical variables
X = pd.get_dummies(df.drop('Churn', axis=1), drop_first=True)
y = df['Churn']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🚀 Starting MLflow experiment and training XGBoost...")
# Set up MLOps tracking
mlflow.set_experiment("Customer_Churn_Prediction")

with mlflow.start_run():
    # Define model parameters
    params = {
        "objective": "binary:logistic",
        "max_depth": 5,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "eval_metric": "logloss"
    }
    
    # Log parameters to MLflow
    mlflow.log_params(params)
    
    # Train the model
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    # Make predictions and evaluate
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    
    # Log metrics to MLflow
    mlflow.log_metric("accuracy", acc)
    print(f"✅ Model trained successfully! Accuracy: {acc:.4f}")
    
    # Save the model locally for our API to use later
    model.save_model("churn_model.json")
    print("💾 Model saved as 'churn_model.json'")
    
    # Save the feature columns so our API knows what shape to expect
    pd.Series(X.columns).to_json("expected_features.json", orient='records')

print("🎉 Phase 2 Complete!")