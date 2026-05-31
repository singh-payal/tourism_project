
# for data manipulation
import pandas as pd
import numpy as np
# for data preprocessing and pipeline creation
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

# Set up MLflow tracking (using file-based URI for GitHub Actions)
mlflow.set_tracking_uri("file:///mlruns")

api = HfApi(token=os.getenv("HF_TOKEN")) # Initialize with token
repo = "singhpayal/tourism_project_dataset"

# Load the preprocessed data from Hugging Face
print("Loading preprocessed data...")
Xtrain = pd.read_csv(hf_hub_download(repo, "Xtrain.csv", repo_type="dataset"))
Xtest  = pd.read_csv(hf_hub_download(repo, "Xtest.csv", repo_type="dataset"))
ytrain = pd.read_csv(hf_hub_download(repo, "ytrain.csv", repo_type="dataset")).squeeze()
ytest  = pd.read_csv(hf_hub_download(repo, "ytest.csv", repo_type="dataset")).squeeze()



print(f"Training set shape: {Xtrain.shape}")
print(f"Test set shape: {Xtest.shape}")

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer

# Automatically identify column types
numeric_features = Xtrain.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = Xtrain.select_dtypes(
    include=["object"]
).columns.tolist()

print("Numeric Features:", numeric_features)
print("Categorical Features:", categorical_features)

# Preprocessor
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
    remainder="passthrough"
)

# Define base XGBoost Classifier
xgb_model = xgb.XGBClassifier(
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss',
    use_label_encoder=False
)

# Hyperparameter grid for classification
param_grid = {
    'xgbclassifier__n_estimators': [100, 200],
    'xgbclassifier__max_depth': [3, 5],
    'xgbclassifier__learning_rate': [0.05, 0.1]
}

# Pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

print("
Starting MLflow experiment...")
with mlflow.start_run():
    print("Performing Grid Search with Cross-Validation...")
    # Grid Search
    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=3,
        n_jobs=-1,
        scoring='roc_auc',
        verbose=1
    )
    grid_search.fit(Xtrain, ytrain)

    # Log parameter sets
    results = grid_search.cv_results_
    print(f"
Evaluated {len(results['params'])} parameter combinations")

    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_roc_auc", mean_score)

    # Best model
    print(f"
Best parameters: {grid_search.best_params_}")
    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Predictions
    print("
Making predictions...")
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    # Probability predictions for ROC-AUC
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]

    # Calculate metrics
    print("
Calculating metrics...")
    train_accuracy = accuracy_score(ytrain, y_pred_train)
    test_accuracy = accuracy_score(ytest, y_pred_test)

    train_precision = precision_score(ytrain, y_pred_train, zero_division=0)
    test_precision = precision_score(ytest, y_pred_test, zero_division=0)

    train_recall = recall_score(ytrain, y_pred_train, zero_division=0)
    test_recall = recall_score(ytest, y_pred_test, zero_division=0)

    train_f1 = f1_score(ytrain, y_pred_train, zero_division=0)
    test_f1 = f1_score(ytest, y_pred_test, zero_division=0)

    train_roc_auc = roc_auc_score(ytrain, y_pred_train_proba)
    test_roc_auc = roc_auc_score(ytest, y_pred_test_proba)

    # Log metrics
    mlflow.log_metrics({
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "train_precision": train_precision,
        "test_precision": test_precision,
        "train_recall": train_recall,
        "test_recall": test_recall,
        "train_f1_score": train_f1,
        "test_f1_score": test_f1,
        "train_roc_auc": train_roc_auc,
        "test_roc_auc": test_roc_auc
    })

    # Print results
    print("
" + "="*50)
    print("MODEL PERFORMANCE METRICS")
    print("="*50)
    print(f"Train Accuracy: {train_accuracy:.4f} | Test Accuracy: {test_accuracy:.4f}")
    print(f"Train Precision: {train_precision:.4f} | Test Precision: {test_precision:.4f}")
    print(f"Train Recall: {train_recall:.4f} | Test Recall: {test_recall:.4f}")
    print(f"Train F1-Score: {train_f1:.4f} | Test F1-Score: {test_f1:.4f}")
    print(f"Train ROC-AUC: {train_roc_auc:.4f} | Test ROC-AUC: {test_roc_auc:.4f}")
    print("="*50)

    print("
Test Set Classification Report:")
    print(classification_report(ytest, y_pred_test, target_names=['No Purchase', 'Purchase']))

    print("
Test Set Confusion Matrix:")
    print(confusion_matrix(ytest, y_pred_test))

    # Save the model locally
    model_path = "best_tourism_model_v1.joblib"
    joblib.dump(best_model, model_path)
    print(f"
Model saved locally as: {model_path}")

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model logged to MLflow")

    # Upload to Hugging Face
    repo_id = "singhpayal/tourism_prediction_model"
    repo_type = "model"

    # Step 1: Check if the repository exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"
Repository '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"
Repository '{repo_id}' not found. Creating new repository...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Repository '{repo_id}' created.")

    # Upload model to Hugging Face
    api.upload_file(
        path_or_fileobj="best_tourism_model_v1.joblib",
        path_in_repo="best_tourism_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print(f"Model uploaded to Hugging Face: {repo_id}")

print("
" + "="*50)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("="*50)
