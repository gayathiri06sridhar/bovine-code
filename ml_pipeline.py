import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (
    train_test_split, GroupShuffleSplit, StratifiedKFold, 
    StratifiedGroupKFold, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
import xgboost as xgb
import shap
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_pipeline():
    print("="*60)
    print("PHASE 3: PREPARE ML DATA")
    print("="*60)
    
    # 1. Load Data
    df = pd.read_csv("cow_milk_mastitis_dataset.csv")
    
    # 2. Define Features and Target
    feature_cols = [
        "Day", "Milk_Temperature", "Milk_pH", "Milk_Conductivity",
        "Somatic_Cell_Count", "Milk_Yield", "Clotting"
    ]
    target_col = "class1"
    
    X = df[feature_cols]
    y = df[target_col]
    groups = df["Cow_ID"]
    
    # Check if there are repeated cows
    has_repeated_cows = df["Cow_ID"].nunique() < len(df)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Has repeated cows: {has_repeated_cows}")
    
    # 3. Train / Test Split (80/20)
    if has_repeated_cows:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        print("Using GroupShuffleSplit to avoid target leakage across cows.")
    else:
        # train_test_split directly
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=42
        )
        # to get indices:
        train_idx = X_train.index
        test_idx = X_test.index
        print("Using Stratified split (no repeated cows).")
        
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    groups_train = groups.loc[train_idx]
    
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    
    # 4. CV Strategy
    if has_repeated_cows:
        cv_strategy = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
    print("\n"+"="*60)
    print("PHASE 4: TRAIN AND COMPARE MODELS")
    print("="*60)
    
    # Define models
    models = {
        "Logistic Regression": Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(class_weight='balanced', random_state=42))]),
        "Decision Tree": Pipeline([('clf', DecisionTreeClassifier(class_weight='balanced', random_state=42))]),
        "Random Forest": Pipeline([('clf', RandomForestClassifier(class_weight='balanced', random_state=42))]),
        "SVM": Pipeline([('scaler', StandardScaler()), ('clf', SVC(probability=True, class_weight='balanced', random_state=42))]),
        "KNN": Pipeline([('scaler', StandardScaler()), ('clf', KNeighborsClassifier())]),
        "XGBoost": Pipeline([('clf', xgb.XGBClassifier(eval_metric='logloss', random_state=42))]) # XGBoost handles imbalance via scale_pos_weight, we'll tune it
    }
    
    results = {}
    best_f1 = 0
    best_model_name = ""
    
    # Train and evaluate
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline[-1], "predict_proba") else None
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0
        
        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "Specificity": spec,
            "F1-Score": f1,
            "ROC-AUC": roc_auc
        }
        
        print(f"--- {name} ---")
        print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
        print(f"Specificity: {spec:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name

    print(f"\nStrongest Model based on F1-Score: {best_model_name}")

    print("\n"+"="*60)
    print("PHASE 5: TUNE THE STRONGEST MODEL")
    print("="*60)
    
    # We will tune Random Forest or XGBoost as they are typically the strongest
    # Let's do a generic setup for the best model found
    best_base_pipeline = models[best_model_name]
    
    param_grid = {}
    if best_model_name == "Random Forest":
        param_grid = {
            'clf__n_estimators': [50, 100, 200],
            'clf__max_depth': [None, 10, 20, 30],
            'clf__min_samples_split': [2, 5, 10]
        }
    elif best_model_name == "XGBoost":
        scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
        best_base_pipeline.set_params(clf__scale_pos_weight=scale_pos_weight)
        param_grid = {
            'clf__n_estimators': [50, 100, 200],
            'clf__max_depth': [3, 5, 7],
            'clf__learning_rate': [0.01, 0.1, 0.2]
        }
    elif best_model_name == "Logistic Regression":
        param_grid = {'clf__C': [0.01, 0.1, 1, 10, 100]}
    elif best_model_name == "Decision Tree":
        param_grid = {'clf__max_depth': [None, 5, 10, 20]}
    elif best_model_name == "SVM":
        param_grid = {'clf__C': [0.1, 1, 10], 'clf__gamma': ['scale', 'auto', 0.1, 1]}
    elif best_model_name == "KNN":
        param_grid = {'clf__n_neighbors': [3, 5, 7, 9], 'clf__weights': ['uniform', 'distance']}
        
    random_search = RandomizedSearchCV(
        estimator=best_base_pipeline,
        param_distributions=param_grid,
        n_iter=10,
        scoring='f1',
        cv=cv_strategy,
        random_state=42,
        n_jobs=-1
    )
    
    if has_repeated_cows:
        random_search.fit(X_train, y_train, groups=groups_train)
    else:
        random_search.fit(X_train, y_train)
        
    best_model = random_search.best_estimator_
    print(f"Best Parameters: {random_search.best_params_}")

    print("\n"+"="*60)
    print("PHASE 6: EVALUATE FINAL MODEL")
    print("="*60)
    
    y_pred_final = best_model.predict(X_test)
    y_prob_final = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model[-1], "predict_proba") else None
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_final))
    
    cm = confusion_matrix(y_test, y_pred_final)
    print("Confusion Matrix:")
    print(f"TN: {cm[0][0]} | FP: {cm[0][1]}")
    print(f"FN: {cm[1][0]} | TP: {cm[1][1]}")
    
    print("\nAnalysis of False Positives/Negatives:")
    print(f"- False Positives ({cm[0][1]}): Healthy cows incorrectly classified as having mastitis (leads to unnecessary checks/costs).")
    print(f"- False Negatives ({cm[1][0]}): Mastitis cases missed by the model (leads to delayed treatment and potential herd spread).")

    print("\n"+"="*60)
    print("PHASE 7: EXPLAINABLE AI (XAI)")
    print("="*60)
    
    print("Generating SHAP values...")
    clf = best_model.named_steps['clf']
    
    try:
        if best_model_name in ["Random Forest", "Decision Tree", "XGBoost"]:
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_test)
            # Create output dir if needed
            os.makedirs("output", exist_ok=True)
            
            plt.figure()
            if isinstance(shap_values, list): # RF/DT multi-class output
                shap.summary_plot(shap_values[1], X_test, show=False)
            else: # XGBoost
                shap.summary_plot(shap_values, X_test, show=False)
            plt.savefig("output/shap_summary.png", bbox_inches='tight')
            plt.close()
            print("Saved SHAP summary plot to 'output/shap_summary.png'")
        else:
            print(f"SHAP TreeExplainer not directly applicable to {best_model_name}. Skipping SHAP plot generation for brevity.")
    except Exception as e:
        print(f"Could not generate SHAP values: {e}")

    print("\n"+"="*60)
    print("PHASE 8: REUSABLE PIPELINE")
    print("="*60)
    
    os.makedirs("models", exist_ok=True)
    model_path = "models/mastitis_model_pipeline.joblib"
    joblib.dump(best_model, model_path)
    print(f"Pipeline saved to {model_path}")
    
    print("\nExample Inference (First 2 Test Samples):")
    sample_data = X_test.head(2)
    loaded_model = joblib.load(model_path)
    probs = loaded_model.predict_proba(sample_data)[:, 1] if hasattr(loaded_model[-1], "predict_proba") else [0,0]
    
    for i, (index, row) in enumerate(sample_data.iterrows()):
        prob = probs[i]
        risk = "HIGH RISK" if prob > 0.5 else "LOW RISK"
        print(f"Cow Sample #{i+1} - Probability of Mastitis: {prob:.4f} -> {risk}")

if __name__ == "__main__":
    run_pipeline()
