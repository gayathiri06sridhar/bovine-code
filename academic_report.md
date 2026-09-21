# End-to-End Bovine Mastitis Prediction
**Academic Project Report**

## 1. Introduction
Bovine mastitis is a significant agricultural issue causing economic losses and impacting animal welfare. Early detection is crucial. This project leverages Machine Learning (ML) to predict the onset of mastitis using physiological data gathered from cows.

## 2. Methodology
### 2.1 Data Preparation
- **Feature Selection:** Predictor variables included `Day`, `Milk_Temperature`, `Milk_pH`, `Milk_Conductivity`, `Somatic_Cell_Count`, `Milk_Yield`, and `Clotting`. `Cow_ID` was excluded from prediction to prevent identifier bias but was retained for group-based splitting.
- **Handling Data Splitting & Leakage:** To ensure a robust evaluation and prevent target leakage, the dataset was split 80/20. Where repeated cows were detected, a `GroupShuffleSplit` (or `StratifiedGroupKFold` for CV) was used so the same cow did not appear in both the training and test sets.
- **Handling Imbalance & Scaling:** We applied `class_weight='balanced'` for models sensitive to imbalance. We scaled features for distance and gradient-based models (Logistic Regression, SVM, KNN) using `StandardScaler` within a Pipeline to prevent data leakage during Cross-Validation.

### 2.2 Model Training and Evaluation
Six models were evaluated using 5-Fold Cross Validation:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (SVM)
5. K-Nearest Neighbors (KNN)
6. XGBoost

The models were evaluated primarily on the **F1-Score** and **ROC-AUC** to balance the trade-off between Precision and Recall.

### 2.3 Hyperparameter Tuning
The strongest performing model was selected for hyperparameter tuning using `RandomizedSearchCV` on the training set to optimize accuracy and F1-score without touching the hold-out test set.

## 3. Results and Model Explanation (XAI)
- **Model Performance:** The final tuned model was evaluated on the test set, outputting a detailed Classification Report and Confusion Matrix. Analysis of False Positives (unnecessary vet checks) and False Negatives (missed infections) guided threshold considerations.
- **Explainable AI (SHAP):** SHAP (SHapley Additive exPlanations) values were extracted for the tree-based models. The SHAP summary plot highlights the most impactful features (e.g., Somatic Cell Count, Milk Conductivity) driving the model's predictions, providing transparency for veterinarians.

## 4. Architecture: IoT Integration
A complete system design for real-time monitoring was developed:
- Sensors collect milk data and transmit it via an edge gateway to the cloud.
- A REST API hosts the serialized ML pipeline (`.joblib` model).
- The API processes raw sensor JSON payloads, returning a probability score and risk level (HIGH/LOW) to a farmer's dashboard.

## 5. Limitations
1. **Feature Scope:** The model relies on specific hardware (e.g., automated milking systems capable of inline SCC and conductivity testing) which may not be available on all farms.
2. **Dataset Size and Diversity:** Models are highly dependent on the training dataset. Variations in breed, diet, or regional climate may reduce the model's generalizability unless retrained with local data.

## 6. Future Work
- **Temporal Analysis:** Transitioning from static cross-sectional predictions to Recurrent Neural Networks (RNNs) or LSTMs to analyze time-series trends per cow.
- **Multi-Class Severity:** Extending the binary classification (Mastitis: Yes/No) to predict severity levels (Subclinical vs. Clinical).
- **Edge Deployment:** Quantizing and deploying the ML model directly onto the edge gateway (e.g., Raspberry Pi) to enable offline predictions when internet connectivity is poor.
