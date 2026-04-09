import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, classification_report

print("Starting training pipeline with Kaggle Dataset...")

# 1. Load Data
df = pd.read_csv('StudentPerformanceFactors.csv')
df = df.dropna()

def get_risk(score):
    if score < 50: return 'Critical'
    if score < 65: return 'High'
    if score < 80: return 'Medium'
    return 'Low'
    
df['Risk_Level'] = df['Exam_Score'].apply(get_risk)

X = df.drop(columns=['Exam_Score', 'Risk_Level'])
y_reg = df['Exam_Score']
y_clf = df['Risk_Level']

num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ]
)

# Train Regressor
X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
reg_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42))
])
reg_pipeline.fit(X_train, y_train_reg)
joblib.dump(reg_pipeline, 'xgb_regressor.pkl')

print(f"Regression RMSE: {np.sqrt(mean_squared_error(y_test_reg, reg_pipeline.predict(X_test))):.2f}")

# Train Classifier
le = LabelEncoder()
y_clf_encoded = le.fit_transform(y_clf)
joblib.dump(le, 'label_encoder.pkl')

X_train_c, X_test_c, y_train_clf, y_test_clf = train_test_split(X, y_clf_encoded, test_size=0.2, random_state=42)
clf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBClassifier(random_state=42, n_estimators=100, max_depth=3, learning_rate=0.1))
])
clf_pipeline.fit(X_train_c, y_train_clf)
joblib.dump(clf_pipeline, 'xgb_classifier.pkl')

print("Classification Report:")
print(classification_report(y_test_clf, clf_pipeline.predict(X_test_c), target_names=le.classes_))

# Save expected features for app.py
joblib.dump(X.columns.tolist(), 'feature_columns.pkl')

print("Completed successfully! Models saved to pkl files.")
