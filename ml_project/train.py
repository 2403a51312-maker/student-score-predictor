import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, classification_report, confusion_matrix
import joblib
import os

print("Starting training pipeline...")

# -------------- PHASE 2: Data Generation --------------
np.random.seed(42)
n_samples = 1000

study_hours = np.random.uniform(0, 12, n_samples)
attendance = np.random.uniform(40, 100, n_samples)
sleep_quality = np.random.choice(['poor', 'average', 'good'], p=[0.2, 0.5, 0.3], size=n_samples)
prev_scores = [list(np.random.normal(loc=70, scale=15, size=3).clip(0, 100)) for _ in range(n_samples)]
assign_comp = np.random.uniform(0, 100, n_samples)
stress = np.random.choice(['low', 'medium', 'high'], p=[0.4, 0.4, 0.2], size=n_samples)
participation = np.random.choice(['low', 'medium', 'high'], size=n_samples)
peer_group = np.random.choice([True, False], size=n_samples)
internet = np.random.choice(['limited', 'moderate', 'full'], size=n_samples)
extra_hours = np.random.uniform(0, 20, n_samples)
parent_inv = np.random.choice(['low', 'moderate', 'high'], size=n_samples)
first_gen = np.random.choice([True, False], p=[0.3, 0.7], size=n_samples)
ses = np.random.choice(['low', 'medium', 'high'], size=n_samples)
grade_level = np.random.choice(['Grade 9', 'Grade 10', 'Grade 11', 'Grade 12'], size=n_samples)

avg_prev = np.array([np.mean(scores) for scores in prev_scores])
base_score = 10 + (study_hours * 2.5) + (attendance * 0.3) + (avg_prev * 0.4)
base_score[sleep_quality == 'poor'] -= 5
base_score[stress == 'high'] -= 4
base_score[peer_group == True] += 3
base_score[ses == 'high'] += 5

exam_score = base_score + np.random.normal(0, 5, n_samples)
exam_score = np.clip(exam_score, 0, 100)

def get_risk(score):
    if score < 50: return 'Critical'
    if score < 65: return 'High'
    if score < 80: return 'Medium'
    return 'Low'
    
risk_level = [get_risk(s) for s in exam_score]

df = pd.DataFrame({
    'study_hours_per_day': study_hours,
    'attendance_percent': attendance,
    'sleep_quality': sleep_quality,
    'previous_exam_scores': prev_scores,
    'assignment_completion': assign_comp,
    'stress_level': stress,
    'participation_level': participation,
    'peer_study_group': peer_group,
    'internet_access': internet,
    'extracurricular_hours': extra_hours,
    'parent_involvement': parent_inv,
    'first_generation_student': first_gen,
    'socioeconomic_status': ses,
    'grade_level': grade_level,
    'exam_score': exam_score,
    'risk_level': risk_level
})

# -------------- PHASE 3: Preprocessing & Cleaning --------------
bool_cols = df.select_dtypes(include=['bool']).columns
df[bool_cols] = df[bool_cols].astype(int)

# -------------- PHASE 4: Feature Engineering --------------
df['prev_score_mean'] = df['previous_exam_scores'].apply(np.mean)
df['prev_score_min'] = df['previous_exam_scores'].apply(np.min)
df['prev_score_max'] = df['previous_exam_scores'].apply(np.max)
df['prev_score_trend'] = df['previous_exam_scores'].apply(lambda x: x[-1] - x[0] if len(x)>=3 else 0)
df = df.drop(columns=['previous_exam_scores'])
df['total_workload_hours'] = df['study_hours_per_day'] * 5 + df['extracurricular_hours']

# -------------- PHASE 5: Model Training --------------
print("Training models... (this might take a few moments)")
X = df.drop(columns=['exam_score', 'risk_level'])
y_reg = df['exam_score']
y_clf = df['risk_level']

num_cols = X.select_dtypes(exclude=['object', 'category']).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ]
)

# Regressor
X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
reg_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42))
])
reg_pipeline.fit(X_train, y_train_reg)
joblib.dump(reg_pipeline, 'xgb_regressor.pkl')

# Classifier
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

# -------------- PHASE 6: basic Evaluation --------------
print("\nEvaluation:")
y_pred_reg = reg_pipeline.predict(X_test)
print(f"Regression RMSE: {np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)):.2f}")

y_pred_clf = clf_pipeline.predict(X_test_c)
print("Classification Report:")
print(classification_report(y_test_clf, y_pred_clf, target_names=le.classes_))

print("Completed successfully! Models saved to pkl files.")
