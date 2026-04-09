# Student Exam Score Prediction & At-Risk Classification
**End-to-End Machine Learning Project Guide**

Welcome to your ML project! This guide will walk you through all 7 phases of building a robust machine learning system that predicts student exam scores (Regression) and classifies them into risk categories (Classification). 

---

## Phase 1: Problem Definition & Success Metrics

**What to do and why:**
Before writing any code, we need to clearly define what we are trying to solve. Our goal is to identify students who are at risk of failing or underperforming so that early interventions can be made. Because we want a nuanced understanding, we will solve two tasks:
1. **Regression:** Predict the exact final exam score (0-100).
2. **Classification:** Categorize students into Risk Levels based on their predicted scores.

**Success Metrics:**
- *Regression:* Root Mean Squared Error (RMSE) < 5.0, Mean Absolute Error (MAE) < 4.0.
- *Classification:* F1-Score (macro) > 0.85, prioritizing recall on the "Critical" and "High" risk classes (we don't want to miss struggling students).

**Common Pitfalls:**
- *Data Leakage:* Including features in the training data that wouldn't be available at prediction time (e.g., questions answered correctly on the final exam).
- *Class Imbalance:* Most students might be "Low Risk". If we don't balance the classes or use the right metrics, the model might just predict "Low Risk" for everyone.

---

## Phase 2: Dataset Creation & EDA

**What to do and why:**
We need data. Since this is a prototype, we'll generate a synthetic but realistic dataset. Then, we perform Exploratory Data Analysis (EDA) to understand feature distributions, correlations, and outliers.

**Full Working Python Code (`phase2_data_eda.py`):**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    
    # Features
    study_hours = np.random.uniform(0, 12, n_samples)
    attendance = np.random.uniform(40, 100, n_samples)
    sleep_quality = np.random.choice(['poor', 'average', 'good'], p=[0.2, 0.5, 0.3], size=n_samples)
    
    # Previous scores: list of 3 past exams
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
    
    # Target Generation (Creating a logical relationship)
    avg_prev = np.array([np.mean(scores) for scores in prev_scores])
    
    # Base score
    base_score = 10 + (study_hours * 2.5) + (attendance * 0.3) + (avg_prev * 0.4)
    # Adjust for categories
    base_score[sleep_quality == 'poor'] -= 5
    base_score[stress == 'high'] -= 4
    base_score[peer_group == True] += 3
    base_score[ses == 'high'] += 5 # assuming SES impact for synthetic demo
    
    # Add noise & clip
    exam_score = base_score + np.random.normal(0, 5, n_samples)
    exam_score = np.clip(exam_score, 0, 100)
    
    # Risk Level classification
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
    return df

df = generate_synthetic_data(1000)
df.to_csv('student_data.csv', index=False)

# EDA Plot
plt.figure(figsize=(10,6))
sns.histplot(df['exam_score'], bins=30, kde=True, color='blue')
plt.title('Distribution of Exam Scores')
plt.xlabel('Exam Score')
plt.ylabel('Frequency')
plt.savefig('score_distribution.png')
print("Dataset created and saved to student_data.csv. EDA plot saved.")
```

**Expected Outputs:**
- A `student_data.csv` file with 1000 rows.
- An image `score_distribution.png` showing a normal-ish distribution of scores.

**Common Pitfalls:**
- *Ignoring missing values early on.* Always check `df.isnull().sum()`. Our synthetic dataset is complete, but real data is messy.

---

## Phase 3: Data Preprocessing & Cleaning

**What to do and why:**
Machine learning models need numbers, not strings or lists. We will unpack the lists (previous scores) and handle the categorical variables appropriately using encoding.

**Full Working Python Code (`phase3_preprocessing.py`):**
```python
import pandas as pd
import ast

df = pd.read_csv('student_data.csv')

# 1. Unpack arrays saved as strings from CSV
df['previous_exam_scores'] = df['previous_exam_scores'].apply(ast.literal_eval)

# 2. Check for missing values
df.fillna(df.median(numeric_only=True), inplace=True) # Basic imputation

# 3. Categorical encoding setup (we'll use scikit-learn pipelines in phase 4/5)
# Convert booleans to integers
bool_cols = df.select_dtypes(include=['bool']).columns
df[bool_cols] = df[bool_cols].astype(int)

print("Data shape after initial cleaning:", df.shape)
df.to_pickle('cleaned_data.pkl')
```

**Common Pitfalls:**
- *Applying transformations incorrectly.* Make sure to save processed data formats (like lists) as pickles or binaries since CSVs convert lists to strings which require parsing (`ast.literal_eval`).

---

## Phase 4: Feature Engineering

**What to do and why:**
We create new features from existing ones to highlight patterns for the model. For instance, the raw list of past scores is less useful than its statistical summary.

**Full Working Python Code (`phase4_features.py`):**
```python
import pandas as pd
import numpy as np

df = pd.read_pickle('cleaned_data.pkl')

# Feature Engineering
# Extract stats from previous_exam_scores
df['prev_score_mean'] = df['previous_exam_scores'].apply(np.mean)
df['prev_score_min'] = df['previous_exam_scores'].apply(np.min)
df['prev_score_max'] = df['previous_exam_scores'].apply(np.max)
# Trend: is the student improving? (Score 3 minus Score 1)
df['prev_score_trend'] = df['previous_exam_scores'].apply(lambda x: x[-1] - x[0] if len(x)>=3 else 0)

# Drop the raw list column as it's no longer needed
df = df.drop(columns=['previous_exam_scores'])

# Total workload feature
df['total_workload_hours'] = df['study_hours_per_day'] * 5 + df['extracurricular_hours']

print("Features engineered. New columns:", df.columns.tolist())
df.to_pickle('featured_data.pkl')
```

**Common Pitfalls:**
- *Over-engineering:* Creating too many highly correlated features (multicollinearity) can confuse linear models, though XGBoost handles it reasonably well.

---

## Phase 5: Model Training & Hyperparameter Tuning

**What to do and why:**
We will build a pipeline that Handles Categorical Encoding (One-Hot), Scaling, and the Model (XGBoost). XGBoost is an industry standard for tabular data. We'll train both a Regressor and a Classifier.

**Full Working Python Code (`phase5_training.py`):**
```python
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from xgboost import XGBRegressor, XGBClassifier
import joblib

df = pd.read_pickle('featured_data.pkl')

# Separate features and targets
X = df.drop(columns=['exam_score', 'risk_level'])
y_reg = df['exam_score']
y_clf = df['risk_level']

# Identify column types
num_cols = X.select_dtypes(exclude=['object', 'category']).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ]
)

# --- REGRESSION MODEL ---
X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)

reg_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(random_state=42))
])

# Simple Grid Search for Tuning
param_grid = {'model__n_estimators': [50, 100], 'model__max_depth': [3, 5], 'model__learning_rate': [0.1, 0.05]}
reg_search = GridSearchCV(reg_pipeline, param_grid, cv=3, scoring='neg_mean_squared_error')
reg_search.fit(X_train, y_train_reg)

best_reg = reg_search.best_estimator_
joblib.dump(best_reg, 'xgb_regressor.pkl')
print(f"Best Regressor Params: {reg_search.best_params_}")


# --- CLASSIFICATION MODEL ---
# Label encode the target variable for XGBoost
le = LabelEncoder()
y_clf_encoded = le.fit_transform(y_clf)
joblib.dump(le, 'label_encoder.pkl')

X_train_c, X_test_c, y_train_clf, y_test_clf = train_test_split(X, y_clf_encoded, test_size=0.2, random_state=42)

clf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'))
])
clf_pipeline.fit(X_train_c, y_train_clf)
joblib.dump(clf_pipeline, 'xgb_classifier.pkl')
print("Models trained and saved.")
```

**Common Pitfalls:**
- *Data Leakage in Scaling:* Applying StandardScaler to the whole dataset before train_test_split leaks information from the test set into the training set. Using Scikit-Learn `Pipeline` prevents this.

---

## Phase 6: Model Evaluation & Interpretability

**What to do and why:**
We assess model performance using evaluation metrics. We also use SHAP (SHapley Additive exPlanations) to interpret *why* the model makes specific predictions (vital for educational contexts).

**Full Working Python Code (`phase6_evaluation.py`):**
```python
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import shap

df = pd.read_pickle('featured_data.pkl')
X = df.drop(columns=['exam_score', 'risk_level'])

# Load test sets and models
from sklearn.model_selection import train_test_split
# Ensure exact same split as training
X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, df['exam_score'], test_size=0.2, random_state=42)
_, _, _, y_test_clf = train_test_split(X, df['risk_level'], test_size=0.2, random_state=42)

reg_model = joblib.load('xgb_regressor.pkl')
clf_model = joblib.load('xgb_classifier.pkl')
le = joblib.load('label_encoder.pkl')

# --- Regression Evaluation ---
y_pred_reg = reg_model.predict(X_test)
print(f"Regression RMSE: {np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)):.2f}")
print(f"Regression MAE: {mean_absolute_error(y_test_reg, y_pred_reg):.2f}")

# --- Classification Evaluation ---
y_test_clf_encoded = le.transform(y_test_clf)
y_pred_clf = clf_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test_clf_encoded, y_pred_clf, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test_clf_encoded, y_pred_clf)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
plt.title('Risk Level Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix.png')

# --- SHAP Interpretability ---
# Extract preprocessor and model from pipeline
preprocessor = reg_model.named_steps['preprocessor']
xgb_underlying = reg_model.named_steps['model']

# Transform X_train to get feature names
X_train_transformed = preprocessor.transform(X_train)
# Get exact feature names
num_cols = reg_model.named_steps['preprocessor'].transformers_[0][2]
cat_cols = list(reg_model.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out())
all_features = num_cols + cat_cols

explainer = shap.TreeExplainer(xgb_underlying)
shap_values = explainer.shap_values(X_train_transformed)

plt.figure()
shap.summary_plot(shap_values, X_train_transformed, feature_names=all_features, show=False)
plt.savefig('shap_summary.png', bbox_inches='tight')
print("Evaluation complete. Plots saved.")
```

**Common Pitfalls:**
- *Focusing only on Accuracy:* For classification, accuracy is misleading if data is imbalanced. Precision, Recall, and F1-score for specific minority classes (like "Critical" risk) matter more.

---

## Phase 7: Deployment (Flask API + Simple Frontend)

**What to do and why:**
A model is only useful if people can use it! We wrap it in a lightweight Flask web application with a simple HTML frontend for teachers to input data and get a prediction.

**Full Working Python Code (`app.py` & `templates/index.html`):**

Create a folder structure:
```
project/
|-- app.py
|-- templates/
|   |-- index.html
|-- xgb_regressor.pkl
|-- label_encoder.pkl
```

**`app.py`:**
```python
from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load models
reg_model = joblib.load('xgb_regressor.pkl')
clf_model = joblib.load('xgb_classifier.pkl')
le = joblib.load('label_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    data = request.form.to_dict()
    
    # Needs to match EXACT training features
    try:
        input_data = pd.DataFrame([{
            'study_hours_per_day': float(data['study_hours']),
            'attendance_percent': float(data['attendance']),
            'sleep_quality': data['sleep'],
            'assignment_completion': float(data['assignments']),
            'stress_level': data['stress'],
            'participation_level': 'medium', # defaulting unspecified ones for demo
            'peer_study_group': 1,
            'internet_access': 'full',
            'extracurricular_hours': 5.0,
            'parent_involvement': 'moderate',
            'first_generation_student': 0,
            'socioeconomic_status': 'medium',
            'grade_level': 'Grade 10',
            # Derived features mock
            'prev_score_mean': float(data['prev_score']),
            'prev_score_min': float(data['prev_score']) - 5,
            'prev_score_max': float(data['prev_score']) + 5,
            'prev_score_trend': 0.0,
            'total_workload_hours': float(data['study_hours']) * 5 + 5.0
        }])
        
        score_pred = reg_model.predict(input_data)[0]
        risk_encoded = clf_model.predict(input_data)[0]
        risk_pred = le.inverse_transform([risk_encoded])[0]
        
        return render_template('index.html', 
                               prediction_text=f'Predicted Score: {score_pred:.1f}/100. Risk Level: {risk_pred}')
    except Exception as e:
        return render_template('index.html', 
                               prediction_text=f'Error Processing Input: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True, port=5000)
```

**`templates/index.html`:**
```html
<!DOCTYPE html>
<html>
<head>
  <title>Student Success Predictor</title>
  <style>
    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f4f7f6; padding: 50px; }
    .card { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); max-width: 500px; margin: auto; }
    input, select, button { width: 100%; padding: 10px; margin: 10px 0; border-radius: 5px; border: 1px solid #ccc; box-sizing: border-box; }
    button { background: #007BFF; color: white; border: none; cursor: pointer; font-size: 16px; font-weight: bold; }
    button:hover { background: #0056b3; }
    .result { margin-top: 20px; font-size: 1.2em; font-weight: bold; color: #333; text-align: center; }
  </style>
</head>
<body>
  <div class="card">
    <h2>🚀 Student Success Predictor</h2>
    <form action="/predict" method="post">
      <label>Average Study Hours/Day:</label>
      <input type="number" name="study_hours" step="0.1" required value="4">
      
      <label>Attendance %:</label>
      <input type="number" name="attendance" step="0.1" required value="85">
      
      <label>Recent Exam Avg (0-100):</label>
      <input type="number" name="prev_score" step="0.1" required value="75">
      
      <label>Assignment Completion %:</label>
      <input type="number" name="assignments" step="0.1" required value="90">

      <label>Sleep Quality:</label>
      <select name="sleep">
        <option value="poor">Poor</option>
        <option value="average" selected>Average</option>
        <option value="good">Good</option>
      </select>
      
      <label>Stress Level:</label>
      <select name="stress">
        <option value="low">Low</option>
        <option value="medium" selected>Medium</option>
        <option value="high">High</option>
      </select>

      <button type="submit">Predict Outcome</button>
    </form>
    
    {% if prediction_text %}
    <div class="result">{{ prediction_text }}</div>
    {% endif %}
  </div>
</body>
</html>
```

**Common Pitfalls:**
- *Data Model Mismatch:* The raw inputs from a web form don't match the exact features the pipeline expects. The Flask code above artificially injects the missing features (like `participation_level`) to map form data safely to the ML pipeline. In production, your form needs to collect every feature or reliably default them.
