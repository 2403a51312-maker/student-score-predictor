from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load models safely
try:
    reg_model = joblib.load('xgb_regressor.pkl')
    clf_model = joblib.load('xgb_classifier.pkl')
    le = joblib.load('label_encoder.pkl')
    print("Models loaded successfully!")
except Exception as e:
    print(f"Warning: Models not found, please run train.py first. Error: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        
        # Build the feature exactly as the pipeline expects
        input_data = pd.DataFrame([{
            'study_hours_per_day': float(data.get('study_hours', 4.0)),
            'attendance_percent': float(data.get('attendance', 85.0)),
            'sleep_quality': data.get('sleep', 'average'),
            'assignment_completion': float(data.get('assignments', 80.0)),
            'stress_level': data.get('stress', 'medium'),
            'participation_level': 'medium',
            'peer_study_group': 1,
            'internet_access': 'full',
            'extracurricular_hours': 5.0,
            'parent_involvement': 'moderate',
            'first_generation_student': 0,
            'socioeconomic_status': 'medium',
            'grade_level': 'Grade 10',
            'prev_score_mean': float(data.get('prev_score', 75.0)),
            'prev_score_min': float(data.get('prev_score', 75.0)) - 5,
            'prev_score_max': float(data.get('prev_score', 75.0)) + 5,
            'prev_score_trend': 0.0,
            'total_workload_hours': float(data.get('study_hours', 4.0)) * 5 + 5.0
        }])
        
        score_pred = reg_model.predict(input_data)[0]
        risk_encoded = clf_model.predict(input_data)[0]
        risk_pred = le.inverse_transform([risk_encoded])[0]
        
        return render_template('index.html', 
                               prediction_text=f'Predicted Final Score: {score_pred:.1f}/100',
                               risk_level=f'Risk Level: {risk_pred}',
                               risk_class=risk_pred.lower())
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return render_template('index.html', 
                               prediction_text=f'Error Processing Input: {str(e)}')

if __name__ == "__main__":
    # Runs the flask app on localhost:5000
    app.run(debug=True, port=5000)
