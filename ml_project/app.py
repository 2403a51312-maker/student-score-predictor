from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np

import os

app = Flask(__name__)

# Build absolute paths to the current directory dynamically for Vercel
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load models safely
try:
    reg_model = joblib.load(os.path.join(BASE_DIR, 'xgb_regressor.pkl'))
    clf_model = joblib.load(os.path.join(BASE_DIR, 'xgb_classifier.pkl'))
    le = joblib.load(os.path.join(BASE_DIR, 'label_encoder.pkl'))
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
        
        # Kaggle dataset features
        input_data = pd.DataFrame([{
            'Hours_Studied': float(data.get('Hours_Studied', 4.0)),
            'Attendance': float(data.get('Attendance', 85.0)),
            'Parental_Involvement': data.get('Parental_Involvement', 'Medium'),
            'Access_to_Resources': data.get('Access_to_Resources', 'Medium'),
            'Extracurricular_Activities': data.get('Extracurricular_Activities', 'No'),
            'Sleep_Hours': float(data.get('Sleep_Hours', 7.0)),
            'Previous_Scores': float(data.get('Previous_Scores', 75.0)),
            'Motivation_Level': data.get('Motivation_Level', 'Medium'),
            'Internet_Access': data.get('Internet_Access', 'Yes'),
            'Tutoring_Sessions': float(data.get('Tutoring_Sessions', 0.0)),
            'Family_Income': data.get('Family_Income', 'Medium'),
            'Teacher_Quality': data.get('Teacher_Quality', 'Medium'),
            'School_Type': data.get('School_Type', 'Public'),
            'Peer_Influence': data.get('Peer_Influence', 'Neutral'),
            'Physical_Activity': float(data.get('Physical_Activity', 3.0)),
            'Learning_Disabilities': data.get('Learning_Disabilities', 'No'),
            'Parental_Education_Level': data.get('Parental_Education_Level', 'High School'),
            'Distance_from_Home': data.get('Distance_from_Home', 'Near'),
            'Gender': data.get('Gender', 'Male')
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
