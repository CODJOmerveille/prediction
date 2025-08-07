
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for web requests

# Load the model when the app starts
try:
    model = joblib.load('property_price_model.pkl')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return render_template('property_price_predictor.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get data from request
        data = request.json
        
        # Extract features in the correct order
        features = [
            int(data['number_rooms']),
            int(data['number_living_rooms']),
            int(data['number_households']),
            int(data['water_meter_type']),
            int(data['sanitary']),
            int(data['is_fence']),
            int(data['electricity_meter_type'])
        ]
        
        # Create DataFrame with correct column names
        df = pd.DataFrame([features], columns=[
            'number_rooms',
            'number_living_rooms',
            'number_households', 
            'water_meter_type',
            'sanitary',
            'is_fence',
            'electricity_meter_type'
        ])
        print(features)
        # Make prediction
        prediction = model.predict(df)[0]
        print(prediction)

        # Return prediction as JSON
        return jsonify({
            'predicted_price': int(prediction),
            'formatted_price': f"${prediction:,.2f}",
            'success': True
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 400

@app.route('/test', methods=['GET'])
def test_prediction():
    """Test endpoint with sample data"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
            
        # Test with your sample data
        test_data = [3, 1, 2, 1, 1, 0, 1]
        df = pd.DataFrame([test_data], columns=[
            'number_rooms', 'number_living_rooms', 'number_households',
            'water_meter_type', 'sanitary', 'is_fence', 'electricity_meter_type'
        ])
        
        prediction = model.predict(df)[0]
        
        return jsonify({
            'test_prediction': int(prediction),
            'test_data': test_data,
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

