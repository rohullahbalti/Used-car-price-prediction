from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import joblib
import os

app = Flask(__name__)

# Load models and data
current_price_model = None
future_price_model = None
car_data = None

def load_models():
    global current_price_model, future_price_model, car_data
    try:
        current_price_model = pickle.load(open('final_prediction_current_price.pkl', 'rb'))
        future_price_model = joblib.load('multioutput_car_price_second_model.pkl')
        car_data = pd.read_csv('OLX_cars_dataset00.csv')
    except Exception as e:
        print(f"Error loading models: {e}")

# Helper functions (same as your code)
def calculate_age(year):
    return 2024 - year

def calculate_year_range(year):
    bins_year = [1999, 2004, 2008, 2012, 2016, 2020, 2024]
    labels_year = [1, 2, 3, 4, 5, 6]
    return pd.cut([year], bins=bins_year, labels=labels_year).astype("int32")[0]

def calculate_km_range(km_driven):
    bins_km = [0, 30000, 60000, 90000, 120000, 150000, 180000, 210000, 224000, 227000, 300000, 
               330000, 360000, 390000, 410000, 440000, 470000, 500000, 533530]
    labels_km = list(range(1, len(bins_km)))
    return pd.cut([km_driven], bins=bins_km, labels=labels_km).astype("int32")[0]

# Routes
@app.route('/')
def index():
    unique_makes = car_data['Make'].dropna().unique().tolist() if car_data is not None else []
    unique_models = car_data['Model'].dropna().unique().tolist() if car_data is not None else []
    return render_template('index.html', makes=unique_makes, models=unique_models)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Prepare input data
        sample_input = {
            "Make": data['make'],
            "Model": data['model'],
            "Year": int(data['year']),
            "KM's driven": int(data['km_driven']),
            "Fuel": data['fuel'],
            "Car documents": data['car_documents'],
            "Assembly": data['assembly'],
            "Transmission": data['transmission'],
            "Age": calculate_age(int(data['year'])),
            "Year_Range": calculate_year_range(int(data['year'])),
            "KM's driven_Range": calculate_km_range(int(data['km_driven']))
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([sample_input])
        
        # Predict current price
        predicted_price = current_price_model.predict(input_df)[0]
        
        return jsonify({
            'success': True,
            'current_price': predicted_price,
            'input_data': sample_input
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/predict_future', methods=['POST'])
def predict_future():
    try:
        data = request.get_json()
        input_data = data['input_data']
        years = int(data['years'])
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Predict future price
        future_prices = future_price_model.predict(input_df)[0]
        future_price = future_prices[years - 1] if years <= len(future_prices) else future_prices[-1]
        
        return jsonify({
            'success': True,
            'future_price': future_price,
            'years': years
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    load_models()
    app.run(debug=True)
