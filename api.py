import joblib
from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

# Load the saved model parameters and the data
try:
    model_deployment_info = joblib.load('model_params.pkl')
    best_pdq = model_deployment_info['best_pdq']
    best_seasonal_pdq = model_deployment_info['best_seasonal_pdq']
    y_full = model_deployment_info['data']
    print("Model parameters loaded successfully.")
except FileNotFoundError:
    print("Error: Model parameter file 'model_params.pkl' not found.")
    exit()

@app.route('/', methods=['GET'])
def home():
    return "Forecast API server is running."

@app.route('/forecast', methods=['POST'])
def get_forecast():
    """
    API endpoint to generate a forecast.
    Expects a JSON payload like:
    {
        "steps": 11
    }
    """
    try:
        # Get the number of steps from the JSON payload
        data = request.get_json()
        n_forecast_steps = data.get('steps', 1)

        if not isinstance(n_forecast_steps, int) or n_forecast_steps <= 0:
            return jsonify({"error": "Invalid number of steps. Must be a positive integer."}), 400

        # Create and fit the SARIMA model instance with the loaded parameters
        # This is the crucial step to ensure the model has access to the latest data
        final_model = SARIMAX(
            y_full,
            order=best_pdq,
            seasonal_order=best_seasonal_pdq,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)

        # Generate the forecast
        forecast = final_model.get_forecast(steps=n_forecast_steps)
        forecast_values = forecast.predicted_mean
        conf_int = forecast.conf_int()

        # Create a DataFrame for the output
        # Use the end of the loaded data for the forecast start point
        forecast_index = pd.date_range(start=y_full.index[-1] + pd.DateOffset(months=1), periods=n_forecast_steps, freq='MS')
        forecast_df = pd.DataFrame({
            'prediction': forecast_values.values,
            'lower_ci': conf_int.iloc[:, 0].values,
            'upper_ci': conf_int.iloc[:, 1].values
        }, index=forecast_index)
        
        # Prepare the response
        response = {
            "start_date": forecast_df.index[0].strftime('%Y-%m-%d'),
            "end_date": forecast_df.index[-1].strftime('%Y-%m-%d'),
            "forecast": forecast_df.reset_index().rename(columns={'index': 'date'}).to_dict('records')
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)