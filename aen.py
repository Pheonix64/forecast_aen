# ==============================================================================
# 1. CONFIGURATION ET PRÉPARATION DES DONNÉES
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import random
import os
import warnings

# Utilitaires de modélisation
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools # Pour la recherche par grille

# Utilitaires Keras/TensorFlow pour LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# --- Ignorer les avertissements de convergence de statsmodels ---
warnings.filterwarnings("ignore", category=UserWarning, module='statsmodels')

# --- Reproductibilité ---
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
tf.random.set_seed(SEED)

# --- Style des graphiques ---
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# --- Fonction de transformation des données (inchangée) ---
def transform_aen_data(df, cln, name):
    """Transforms the raw AEN DataFrame into a clean time series format."""
    aen_series = df.drop(columns=cln).iloc[0]
    aen_transformed_df = aen_series.reset_index()
    aen_transformed_df.columns = ['Date', name]
    aen_transformed_df[name] = pd.to_numeric(aen_transformed_df[name], errors='coerce')
    month_map = {
        'JAN': '01', 'FEV': '02', 'MAR': '03', 'AVR': '04', 'MAY': '05', 'JUN': '06',
        'JUL': '07', 'AUG': '08', 'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12'
    }
    def convert_date(d):
        if isinstance(d, pd.Timestamp): return d
        try:
            date_str = str(d)
            month_str, year_str = date_str[:3].upper(), date_str[3:]
            if month_str in month_map:
                return pd.to_datetime(f"{year_str}-{month_map[month_str]}-01")
        except (TypeError, ValueError): pass
        return pd.NaT
    aen_transformed_df['Date'] = aen_transformed_df['Date'].apply(convert_date)
    aen_transformed_df.dropna(subset=['Date', name], inplace=True)
    aen_transformed_df = aen_transformed_df.sort_values(by='Date').reset_index(drop=True)
    return aen_transformed_df

# --- Chargement et préparation de la série ---
aen_raw_df = pd.read_excel('aen.xlsx')
df_ts = transform_aen_data(aen_raw_df, 'LIBELLE', 'Valeur AEN')
df_ts = df_ts.set_index('Date')
df_ts = df_ts.asfreq('MS') # S'assurer d'une fréquence mensuelle
y = df_ts['Valeur AEN'].interpolate(method='time') # Interpolation linéaire pour les mois manquants

# --- Découpage temporel (70% Train, 15% Val, 15% Test) ---
n = len(y)
i_train = int(n * 0.70)
i_val = int(n * 0.85)
y_train, y_val, y_test = y.iloc[:i_train], y.iloc[i_train:i_val], y.iloc[i_val:]

print(f"Série temporelle préparée avec {n} observations.")
print(f"Tailles des ensembles -> Train:{len(y_train)} | Validation:{len(y_val)} | Test:{len(y_test)}")
print(f"Périodes -> Train: {y_train.index.min().year}-{y_train.index.max().year} | Val: {y_val.index.min().year}-{y_val.index.max().year} | Test: {y_test.index.min().year}-{y_test.index.max().year}")

# --- Métriques d'évaluation ---
def evaluate_forecast(y_true, y_pred, model_name=""):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred) # Calculate MSE first
    rmse = np.sqrt(mse) # Then take the square root for RMSE
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"--- Évaluation {model_name} ---")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
    return {'mae': mae, 'rmse': rmse, 'mape': mape}

# ==============================================================================
# 2. MODÉLISATION ARIMA SAISONNIER (AVEC RECHERCHE PAR GRILLE)
# ==============================================================================
print("\n--- Recherche du meilleur modèle SARIMA via Grid Search ---")
# Définir les plages de p, d, q, P, D, Q à tester
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

best_aic = float("inf")
best_pdq = None
best_seasonal_pdq = None
best_model = None

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = SARIMAX(y_train,
                          order=param,
                          seasonal_order=param_seasonal,
                          enforce_stationarity=False,
                          enforce_invertibility=False)
            results = mod.fit(disp=False)
            if results.aic < best_aic:
                best_aic = results.aic
                best_pdq = param
                best_seasonal_pdq = param_seasonal
                best_model = results
        except:
            continue

print(f"Meilleurs paramètres SARIMA trouvés: {best_pdq} {best_seasonal_pdq} avec AIC={best_aic:.2f}")

# Évaluation sur le jeu de test
arima_pred = best_model.forecast(steps=len(y_test))
arima_pred.index = y_test.index # Assigner le bon index pour l'évaluation
metrics_arima = evaluate_forecast(y_test, arima_pred, "ARIMA")


# ==============================================================================
# 3. MODÉLISATION LSTM (LONG SHORT-TERM MEMORY) - Inchangé
# ==============================================================================
# --- Préparation des données spécifiques au LSTM ---
scaler = MinMaxScaler(feature_range=(0, 1))
y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1))
y_val_scaled = scaler.transform(y_val.values.reshape(-1, 1))
y_test_scaled = scaler.transform(y_test.values.reshape(-1, 1))

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), 0])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

SEQ_LENGTH = 12
X_train, y_train_seq = create_sequences(y_train_scaled, SEQ_LENGTH)
X_val, y_val_seq = create_sequences(y_val_scaled, SEQ_LENGTH)
X_test, y_test_seq = create_sequences(y_test_scaled, SEQ_LENGTH)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# --- Construction et entraînement du modèle LSTM ---
print("\n--- Entraînement du modèle LSTM ---")
model_lstm = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(units=1)
])
model_lstm.compile(optimizer='adam', loss='mean_squared_error')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model_lstm.fit(
    X_train, y_train_seq,
    epochs=100, batch_size=32,
    validation_data=(X_val, y_val_seq),
    callbacks=[early_stopping],
    verbose=0
)

# --- Évaluation LSTM sur le jeu de test ---
lstm_pred_scaled = model_lstm.predict(X_test, verbose=0)
lstm_pred = scaler.inverse_transform(lstm_pred_scaled)
y_test_actual = scaler.inverse_transform(y_test_seq.reshape(-1, 1))
metrics_lstm = evaluate_forecast(y_test_actual, lstm_pred, "LSTM")


# ==============================================================================
# 4. PRÉVISION FINALE POUR FÉVRIER-DÉCEMBRE 2025
# ==============================================================================
print("\n--- Préparation de la prévision finale pour 2025 ---")
# On ré-entraîne le meilleur modèle ARIMA sur toutes les données
y_full = y.loc[:'2025-01-01']

final_model = SARIMAX(
    y_full,
    order=best_pdq,
    seasonal_order=best_seasonal_pdq,
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

print(f"Modèle final ré-entraîné: SARIMA{best_pdq}{best_seasonal_pdq}")

# Prévision pour les 11 mois restants de 2025
n_forecast = 11
forecast = final_model.get_forecast(steps=n_forecast)
forecast_values = forecast.predicted_mean
conf_int = forecast.conf_int()

# Création du DataFrame de prévision
forecast_index = pd.date_range(start='2025-02-01', periods=n_forecast, freq='MS')
forecast_df = pd.DataFrame({
    'Prediction': forecast_values,
    'Lower_CI': conf_int.iloc[:, 0],
    'Upper_CI': conf_int.iloc[:, 1]
}, index=forecast_index)

import joblib

# In the training script, after finding the best_model:
# best_model is the fitted SARIMAXResultsWrapper object
# y_full is the full training data up to the last known point

# Save the model object and the data
# Save the best parameters and the full dataset
model_deployment_info = {
    'best_pdq': best_pdq,
    'best_seasonal_pdq': best_seasonal_pdq,
    'data': y # Save the entire time series
}
joblib.dump(model_deployment_info, 'model_params.pkl')
print("Model parameters and data saved for deployment.")

print("\nPrévisions pour Février-Décembre 2025:")
print(forecast_df)

# Export en CSV
forecast_df.to_csv('forecast_aen_2025.csv')
print("\n'forecast_aen_2025.csv' a été exporté.")

# ==============================================================================
# 5. VISUALISATION FINALE
# ==============================================================================
plt.figure(figsize=(14, 7))
plt.plot(y.loc['2018':], label='Données historiques (depuis 2018)')
plt.plot(forecast_df['Prediction'], 'r--', label='Prévision ARIMA 2025')
plt.fill_between(
    forecast_df.index,
    forecast_df['Lower_CI'],
    forecast_df['Upper_CI'],
    color='r', alpha=0.1, label='Intervalle de confiance à 95%'
)
plt.title('Prévision des Actifs Extérieurs Nets pour 2025', fontsize=16)
plt.ylabel('Valeur AEN')
plt.xlabel('Date')
plt.legend()
plt.grid(True)
plt.show()

"""
To improve the performance of your time series forecasting models, you can implement several strategies focusing on both data preprocessing and model selection/tuning. Here are key approaches for both SARIMA and LSTM.

1. Data Preprocessing Enhancements
Handling Outliers: The current script uses interpolation to fill missing values, but it does not address potential outliers. Identify and handle them using methods like winsorizing or replacing them with the median or a robust average to prevent them from skewing the model's training.

Feature Engineering: Incorporate additional features that could influence Net External Assets (AEN), such as:

Lag Features: Create new variables by lagging the AEN series.

Time-based Features: Extract features from the date index, like the month, quarter, or day of the week (although the latter is less relevant for monthly data).

Exogenous Variables (SARIMAX): Use other time series that are known to correlate with AEN, such as GDP, exchange rates, or key import/export figures. The SARIMA model can be extended to SARIMAX to include these.

Data Transformation: Beyond scaling for the LSTM, you can apply other transformations to stabilize variance and improve stationarity for both models. For instance, a log transformation can stabilize the variance of a series with an increasing trend.

2. SARIMA Model Improvements
Expand the Grid Search: The current grid search for the best (p, d, q) and (P, D, Q, s) parameters is limited to a range of 0 to 1. Expanding this range (e.g., to range(0, 3)) could find better-performing parameter combinations, although it would significantly increase the computation time.

Automated Parameter Selection: Instead of a manual grid search, consider using automated libraries like pmdarima (auto_arima). This package automatically finds the best ARIMA model based on information criteria like AIC or BIC, often more efficiently than a brute-force grid search.

3. LSTM Model Improvements
Hyperparameter Tuning: The LSTM model has fixed hyperparameters. You can use a more systematic approach to find the optimal values for:

Number of LSTM Layers: Experiment with adding more layers.

Number of Neurons (units): Test different values (e.g., 50, 100, 150) to find the right balance between model capacity and risk of overfitting.

Dropout Rate: Tune the dropout rate (e.g., from 0.1 to 0.5) to prevent overfitting.

Learning Rate: Use callbacks like ReduceLROnPlateau to dynamically adjust the learning rate during training.

Sequence Length: The current script uses a fixed SEQ_LENGTH of 12. Experiment with different sequence lengths, as the optimal length depends on the periodicity and patterns in your data.

Stateful LSTM: For long time series, consider using a stateful LSTM where the internal state is passed from one batch to the next. This can be more effective at learning long-term dependencies than the default stateless LSTM.

Bidirectional LSTM (Bi-LSTM): A Bidirectional LSTM processes the sequence both forwards and backward. This can be particularly useful for time series with complex or bidirectional dependencies.

Hybrid Models: Combine the strengths of different models. A SARIMA-LSTM hybrid model can be created by first using SARIMA to capture linear trends and seasonality, and then using an LSTM on the residuals to model non-linear patterns. This approach can often outperform a single model.
"""