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
import joblib

# Utilitaires de modélisation
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools # Gardé pour référence, mais auto_arima est maintenant utilisé

## AJOUT: Bibliothèques pour l'analyse exploratoire et auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm # Pour auto_arima

# Utilitaires Keras/TensorFlow pour LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# --- Ignorer les avertissements ---
warnings.filterwarnings("ignore", category=UserWarning, module='statsmodels')
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Reproductibilité ---
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
tf.random.set_seed(SEED)

# --- Style des graphiques ---
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 7)

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

# ==============================================================================
## AJOUT: 1. BIS. ANALYSE EXPLORATOIRE DES DONNÉES (EDA)
# ==============================================================================
print("\n--- Lancement de l'Analyse Exploratoire des Données (EDA) ---")

# --- 1. Visualisation de la série temporelle ---
y.plot(title='Actifs Extérieurs Nets - Série Temporelle Complète')
plt.ylabel('Valeur AEN')
plt.xlabel('Date')
plt.show()

# --- 2. Décomposition de la série ---
# Permet de visualiser la tendance, la saisonnalité et les résidus
decomposition = seasonal_decompose(y, model='additive')
fig = decomposition.plot()
plt.suptitle('Décomposition de la série (Tendance, Saisonnalité, Résidus)', y=1.02)
plt.show()

# --- 3. Test de stationnarité (Augmented Dickey-Fuller) ---
# L'hypothèse nulle (H0) est que la série n'est pas stationnaire.
# Si p-value < 0.05, on rejette H0 -> la série est stationnaire.
adf_result = adfuller(y)
print(f'ADF Statistic: {adf_result[0]:.4f}')
print(f'p-value: {adf_result[1]:.4f}')
print('La série est stationnaire' if adf_result[1] < 0.05 else 'La série n\'est pas stationnaire')

# --- 4. Autocorrélation (ACF et PACF) ---
# Aide à identifier les ordres p, q, P, Q pour un modèle ARIMA manuel
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(y, ax=ax1, lags=36)
plot_pacf(y, ax=ax2, lags=36)
plt.show()


# --- Découpage temporel (70% Train, 15% Val, 15% Test) ---
n = len(y)
i_train = int(n * 0.70)
i_val = int(n * 0.85)
y_train, y_val, y_test = y.iloc[:i_train], y.iloc[i_train:i_val], y.iloc[i_val:]

print(f"\nSérie temporelle préparée avec {n} observations.")
print(f"Tailles des ensembles -> Train:{len(y_train)} | Validation:{len(y_val)} | Test:{len(y_test)}")
print(f"Périodes -> Train: {y_train.index.min().year}-{y_train.index.max().year} | Val: {y_val.index.min().year}-{y_val.index.max().year} | Test: {y_test.index.min().year}-{y_test.index.max().year}")

# --- Métriques d'évaluation (inchangée) ---
def evaluate_forecast(y_true, y_pred, model_name=""):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"--- Évaluation {model_name} ---")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
    return {'mae': mae, 'rmse': rmse, 'mape': mape}

# ==============================================================================
# 2. MODÉLISATION ARIMA SAISONNIER (AVEC AUTO_ARIMA)
# ==============================================================================
## AJOUT: Utilisation de pmdarima.auto_arima pour une recherche optimisée
print("\n--- Recherche du meilleur modèle SARIMA via auto_arima ---")

# auto_arima va tester différentes combinaisons de p, d, q, P, D, Q
# m=12 indique une saisonnalité annuelle (données mensuelles)
# stepwise=True rend la recherche plus rapide
auto_model = pm.auto_arima(y_train,
                           start_p=1, start_q=1,
                           test='adf',       # Utilise le test ADF pour trouver le meilleur 'd'
                           max_p=3, max_q=3, # Range max pour p et q
                           m=12,             # Période de saisonnalité
                           d=None,           # Laisse auto_arima trouver d
                           seasonal=True,    # Active la recherche saisonnière
                           start_P=0,
                           D=None,           # Laisse auto_arima trouver D
                           trace=True,       # Affiche les modèles testés
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)

print(auto_model.summary())

# Le meilleur modèle est directement disponible dans l'objet `auto_model`
best_model = auto_model
best_pdq = best_model.order
best_seasonal_pdq = best_model.seasonal_order

print(f"\nMeilleurs paramètres SARIMA trouvés par auto_arima: {best_pdq} {best_seasonal_pdq}")

## AJOUT: Analyse des résidus du meilleur modèle
# Les résidus doivent se comporter comme un bruit blanc pour que le modèle soit valide
print("\n--- Analyse des résidus du modèle SARIMA ---")
best_model.plot_diagnostics(figsize=(15, 12))
plt.show()

# Évaluation sur le jeu de test
arima_pred = best_model.predict(n_periods=len(y_test))
arima_pred.index = y_test.index # Assigner le bon index pour l'évaluation
metrics_arima = evaluate_forecast(y_test, arima_pred, "ARIMA (auto)")


# ==============================================================================
# 3. MODÉLISATION LSTM (LONG SHORT-TERM MEMORY)
# ==============================================================================
# --- Préparation des données spécifiques au LSTM (inchangée) ---
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
## AJOUT: Le modèle est maintenant dans une fonction pour faciliter le tuning
def build_lstm_model(seq_length, lstm_units=50, dropout_rate=0.2):
    """Construit un modèle LSTM séquentiel."""
    model = Sequential([
        LSTM(units=lstm_units, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(dropout_rate),
        LSTM(units=lstm_units),
        Dropout(dropout_rate),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Note: Pour un tuning complet, on utiliserait une boucle ou une bibliothèque comme KerasTuner
# pour tester différentes valeurs de `lstm_units`, `dropout_rate`, `SEQ_LENGTH`, etc.
# Par exemple: for units in [30, 50, 80]: for rate in [0.1, 0.2, 0.3]: ...

print("\n--- Entraînement du modèle LSTM ---")
model_lstm = build_lstm_model(SEQ_LENGTH)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model_lstm.fit(
    X_train, y_train_seq,
    epochs=100, batch_size=32,
    validation_data=(X_val, y_val_seq),
    callbacks=[early_stopping],
    verbose=0
)
print("Entraînement LSTM terminé.")

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

# On utilise les meilleurs paramètres trouvés par auto_arima
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
forecast_result = final_model.get_forecast(steps=n_forecast)
forecast_values = forecast_result.predicted_mean
conf_int = forecast_result.conf_int()

# Création du DataFrame de prévision
forecast_index = pd.date_range(start='2025-02-01', periods=n_forecast, freq='MS')
forecast_df = pd.DataFrame({
    'Prediction': forecast_values,
    'Lower_CI': conf_int.iloc[:, 0],
    'Upper_CI': conf_int.iloc[:, 1]
}, index=forecast_index)

## AJOUT: Sauvegarde du modèle et des paramètres
# Méthode 1: Sauvegarder les paramètres et les données (votre méthode originale)
model_deployment_info = {
    'best_pdq': best_pdq,
    'best_seasonal_pdq': best_seasonal_pdq,
    'data_tail': y.tail(36) # Sauvegarder seulement la fin des données peut suffire
}
joblib.dump(model_deployment_info, 'sarima_params.pkl')
print("Paramètres du modèle sauvegardés dans 'sarima_params.pkl'.")

# Méthode 2 (Alternative): Sauvegarder l'objet modèle complet et entraîné
# Avantage: Pas besoin de ré-entraîner lors du chargement.
# Inconvénient: Peut être sensible aux changements de version de la bibliothèque.
joblib.dump(final_model, 'sarima_model_fitted.pkl')
print("Objet modèle complet sauvegardé dans 'sarima_model_fitted.pkl'.")


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