# train_model.py
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Cargar datos
df = pd.read_csv('btc_usdt_data.csv', index_col='timestamp', parse_dates=True)

# Calcular indicadores técnicos
df['SMA'] = ta.sma(df['close'], length=14)
df['RSI'] = ta.rsi(df['close'], length=14)
df['EMA'] = ta.ema(df['close'], length=14)
df['MACD'] = ta.macd(df['close'])['MACD']
df['Bollinger_High'] = ta.bbands(df['close'])['BBL_14']
df['Bollinger_Low'] = ta.bbands(df['close'])['BBU_14']

# Eliminar filas con valores nulos
df.dropna(inplace=True)

# Crear variable objetivo (Target): 1 si el precio sube, 0 si baja
df['Target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)

# Preparar datos para el modelo
features = df[['SMA', 'RSI', 'EMA', 'MACD', 'Bollinger_High', 'Bollinger_Low']]
target = df['Target']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X_train, y_train)

# Guardar el modelo y el escalador
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Modelo entrenado y guardado exitosamente.")

