import pandas as pd
import numpy as np
import requests
import json
import os
import joblib
import time
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as keras

# --- CONFIGURACIÓN ---
SEQ_LEN = 72       # 6 horas de contexto (5min * 72)
PRED_HORIZON = 1   # Predecir la siguiente vela
MODEL_DIR = "ADAUSD_MODELS"
DATA_FILE = "ADAUSD_1h_data.csv"
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.pkl")
MODEL_FILE = os.path.join(MODEL_DIR, "ada_lstm_model.h5")

# Configuración Telegram (Reemplaza con tus datos o usa secrets)
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

os.makedirs(MODEL_DIR, exist_ok=True)

# --- 1. OBTENCIÓN DE DATOS (Kraken) ---
def fetch_kraken_data(pair="ADAUSD", interval=5):
    url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval={interval}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            print("Error Kraken:", data["error"])
            return None
        
        # Parsear
        ohlc = data["result"][list(data["result"].keys())[0]]
        df = pd.DataFrame(ohlc, columns=["time", "open", "high", "low", "close", "vwap", "vol", "count"])
        df["time"] = pd.to_datetime(df["time"], unit='s')
        df = df[["time", "open", "high", "low", "close", "vol"]].astype({
            "open": float, "high": float, "low": float, "close": float, "vol": float
        })
        # Ordenar por tiempo
        df = df.sort_values("time").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"Error descargando datos: {e}")
        return None

# --- 2. PREPARACIÓN DE DATOS (CON DELTAS) ---
def prepare_data(df):
    """
    Normaliza inputs y genera targets basados en cambio porcentual 
    para anclar la predicción al precio actual.
    """
    df = df.copy()
    
    # Inputs: Usamos precios normalizados para que el LSTM vea la "forma" del gráfico
    # Nota: Podríamos usar log-returns para inputs también, pero MinMax funciona bien si se reentrena diario.
    feature_cols = ["open", "high", "low", "close", "vol"]
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[feature_cols])
    
    # Guardar scaler para inferencia futura
    joblib.dump(scaler, SCALER_FILE)
    
    X, y = [], []
    
    # Generar secuencias
    # Target: Cambio porcentual respecto al Close de la última vela de la secuencia (t)
    # y_high = (High(t+1) - Close(t)) / Close(t)
    # y_low  = (Low(t+1) - Close(t)) / Close(t)
    # y_close= (Close(t+1) - Close(t)) / Close(t)
    
    data_values = df[feature_cols].values
    
    for i in range(SEQ_LEN, len(df) - PRED_HORIZON):
        # Input: ventana de SEQ_LEN
        X.append(scaled_data[i-SEQ_LEN : i])
        
        current_close = data_values[i-1, 3] # Close de la última vela de entrada
        
        next_high = data_values[i, 1]
        next_low = data_values[i, 2]
        next_close = data_values[i, 3]
        
        # Calcular Deltas (Porcentaje de cambio)
        delta_high = (next_high - current_close) / current_close
        delta_low = (next_low - current_close) / current_close
        delta_close = (next_close - current_close) / current_close
        
        y.append([delta_high, delta_low, delta_close])
        
    return np.array(X), np.array(y), scaler

# --- 3. ENTRENAMIENTO ---
class ConstrainedMSELoss(keras.losses.Loss):
    def call(self, y_true, y_pred):
        # y_pred = [delta_high, delta_low, delta_close]
        high_pred = y_pred[:, 0]
        low_pred = y_pred[:, 1]
        
        # MSE estándar
        mse = keras.losses.mean_squared_error(y_true, y_pred)
        
        # Penalización si High < Low (lógica sigue aplicando a los deltas)
        # Si delta_high < delta_low, añadimos penalización
        penalty = keras.backend.mean(keras.backend.maximum(0.0, low_pred - high_pred))
        
        return mse + penalty

def build_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(3)  # [delta_high, delta_low, delta_close]
    ])
    model.compile(optimizer="adam", loss=ConstrainedMSELoss())
    return model

def train_job():
    print("Iniciando entrenamiento...")
    df = fetch_kraken_data()
    if df is None or len(df) < SEQ_LEN + 10:
        print("Datos insuficientes.")
        return False
    
    # Guardar CSV crudo para debug/historial
    df.to_csv(DATA_FILE, index=False)
    
    X, y, scaler = prepare_data(df)
    
    # Split
    split = int(0.9 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    model = build_model((X_train.shape[1], X_train.shape[2]))
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=30,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )
    
    model.save(MODEL_FILE)
    print(f"Modelo guardado en {MODEL_FILE}")
    
    # Generar gráfica simple de loss
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Training Loss (Delta Prediction)')
    plt.legend()
    plt.savefig('training_loss.png')
    
    return True

# --- 4. PREDICCIÓN E INFERENCIA ---
def send_telegram_msg(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram creds no configuradas.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print(f"Error enviando Telegram: {e}")

def predict_next_candle():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
        print("Modelo o Scaler no encontrados. Entrena primero.")
        return

    # Cargar modelo y scaler
    custom_objects = {"ConstrainedMSELoss": ConstrainedMSELoss}
    model = load_model(MODEL_FILE, custom_objects=custom_objects)
    scaler = joblib.load(SCALER_FILE)
    
    # Obtener datos recientes
    df = fetch_kraken_data()
    if df is None: return

    # Preparar última secuencia
    last_sequence = df.iloc[-SEQ_LEN:][["open", "high", "low", "close", "vol"]]
    last_close_price = last_sequence.iloc[-1]["close"] # EL ANCLA
    
    # Escalar
    input_seq = scaler.transform(last_sequence)
    input_seq = input_seq.reshape(1, SEQ_LEN, 5)
    
    # Predecir (Output son Deltas)
    pred_deltas = model.predict(input_seq)[0] # [d_high, d_low, d_close]
    
    # Reconstruir precios absolutos
    pred_high = last_close_price * (1 + pred_deltas[0])
    pred_low  = last_close_price * (1 + pred_deltas[1])
    pred_close= last_close_price * (1 + pred_deltas[2])
    
    # Lógica de Trading Básica
    signal = "HOLD"
    # Si predice subida significativa (> 0.2%) y el low no rompe mucho
    if pred_close > last_close_price * 1.002: 
        signal = "BUY"
    # Si predice bajada
    elif pred_close < last_close_price * 0.998:
        signal = "SELL"
        
    # Guardar señal para el bot de trading (kraken_trader.py)
    # Mantenemos compatibilidad con tu estructura JSON
    signal_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "base_price": round(last_close_price, 4),
        "predicted_high": round(pred_high, 4),
        "predicted_low": round(pred_low, 4),
        "predicted_close": round(pred_close, 4),
        "signal": signal,
        # Stop loss sugerido: un poco por debajo del Low predicho si es BUY
        "stop_loss": round(pred_low * 0.995, 4) if signal == "BUY" else round(pred_high * 1.005, 4),
        "take_profit": round(pred_high, 4) if signal == "BUY" else round(pred_low, 4)
    }
    
    with open("latest_pred.json", "w") as f:
        json.dump(signal_data, f, indent=4)
        
    # --- MENSAJE TELEGRAM (Formato Original) ---
    emoji = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "⚪"
    msg = f"""
*ADA/USD AI Prediction (5m)* {emoji}

*Signal:* `{signal}`
*Base Price:* `${last_close_price:.4f}`

*Predictions (Next Candle):*
High:  `${pred_high:.4f}`
Low:   `${pred_low:.4f}`
Close: `${pred_close:.4f}`

_Confidence is based on constrained LSTM model._
"""
    send_telegram_msg(msg)
    print("Predicción completada y notificada.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--predict":
        predict_next_candle()
    else:
        train_job()
