import numpy as np
import pandas as pd
import yfinance as yf
import requests
import os
import time
import json
import joblib
from datetime import datetime
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# --- CONFIGURACIÓN ---
SEQ_LEN = 60            # Usar 60 velas de historia (contexto)
PREDICT_AHEAD = 1       # Predecir la siguiente vela
MODEL_FILE = "ADAUSD_MODELS/ada_lstm_robust.keras" # Usamos formato .keras moderno
DATA_FILE = "ADAUSD_1h_data.csv"
os.makedirs("ADAUSD_MODELS", exist_ok=True)

# Credenciales (desde GitHub Secrets)
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN") or os.environ.get("TELEGRAM_API")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID") or os.environ.get("CHAT_ID")

# --- 1. OBTENCIÓN DE DATOS ROBUSTA ---
def get_data():
    """Descarga datos históricos fiables usando Yahoo Finance (más estable para history)"""
    print("📥 Descargando datos históricos de ADA-USD...")
    try:
        # Descargar 60 días de datos en intervalo de 1h o 5m según prefieras
        # Para trading intradía robusto, entrenamos con 1h y ajustamos predicción
        df = yf.download("ADA-USD", period="60d", interval="1h", progress=False)
        
        if len(df) < 100:
            raise ValueError("Datos insuficientes de Yahoo Finance")
            
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns] # open, high, low, close...
        
        # Limpieza básica
        df = df[['close', 'high', 'low', 'open', 'volume']]
        
        # Guardar para referencia
        df.to_csv(DATA_FILE)
        return df
    except Exception as e:
        print(f"❌ Error descargando datos: {e}")
        return None

def get_current_price_kraken():
    """Obtiene el precio actual REAL de Kraken para la inferencia"""
    try:
        url = "https://api.kraken.com/0/public/Ticker?pair=ADAUSD"
        resp = requests.get(url, timeout=5).json()
        price = float(resp['result'][list(resp['result'].keys())[0]]['c'][0])
        return price
    except:
        return None

# --- 2. NORMALIZACIÓN POR VENTANA (LA CLAVE DEL ÉXITO) ---
def window_normalization(window_data):
    """
    Normaliza una ventana dividiendo todo por el primer valor de la ventana.
    Esto enseña al modelo el CAMBIO PORCENTUAL, no el precio absoluto.
    Retorna: (ventana_normalizada, factor_de_escala)
    """
    base_price = window_data[0]
    return (window_data / base_price) - 1, base_price

def denormalize(val, base_price):
    """Convierte la predicción del modelo de vuelta a precio real"""
    return base_price * (val + 1)

def prepare_dataset(df):
    data = df['close'].values
    X, y = [], []
    
    for i in range(len(data) - SEQ_LEN - PREDICT_AHEAD):
        window = data[i : i + SEQ_LEN]
        target = data[i + SEQ_LEN] # El precio que queremos predecir
        
        # Normalizar ventana
        norm_window, base = window_normalization(window)
        
        # Normalizar target respecto al MISMO base de la ventana
        norm_target = (target / base) - 1
        
        X.append(norm_window)
        y.append(norm_target)
        
    return np.array(X), np.array(y)

# --- 3. CONSTRUCCIÓN DEL MODELO ---
def build_model(input_shape):
    model = Sequential([
        # Bidirectional LSTM captura patrones en ambas direcciones
        Bidirectional(LSTM(50, return_sequences=True), input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1) # Predice el cambio porcentual (normalizado)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# --- 4. FUNCIONES PRINCIPALES ---

def train_job():
    df = get_data()
    if df is None: return
    
    X, y = prepare_dataset(df)
    
    # Reshape para LSTM [samples, time steps, features]
    # Aquí usamos solo 'Close' como feature para máxima estabilidad
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    print(f"🧠 Entrenando con {len(X)} secuencias...")
    
    model = build_model((X.shape[1], 1))
    
    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        X, y,
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )
    
    model.save(MODEL_FILE)
    print("✅ Modelo guardado exitosamente.")
    
    # Generar gráfica de loss
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,6))
        plt.plot(history.history['loss'], label='Loss')
        plt.title('Error del Modelo (Training Loss)')
        plt.savefig('training_loss.png')
    except:
        pass

def predict_job():
    if not os.path.exists(MODEL_FILE):
        print("❌ No hay modelo entrenado.")
        return

    # 1. Cargar datos recientes y modelo
    df = get_data()
    model = load_model(MODEL_FILE)
    
    # 2. Preparar la ÚLTIMA secuencia disponible
    last_window_closes = df['close'].values[-SEQ_LEN:]
    
    # 3. Normalizar igual que en el entrenamiento
    norm_window, base_price_window = window_normalization(last_window_closes)
    input_seq = np.reshape(norm_window, (1, SEQ_LEN, 1))
    
    # 4. Predecir
    pred_normalized = model.predict(input_seq)[0][0]
    
    # 5. Desnormalizar usando el precio base de la ventana
    predicted_price = denormalize(pred_normalized, base_price_window)
    
    # 6. Obtener precio actual real de Kraken para comparar
    current_market_price = get_current_price_kraken()
    if current_market_price is None:
        current_market_price = last_window_closes[-1]
        
    # --- CÁLCULO DE BANDAS DE PRECIO (High/Low simulados por volatilidad) ---
    # Como el modelo predice Close, estimamos High/Low usando volatilidad reciente
    recent_volatility = df['high'].iloc[-10:] - df['low'].iloc[-10:]
    avg_vol = recent_volatility.mean()
    
    pred_high = predicted_price + (avg_vol * 0.5)
    pred_low = predicted_price - (avg_vol * 0.5)

    # --- LÓGICA DE SEÑAL ---
    # Si la predicción es > 0.3% del precio actual
    threshold = 0.003
    change_pct = (predicted_price - current_market_price) / current_market_price
    
    signal = "HOLD"
    if change_pct > threshold:
        signal = "BUY"
    elif change_pct < -threshold:
        signal = "SELL"

    # --- RESULTADOS ---
    print(f"\n💡 Precio Actual: {current_market_price:.4f}")
    print(f"🔮 Predicción:   {predicted_price:.4f} ({change_pct*100:.2f}%)")
    
    output = {
        "timestamp": datetime.utcnow().isoformat(),
        "base_price": current_market_price,
        "predicted_close": round(predicted_price, 4),
        "predicted_high": round(pred_high, 4),
        "predicted_low": round(pred_low, 4),
        "signal": signal
    }
    
    with open("latest_pred.json", "w") as f:
        json.dump(output, f, indent=2)

    send_telegram_report(signal, current_market_price, predicted_price, pred_high, pred_low)

def send_telegram_report(signal, current, pred, high, low):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    
    emoji = "🚀" if signal == "BUY" else "🔻" if signal == "SELL" else "⚖️"
    msg = f"""
*ADA/USD AI Forecast* {emoji}

*Signal:* `{signal}`
*Current:* `${current:.4f}`
*Target:* `${pred:.4f}`

_Range:_
H: `${high:.4f}` | L: `${low:.4f}`
"""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"})

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--predict":
        predict_job()
    else:
        train_job()
