import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import yfinance as yf
import os
import json
import joblib
import requests
import time
from datetime import datetime

# --- CONFIGURACIÓN ---
SEQ_LEN = 72       # 72 velas de contexto
MODEL_DIR = "ADAUSD_MODELS"
DATA_FILE = "ADAUSD_1h_data.csv"
os.makedirs(MODEL_DIR, exist_ok=True)

# Telegram
TELEGRAM_API = os.environ.get('TELEGRAM_API', '')
CHAT_ID = os.environ.get('CHAT_ID', '')

def send_telegram(message):
    if not TELEGRAM_API or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_API}/sendMessage"
    requests.post(url, data={'chat_id': CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}, timeout=10)

# --- 1. DESCARGA DE DATOS (yfinance con Volumen) ---
def download_data():
    print("📥 Descargando datos de yfinance...")
    # Descargamos 60 días para tener suficiente histórico para entrenar
    df = yf.download("ADA-USD", period="60d", interval="1h", progress=False)
    if df.empty: return None
    
    # Nos aseguramos de tener: Open, High, Low, Close, Volume
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.to_csv(DATA_FILE)
    return df

# --- 2. NORMALIZACIÓN POR VENTANA ---
def prepare_sequences(df):
    data = df.values # [Open, High, Low, Close, Volume]
    X, y = [], []
    
    for i in range(len(data) - SEQ_LEN):
        window = data[i:i+SEQ_LEN].copy()
        target = data[i+SEQ_LEN, 1:4].copy() # Queremos High, Low, Close (indices 1,2,3)
        
        # El precio de anclaje es el Close de la última vela de la ventana
        base_price = window[-1, 3] 
        
        # Normalizamos la ventana dividiendo por el precio base
        # (Esto hace que el modelo aprenda porcentajes de movimiento)
        norm_window = (window / base_price) - 1
        norm_target = (target / base_price) - 1
        
        X.append(norm_window)
        y.append(norm_target)
        
    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y))

# --- 3. MODELO LSTM (PyTorch) ---
class LSTMModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 3) # Salida: [High, Low, Close] normalizados

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# --- 4. ENTRENAMIENTO Y PREDICCIÓN ---
def train_and_predict():
    df = download_data()
    if df is None: return
    
    X, y = prepare_sequences(df)
    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"🧠 Entrenando con {X.shape[0]} secuencias...")
    model.train()
    for epoch in range(20): # Entrenamiento rápido para cada ejecución
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    
    # Guardar modelo
    torch.save(model.state_dict(), f"{MODEL_DIR}/ada_pytorch_5min.pth")
    
    # --- PREDICCIÓN ---
    model.eval()
    with torch.no_grad():
        last_window = df.values[-SEQ_LEN:]
        base_price = last_window[-1, 3] # Precio de cierre actual
        
        input_norm = (last_window / base_price) - 1
        input_tensor = torch.FloatTensor(input_norm).unsqueeze(0)
        
        pred_norm = model(input_tensor).numpy()[0]
        
        # Desnormalizar: (Norm + 1) * Base
        pred_high = (pred_norm[0] + 1) * base_price
        pred_low  = (pred_norm[1] + 1) * base_price
        pred_close = (pred_norm[2] + 1) * base_price
        
    # --- LÓGICA DE TRADING ---
    signal = "HOLD"
    if pred_close > base_price * 1.002: signal = "BUY"
    elif pred_close < base_price * 0.998: signal = "SELL"
    
    # Telegram (tu formato)
    emoji = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "⚪"
    msg = f"""
✅ *Nueva Predicción ADA/USD* {emoji}

*Signal:* `{signal}`
*Base Price:* `${base_price:.4f}`

📊 *Predicciones:*
  • High:  `${pred_high:.4f}`
  • Low:   `${pred_low:.4f}`
  • Close: `${pred_close:.4f}`

⏱️ _{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_
"""
    send_telegram(msg)
    
    # Guardar para el trader
    with open("latest_pred.json", "w") as f:
        json.dump({
            "base_price": float(base_price),
            "predicted_high": float(pred_high),
            "predicted_low": float(pred_low),
            "predicted_close": float(pred_close),
            "signal": signal
        }, f)

if __name__ == "__main__":
    train_and_predict()
