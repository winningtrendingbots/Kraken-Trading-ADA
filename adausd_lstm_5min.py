import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import yfinance as yf
import os
import json
import requests
from datetime import datetime

# --- CONFIGURACIÓN ---
SEQ_LEN = 72       
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

# --- 1. DESCARGA DE DATOS (Copiado de tu archivo original) ---
def download_adausd(interval='1h', path=DATA_FILE):
    """
    🔥 DESCARGA DE DATOS ORIGINAL
    """
    print(f"📥 Descargando ADA-USD ({interval})...")
    try:
        # Descargamos suficiente historia para las secuencias
        df = yf.download("ADA-USD", period="60d", interval=interval, progress=False)
        if df.empty:
            return None
        
        # Mantener las columnas que pides, incluyendo Volumen
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.to_csv(path)
        return df
    except Exception as e:
        print(f"❌ Error yfinance: {e}")
        return None

# --- 2. PREPARACIÓN CON NORMALIZACIÓN RELATIVA ---
def prepare_sequences(df):
    # Usamos Open, High, Low, Close, Volume
    data = df.values 
    X, y = [], []
    
    for i in range(len(data) - SEQ_LEN):
        window = data[i:i+SEQ_LEN].copy()
        target = data[i+SEQ_LEN, 1:4].copy() # High, Low, Close
        
        # PRECIO BASE (Último cierre de la ventana)
        base_price = window[-1, 3] 
        
        # NORMALIZACIÓN: Todo es relativo al precio actual
        # Esto evita que la predicción salga "lejos"
        norm_window = (window / base_price) - 1
        norm_target = (target / base_price) - 1
        
        X.append(norm_window)
        y.append(norm_target)
        
    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y))

# --- 3. MODELO ---
class LSTMModel(nn.Module):
    def __init__(self, input_dim=5):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, 64, num_layers=2, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(64, 3) 

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# --- 4. EJECUCIÓN ---
def run():
    df = download_adausd()
    if df is None: return
    
    X, y = prepare_sequences(df)
    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"🧠 Entrenando... (Deltas)")
    model.train()
    for epoch in range(30):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    
    # PREDICCIÓN
    model.eval()
    with torch.no_grad():
        last_window = df.values[-SEQ_LEN:]
        base_price = last_window[-1, 3]
        
        input_norm = torch.FloatTensor((last_window / base_price) - 1).unsqueeze(0)
        pred_norm = model(input_norm).numpy()[0]
        
        # RECONSTRUCCIÓN (Base * (1 + Pred))
        # Esto garantiza que High/Low/Close estén pegados al Base Price
        pred_high = (pred_norm[0] + 1) * base_price
        pred_low  = (pred_norm[1] + 1) * base_price
        pred_close = (pred_norm[2] + 1) * base_price

    # SEÑAL
    diff = (pred_close - base_price) / base_price
    signal = "BUY" if diff > 0.002 else "SELL" if diff < -0.002 else "HOLD"
    
    # Tu mensaje de Telegram
    emoji = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "⚪"
    msg = f"""
✅ *Predicción ADA/USD (Relativa)* {emoji}

*Signal:* `{signal}`
*Base Price:* `${base_price:.4f}`

📊 *Futuro (1h):*
  • High:  `${pred_high:.4f}`
  • Low:   `${pred_low:.4f}`
  • Close: `${pred_close:.4f}`

_Nota: Predicción basada en variación % sobre precio base._
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
    run()
