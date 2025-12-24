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
SEQ_LEN = 72       # Contexto de 72 horas
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

# --- 1. DESCARGA DE DATOS (Tu función original) ---
def download_adausd(interval='1h', path=DATA_FILE):
    try:
        df = yf.download("ADA-USD", period="60d", interval=interval, progress=False)
        if df.empty: return None
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.to_csv(path)
        return df
    except Exception as e:
        print(f"Error: {e}"); return None

# --- 2. PREPARACIÓN CON NORMALIZACIÓN RELATIVA ---
def prepare_sequences(df):
    data = df.values # [O, H, L, C, V]
    X, y = [], []
    
    for i in range(len(data) - SEQ_LEN):
        window = data[i:i+SEQ_LEN].copy()
        # Target: High, Low, Close, Volume de la siguiente vela
        target = data[i+SEQ_LEN, [1, 2, 3, 4]].copy() 
        
        # Normalización Relativa (Evita que el precio se "escape")
        base_price = window[-1, 3]  # Último Close
        base_vol = window[-1, 4]    # Último Volume
        
        # Precios / base_price | Volumen / base_vol
        norm_window = window.copy()
        norm_window[:, :4] = (norm_window[:, :4] / base_price) - 1
        norm_window[:, 4] = (norm_window[:, 4] / base_vol) - 1
        
        norm_target = target.copy()
        norm_target[:3] = (norm_target[:3] / base_price) - 1
        norm_target[3] = (norm_target[3] / base_vol) - 1
        
        X.append(norm_window)
        y.append(norm_target)
        
    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y))

# --- 3. MODELO LSTM (5 Entradas -> 4 Salidas) ---
class MultiTaskLSTM(nn.Module):
    def __init__(self):
        super(MultiTaskLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=5, hidden_size=128, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(128, 4) # H, L, C, V

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# --- 4. LÓGICA DE TRADING AVANZADA (Trend & Divergence) ---
def analyze_advanced_signal(current_p, pred_p, current_v, pred_v):
    price_change = (pred_p - current_p) / current_p
    vol_change = (pred_v - current_v) / current_v
    
    signal = "HOLD"
    reason = "Sin tendencia clara"

    # 1. Trend Confirmation & Breakout Validation
    if price_change > 0.005:
        if vol_change > 0.10: # Confirmación por volumen (+10%)
            signal = "BUY"
            reason = "Ruptura Alcista Validada (High Volume)"
        else:
            reason = "Subida con volumen bajo (Posible Bull Trap)"
            
    elif price_change < -0.005:
        if vol_change > 0.10:
            signal = "SELL"
            reason = "Presión de Venta Confirmada"
        else:
            reason = "Bajada con volumen bajo (Soporte Débil)"

    # 2. Divergencia
    if price_change > 0.01 and vol_change < -0.20:
        signal = "HOLD"
        reason = "Divergencia: Precio sube, Volumen cae (Agotamiento)"

    return signal, reason

# --- 5. EJECUCIÓN ---
def run():
    df = download_adausd()
    if df is None: return
    
    X, y = prepare_sequences(df)
    model = MultiTaskLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print("🚀 Entrenando Modelo Multivariante...")
    model.train()
    for e in range(40):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward(); optimizer.step()

    # Predicción
    model.eval()
    with torch.no_grad():
        last_window = df.values[-SEQ_LEN:]
        base_p = last_window[-1, 3]
        base_v = last_window[-1, 4]
        
        # Escalar input
        input_norm = last_window.copy()
        input_norm[:, :4] = (input_norm[:, :4] / base_p) - 1
        input_norm[:, 4] = (input_norm[:, 4] / base_v) - 1
        input_tensor = torch.FloatTensor(input_norm).unsqueeze(0)
        
        pred = model(input_tensor).numpy()[0]
        
        # Desescalar
        p_high = (pred[0] + 1) * base_p
        p_low  = (pred[1] + 1) * base_p
        p_close = (pred[2] + 1) * base_p
        p_vol  = (pred[3] + 1) * base_v

    # Análisis avanzado
    signal, reason = analyze_advanced_signal(base_p, p_close, base_v, p_vol)
    
    # Telegram
    emoji = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "⚪"
    msg = f"""
{emoji} *AI ADA/USD Report*
*Señal:* `{signal}`
*Razón:* _{reason}_

💰 *Precio:* `${base_p:.4f}` → `${p_close:.4f}`
📊 *Rango:* `${p_low:.4f}` - `${p_high:.4f}`
📈 *Volumen:* `{base_v:,.0f}` → `{p_vol:,.0f}`
"""
    send_telegram(msg)
    
    # Guardar modelo y última predicción
    torch.save(model.state_dict(), f"{MODEL_DIR}/adausd_lstm_5min.pth")
    with open("latest_pred.json", "w") as f:
        json.dump({"signal": signal, "base_price": base_p, "predicted_close": p_close}, f)

if __name__ == "__main__":
    run()
