"""
PREDICCIÓN + FILTROS TÉCNICOS - VERSIÓN CORREGIDA
✅ Sintaxis correcta (SyntaxError fixed)
✅ 4 decimales en predicciones
✅ CSV unificado de tracking
✅ Conversión de tipos numpy → Python
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
import os
from datetime import datetime
import yfinance as yf
import requests

# Telegram
TELEGRAM_API = os.environ.get('TELEGRAM_API', '')
CHAT_ID = os.environ.get('CHAT_ID', '')

# 🆕 Archivo de tracking unificado
PREDICTION_TRACKER_FILE = 'prediction_tracker.csv'

def send_telegram(msg):
    if not TELEGRAM_API or not CHAT_ID:
        print("⚠️ Telegram no configurado")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_API}/sendMessage"
        requests.post(url, data={'chat_id': CHAT_ID, 'text': msg, 'parse_mode': 'Markdown'}, timeout=10)
    except Exception as e:
        print(f"❌ Telegram: {e}")


class MultiOutputLSTM(nn.Module):
    """Versión con BatchNorm - SINCRONIZADA con entrenamiento"""
    def __init__(self, input_size=4, hidden_size=192, num_layers=2,
                 output_size=3, dropout=0.35):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, 
                           dropout=dropout if num_layers > 1 else 0)
        
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.bn1(x)
        x = self.fc1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)


def calculate_rsi(prices, period=14):
    """Calcula RSI"""
    delta = pd.Series(prices).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

def calculate_atr(df, period=14):
    """Calcula ATR (Average True Range)"""
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)
    
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean().iloc[-1]
    
    return atr

def detect_trend(df, window=20):
    """Detecta tendencia con SMA"""
    sma = df['close'].rolling(window=window).mean()
    current_price = df['close'].iloc[-1]
    sma_value = sma.iloc[-1]
    
    if current_price > sma_value * 1.02:
        return "UPTREND"
    elif current_price < sma_value * 0.98:
        return "DOWNTREND"
    else:
        return "SIDEWAYS"

def generate_signal(pred_high, pred_low, pred_close, current_price, rsi, atr, trend):
    """Genera señal de trading"""
    
    pred_change_pct = ((pred_close - current_price) / current_price) * 100
    
    confidence_score = 50
    signal = "HOLD"
    
    # 1. PREDICCIÓN DEL MODELO
    if pred_change_pct > 1.0:
        signal = "BUY"
        confidence_score += min(pred_change_pct * 10, 30)
    elif pred_change_pct < -1.0:
        signal = "SELL"
        confidence_score += min(abs(pred_change_pct) * 10, 30)
    else:
        confidence_score -= 20
    
    # 2. RSI
    if rsi < 30:
        if signal == "BUY":
            confidence_score += 15
        elif signal == "SELL":
            confidence_score -= 15
    elif rsi > 70:
        if signal == "SELL":
            confidence_score += 15
        elif signal == "BUY":
            confidence_score -= 15
    
    # 3. TENDENCIA
    if trend == "UPTREND":
        if signal == "BUY":
            confidence_score += 10
        elif signal == "SELL":
            confidence_score -= 10
    elif trend == "DOWNTREND":
        if signal == "SELL":
            confidence_score += 10
        elif signal == "BUY":
            confidence_score -= 10
    
    # 4. VOLATILIDAD
    volatility = (atr / current_price) * 100
    if volatility > 2.0:
        confidence_score -= 10
    elif volatility < 0.5:
        confidence_score += 5
    
    # 5. ALINEACIÓN
    if pred_high > current_price and pred_low > current_price and pred_close > current_price:
        if signal == "BUY":
            confidence_score += 10
    elif pred_high < current_price and pred_low < current_price and pred_close < current_price:
        if signal == "SELL":
            confidence_score += 10
    
    confidence = max(0, min(100, confidence_score))
    
    if confidence < 55:
        signal = "HOLD"
        confidence = 50
    
    return {
        'signal': signal,
        'confidence': confidence,
        'pred_change_%': pred_change_pct,
        'rsi': rsi,
        'atr': atr,
        'volatility_%': volatility,
        'trend': trend
    }


def save_to_prediction_tracker(timestamp, current_price, pred_high, pred_low, pred_close, 
                               signal, confidence, rsi, atr, trend):
    """
    🆕 NUEVO: Guarda predicción en CSV de tracking unificado
    ✅ FIX: Sintaxis correcta + Conversión de tipos
    """
    
    # Calcular cambios predichos
    pred_high_change = ((pred_high - current_price) / current_price) * 100
    pred_low_change = ((pred_low - current_price) / current_price) * 100
    pred_close_change = ((pred_close - current_price) / current_price) * 100
    
    # Rango predicho
    pred_range = pred_high - pred_low
    pred_range_pct = (pred_range / current_price) * 100
    
    # ✅ FIX: Conversión explícita de tipos numpy → Python
    tracking_data = {
        'timestamp': timestamp,
        'current_price': float(round(current_price, 4)),
        'pred_high': float(round(pred_high, 4)),
        'pred_low': float(round(pred_low, 4)),
        'pred_close': float(round(pred_close, 4)),
        'pred_high_change_%': float(round(pred_high_change, 2)),
        'pred_low_change_%': float(round(pred_low_change, 2)),
        'pred_close_change_%': float(round(pred_close_change, 2)),
        'pred_range': float(round(pred_range, 4)),
        'pred_range_%': float(round(pred_range_pct, 2)),
        'signal': str(signal),
        'confidence': float(round(confidence, 1)),
        'rsi': float(round(rsi, 1)),
        'atr': float(round(atr, 4)),
        'trend': str(trend),
        'order_opened': 'NO',
        'order_id': None,
        'entry_price': None,
        'exit_price': None,
        'pnl_usd': None,
        'pnl_%': None,
        'close_reason': None,
        'actual_high': None,
        'actual_low': None,
        'actual_close': None,
        'pred_accuracy_%': None
    }
    
    df_track = pd.DataFrame([tracking_data])
    
    if os.path.exists(PREDICTION_TRACKER_FILE):
        df_track.to_csv(PREDICTION_TRACKER_FILE, mode='a', header=False, index=False)
    else:
        df_track.to_csv(PREDICTION_TRACKER_FILE, index=False)
    
    print(f"✅ Predicción guardada en {PREDICTION_TRACKER_FILE}")


def main():
    print("="*70)
    print("  🔮 PREDICCIÓN + FILTROS TÉCNICOS (4 DECIMALES)")
    print("="*70 + "\n")
    
    # 1. CARGAR MODELO
    model_dir = 'ADAUSD_MODELS'
    interval = '1h'
    
    if not os.path.exists(model_dir):
        error_msg = "❌ No existe ADAUSD_MODELS/. Ejecuta primero el entrenamiento."
        print(error_msg)
        send_telegram(error_msg)
        return
    
    print("📂 Cargando modelo...")
    
    try:
        with open(f'{model_dir}/config_{interval}.json', 'r') as f:
            config = json.load(f)
        
        seq_len = config['seq_len']
        
        scaler_in = joblib.load(f'{model_dir}/scaler_input_{interval}.pkl')
        scaler_out = joblib.load(f'{model_dir}/scaler_output_{interval}.pkl')
        
        checkpoint = torch.load(
            f'{model_dir}/adausd_lstm_{interval}.pth', 
            map_location=torch.device('cpu'),
            weights_only=False
        )
        
        model_config = checkpoint.get('config', {})
        hidden_size = model_config.get('hidden', 192)
        num_layers = model_config.get('layers', 2)
        
        print(f"📋 Configuración del modelo:")
        print(f"   Hidden Size: {hidden_size}")
        print(f"   Num Layers: {num_layers}")
        print(f"   Seq Length: {seq_len}")
        
        model = MultiOutputLSTM(
            input_size=4,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=3,
            dropout=0.35
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print("✅ Modelo cargado correctamente\n")
        
    except Exception as e:
        error_msg = f"❌ Error cargando modelo: {str(e)}"
        print(error_msg)
        send_telegram(error_msg)
        return
    
    # 2. DESCARGAR DATOS RECIENTES
    print("📥 Descargando datos recientes...")
    
    try:
        ticker = yf.Ticker("ADA-USD")
        df = ticker.history(period="5d", interval=interval)
        df = df.reset_index()
        df.columns = [str(c).lower() for c in df.columns]
        df.rename(columns={'date': 'time', 'datetime': 'time'}, inplace=True)
        
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
        
        df = df[['time', 'open', 'high', 'low', 'close']].tail(seq_len + 20)
        
        print(f"✅ {len(df)} velas descargadas\n")
        
    except Exception as e:
        error_msg = f"❌ Error descargando datos: {str(e)}"
        print(error_msg)
        send_telegram(error_msg)
        return
    
    # 3. PREPARAR SECUENCIA
    print("🔧 Preparando datos para predicción...")
    
    inp = df[['open', 'high', 'low', 'close']].values[-seq_len:]
    inp_scaled = scaler_in.transform(inp)
    X = torch.FloatTensor(inp_scaled).unsqueeze(0)
    
    # 4. GENERAR PREDICCIÓN
    print("🔮 Generando predicción...\n")
    
    with torch.no_grad():
        pred = model(X)
    
    pred_denorm = scaler_out.inverse_transform(pred.numpy())
    pred_high, pred_low, pred_close = pred_denorm[0]
    
    current_price = df['close'].iloc[-1]
    
    print("="*70)
    print("  PREDICCIÓN")
    print("="*70)
    print(f"Precio Actual:   ${current_price:.4f}")
    print(f"Pred High:       ${pred_high:.4f}")
    print(f"Pred Low:        ${pred_low:.4f}")
    print(f"Pred Close:      ${pred_close:.4f}")
    print(f"Cambio Pred:     {((pred_close - current_price) / current_price * 100):+.2f}%")
    print("="*70 + "\n")
    
    # 5. CALCULAR INDICADORES
    print("📊 Calculando indicadores técnicos...")
    
    rsi = calculate_rsi(df['close'].values)
    atr = calculate_atr(df)
    trend = detect_trend(df)
    
    print(f"RSI:        {rsi:.1f}")
    print(f"ATR:        ${atr:.4f}")
    print(f"Tendencia:  {trend}\n")
    
    # 6. GENERAR SEÑAL
    print("🎯 Generando señal de trading...")
    
    result = generate_signal(
        pred_high, pred_low, pred_close,
        current_price, rsi, atr, trend
    )
    
    signal = result['signal']
    confidence = result['confidence']
    
    print("="*70)
    print("  SEÑAL DE TRADING")
    print("="*70)
    print(f"🚦 Señal:      {signal}")
    print(f"🎲 Confianza:  {confidence:.1f}%")
    print(f"📈 RSI:        {result['rsi']:.1f}")
    print(f"📊 ATR:        ${result['atr']:.4f}")
    print(f"📉 Volatilidad: {result['volatility_%']:.2f}%")
    print(f"📍 Tendencia:  {result['trend']}")
    print("="*70 + "\n")
    
    # 7. GUARDAR EN prediction_tracker.csv
    timestamp = datetime.now()
    save_to_prediction_tracker(
        timestamp, current_price, pred_high, pred_low, pred_close,
        signal, confidence, rsi, atr, trend
    )
    
    # 8. GUARDAR SEÑAL (CSV original)
    # ✅ FIX: Conversión de tipos
    signal_data = {
        'timestamp': timestamp,
        'current_price': float(round(current_price, 4)),
        'pred_high': float(round(pred_high, 4)),
        'pred_low': float(round(pred_low, 4)),
        'pred_close': float(round(pred_close, 4)),
        'pred_change_%': float(round(result['pred_change_%'], 2)),
        'atr': float(round(result['atr'], 4)),
        'volatility': float(round(result['volatility_%'], 2)),
        'trend': str(result['trend']),
        'signal': str(signal),
        'confidence': float(round(confidence, 1)),
        'rsi': float(round(result['rsi'], 1))
    }
    
    signals_file = 'trading_signals.csv'
    df_signal = pd.DataFrame([signal_data])
    
    if os.path.exists(signals_file):
        df_signal.to_csv(signals_file, mode='a', header=False, index=False)
    else:
        df_signal.to_csv(signals_file, index=False)
    
    print(f"✅ Señal guardada en {signals_file}\n")
    
    # 9. NOTIFICAR TELEGRAM
    emoji = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "⚪"
    
    msg = f"""
{emoji} *Nueva Señal: {signal}*

💰 Precio: ${current_price:.4f}
🔮 Pred Close: ${pred_close:.4f} ({result['pred_change_%']:+.2f}%)
🎲 Confianza: {confidence:.1f}%

📊 *Indicadores:*
   RSI: {result['rsi']:.1f}
   ATR: ${result['atr']:.4f}
   Volatilidad: {result['volatility_%']:.2f}%
   Tendencia: {result['trend']}

📍 *Rango Predicho:*
   High: ${pred_high:.4f}
   Low: ${pred_low:.4f}
"""
    
    send_telegram(msg)
    
    print("="*70)
    print("  ✅ PREDICCIÓN COMPLETADA")
    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        print(error_msg)
        send_telegram(error_msg)
        raise
