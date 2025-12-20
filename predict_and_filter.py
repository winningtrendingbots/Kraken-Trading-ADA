"""
PREDICCIÓN + FILTROS TÉCNICOS

Este script:
1. Carga el modelo LSTM entrenado
2. Descarga datos recientes de ETHUSD (1h)
3. Genera predicción (High, Low, Close)
4. Aplica filtros técnicos (RSI, ATR, Tendencia)
5. Genera señal: BUY, SELL o HOLD
6. Guarda en trading_signals.csv
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
CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

def send_telegram(msg):
    if not TELEGRAM_API or not CHAT_ID:
        print("⚠️ Telegram no configurado")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_API}/sendMessage"
        requests.post(url, data={'chat_id': CHAT_ID, 'text': msg, 'parse_mode': 'Markdown'}, timeout=10)
    except Exception as e:
        print(f"❌ Telegram: {e}")

# Clase del modelo (debe coincidir con ethusd_lstm.py)
class MultiOutputLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=256, num_layers=3,
                 output_size=3, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = self.fc1(lstm_out[:, -1, :])
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
    """
    Genera señal de trading basada en:
    - Predicción del modelo
    - RSI
    - ATR (volatilidad)
    - Tendencia
    
    Returns:
        dict con 'signal' (BUY/SELL/HOLD) y 'confidence' (0-100)
    """
    
    # Calcular cambio predicho
    pred_change_pct = ((pred_close - current_price) / current_price) * 100
    
    # Sistema de puntos para confianza
    confidence_score = 50  # Base
    signal = "HOLD"
    
    # 1. PREDICCIÓN DEL MODELO (peso: 40%)
    if pred_change_pct > 1.0:  # Predicción alcista fuerte
        signal = "BUY"
        confidence_score += min(pred_change_pct * 10, 30)
    elif pred_change_pct < -1.0:  # Predicción bajista fuerte
        signal = "SELL"
        confidence_score += min(abs(pred_change_pct) * 10, 30)
    else:
        confidence_score -= 20  # Predicción neutral
    
    # 2. RSI (peso: 20%)
    if rsi < 30:  # Sobreventa
        if signal == "BUY":
            confidence_score += 15
        elif signal == "SELL":
            confidence_score -= 15
    elif rsi > 70:  # Sobrecompra
        if signal == "SELL":
            confidence_score += 15
        elif signal == "BUY":
            confidence_score -= 15
    
    # 3. TENDENCIA (peso: 20%)
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
    
    # 4. VOLATILIDAD con ATR (peso: 10%)
    volatility = (atr / current_price) * 100
    if volatility > 2.0:  # Alta volatilidad
        confidence_score -= 10
    elif volatility < 0.5:  # Baja volatilidad
        confidence_score += 5
    
    # 5. ALINEACIÓN DE PREDICCIONES (peso: 10%)
    # Si High, Low y Close apuntan en misma dirección
    if pred_high > current_price and pred_low > current_price and pred_close > current_price:
        if signal == "BUY":
            confidence_score += 10
    elif pred_high < current_price and pred_low < current_price and pred_close < current_price:
        if signal == "SELL":
            confidence_score += 10
    
    # Limitar confianza entre 0-100
    confidence = max(0, min(100, confidence_score))
    
    # Si confianza es muy baja, cambiar a HOLD
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

def main():
    print("="*70)
    print("  🔮 PREDICCIÓN + FILTROS TÉCNICOS")
    print("="*70 + "\n")
    
    # 1. CARGAR MODELO
    model_dir = 'ETHUSD_MODELS'
    interval = '1h'
    
    if not os.path.exists(model_dir):
        error_msg = "❌ No existe ETHUSD_MODELS/. Ejecuta primero el entrenamiento."
        print(error_msg)
        send_telegram(error_msg)
        return
    
    print("📂 Cargando modelo...")
    
    try:
        # Cargar configuración
        with open(f'{model_dir}/config_{interval}.json', 'r') as f:
            config = json.load(f)
        
        seq_len = config['seq_len']
        
        # Cargar scalers
        scaler_in = joblib.load(f'{model_dir}/scaler_input_{interval}.pkl')
        scaler_out = joblib.load(f'{model_dir}/scaler_output_{interval}.pkl')
        
        # Cargar modelo
        # DESPUÉS (arreglado)
        checkpoint = torch.load(
            f'{model_dir}/ethusd_lstm_{interval}.pth', 
            map_location=torch.device('cpu'),
            weights_only=False  # Necesario para PyTorch >= 2.6
        )
        
        model = MultiOutputLSTM()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print("✅ Modelo cargado\n")
        
    except Exception as e:
        error_msg = f"❌ Error cargando modelo: {str(e)}"
        print(error_msg)
        send_telegram(error_msg)
        return
    
    # 2. DESCARGAR DATOS RECIENTES
    print("📥 Descargando datos recientes...")
    
    try:
        ticker = yf.Ticker("ETH-USD")
        df = ticker.history(period="5d", interval=interval)
        df = df.reset_index()
        df.columns = [str(c).lower() for c in df.columns]
        df.rename(columns={'date': 'time', 'datetime': 'time'}, inplace=True)
        
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
        
        df = df[['time', 'open', 'high', 'low', 'close']].tail(seq_len + 20)  # Extra para indicadores
        
        print(f"✅ {len(df)} velas descargadas\n")
        
    except Exception as e:
        error_msg = f"❌ Error descargando datos: {str(e)}"
        print(error_msg)
        send_telegram(error_msg)
        return
    
    # 3. PREPARAR SECUENCIA PARA PREDICCIÓN
    print("🔧 Preparando datos para predicción...")
    
    inp = df[['open', 'high', 'low', 'close']].values[-seq_len:]
    inp_scaled = scaler_in.transform(inp)
    
    # Convertir a tensor
    X = torch.FloatTensor(inp_scaled).unsqueeze(0)  # [1, seq_len, 4]
    
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
    print(f"Precio Actual:   ${current_price:.2f}")
    print(f"Pred High:       ${pred_high:.2f}")
    print(f"Pred Low:        ${pred_low:.2f}")
    print(f"Pred Close:      ${pred_close:.2f}")
    print(f"Cambio Pred:     {((pred_close - current_price) / current_price * 100):+.2f}%")
    print("="*70 + "\n")
    
    # 5. CALCULAR INDICADORES TÉCNICOS
    print("📊 Calculando indicadores técnicos...")
    
    rsi = calculate_rsi(df['close'].values)
    atr = calculate_atr(df)
    trend = detect_trend(df)
    
    print(f"RSI:        {rsi:.1f}")
    print(f"ATR:        ${atr:.2f}")
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
    print(f"📊 ATR:        ${result['atr']:.2f}")
    print(f"📉 Volatilidad: {result['volatility_%']:.2f}%")
    print(f"📐 Tendencia:  {result['trend']}")
    print("="*70 + "\n")
    
    # 7. GUARDAR SEÑAL
    signal_data = {
        'timestamp': datetime.now(),
        'current_price': current_price,
        'pred_high': pred_high,
        'pred_low': pred_low,
        'pred_close': pred_close,
        'pred_change_%': result['pred_change_%'],
        'atr': result['atr'],
        'volatility': result['volatility_%'],
        'trend': result['trend'],
        'signal': signal,
        'confidence': confidence,
        'rsi': result['rsi']
    }
    
    signals_file = 'trading_signals.csv'
    df_signal = pd.DataFrame([signal_data])
    
    if os.path.exists(signals_file):
        df_signal.to_csv(signals_file, mode='a', header=False, index=False)
    else:
        df_signal.to_csv(signals_file, index=False)
    
    print(f"✅ Señal guardada en {signals_file}\n")
    
    # 8. NOTIFICAR POR TELEGRAM
    emoji = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "⚪"
    
    msg = f"""
{emoji} *Nueva Señal: {signal}*

💰 Precio: ${current_price:.2f}
🔮 Pred Close: ${pred_close:.2f} ({result['pred_change_%']:+.2f}%)
🎲 Confianza: {confidence:.1f}%

📊 *Indicadores:*
   RSI: {result['rsi']:.1f}
   ATR: ${result['atr']:.2f}
   Volatilidad: {result['volatility_%']:.2f}%
   Tendencia: {result['trend']}

🔍 *Rango Predicho:*
   High: ${pred_high:.2f}
   Low: ${pred_low:.2f}
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
