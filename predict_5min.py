"""
PREDICCIÓN 5 MINUTOS - Sistema de Trading Rápido

✅ Usa modelo 1h pero predice cambios de corto plazo
✅ Señales cada 5 minutos
✅ Filtros técnicos adaptados a 5min
✅ Integración con trading_orchestrator.py
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
import os
from datetime import datetime, timedelta
import yfinance as yf
import requests

# Telegram
TELEGRAM_API = os.environ.get('TELEGRAM_API', '')
CHAT_ID = os.environ.get('CHAT_ID', '')

# Archivo de señales
SIGNALS_FILE = 'trading_signals.csv'
PREDICTION_TRACKER_FILE = 'prediction_tracker.csv'

def send_telegram(msg):
    if not TELEGRAM_API or not CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_API}/sendMessage"
        requests.post(url, data={'chat_id': CHAT_ID, 'text': msg, 'parse_mode': 'Markdown'}, timeout=10)
    except:
        pass

class MultiOutputLSTM(nn.Module):
    """Modelo LSTM - Sincronizado con entrenamiento"""
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
    """RSI adaptado a timeframe corto"""
    delta = pd.Series(prices).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not rsi.empty else 50

def calculate_atr(df, period=14):
    """ATR para volatilidad"""
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)
    
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean().iloc[-1]
    return atr if not pd.isna(atr) else 0.01

def detect_momentum_5min(df):
    """
    🔥 Detector de momentum para 5 minutos
    Más sensible que el de 1h
    """
    if len(df) < 12:
        return "NEUTRAL"
    
    # Últimas 12 velas (1 hora en 5min)
    recent = df.tail(12)
    
    price_change = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]
    
    # Contar velas alcistas vs bajistas
    bullish = (recent['close'] > recent['open']).sum()
    bearish = (recent['close'] < recent['open']).sum()
    
    # SMA rápida (12 periodos = 1h)
    sma_fast = recent['close'].mean()
    current = recent['close'].iloc[-1]
    
    if current > sma_fast * 1.005 and bullish > bearish and price_change > 0.01:
        return "STRONG_UP"
    elif current > sma_fast * 1.002 and bullish >= bearish:
        return "UP"
    elif current < sma_fast * 0.995 and bearish > bullish and price_change < -0.01:
        return "STRONG_DOWN"
    elif current < sma_fast * 0.998 and bearish >= bullish:
        return "DOWN"
    else:
        return "NEUTRAL"

def check_recent_signals():
    """
    🔥 Verifica si ya hay una señal reciente (< 5 min)
    Evita señales duplicadas
    """
    if not os.path.exists(SIGNALS_FILE):
        return False
    
    try:
        df = pd.read_csv(SIGNALS_FILE)
        if len(df) == 0:
            return False
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        last_signal = df.iloc[-1]
        
        time_diff = datetime.now() - last_signal['timestamp']
        
        # Si hay señal en los últimos 4 minutos, skip
        if time_diff.total_seconds() < 240:
            print(f"⏸️ Señal reciente hace {time_diff.total_seconds():.0f}s - Skip")
            return True
        
        return False
        
    except Exception as e:
        print(f"⚠️ Error verificando señales: {e}")
        return False

def generate_signal_5min(pred_high, pred_low, pred_close, current_price, rsi, atr, momentum):
    """
    🎯 Genera señal optimizada para 5 minutos
    Más agresivo pero con filtros fuertes
    """
    # Cambio predicho (proyección a próximos 5-15 min)
    pred_change_pct = ((pred_close - current_price) / current_price) * 100
    
    # Rango predicho
    pred_range_pct = ((pred_high - pred_low) / current_price) * 100
    
    confidence_score = 50
    signal = "HOLD"
    
    # 1. PREDICCIÓN DEL MODELO (más peso)
    if pred_change_pct > 0.5:  # ⬇️ Umbral más bajo para 5min
        signal = "BUY"
        confidence_score += min(pred_change_pct * 15, 35)
    elif pred_change_pct < -0.5:
        signal = "SELL"
        confidence_score += min(abs(pred_change_pct) * 15, 35)
    else:
        confidence_score -= 15
    
    # 2. MOMENTUM (crítico en 5min)
    if momentum == "STRONG_UP":
        if signal == "BUY":
            confidence_score += 20
        elif signal == "SELL":
            confidence_score -= 20
    elif momentum == "UP":
        if signal == "BUY":
            confidence_score += 10
    elif momentum == "STRONG_DOWN":
        if signal == "SELL":
            confidence_score += 20
        elif signal == "BUY":
            confidence_score -= 20
    elif momentum == "DOWN":
        if signal == "SELL":
            confidence_score += 10
    
    # 3. RSI (sobrecompra/sobreventa)
    if rsi < 35:  # Más agresivo
        if signal == "BUY":
            confidence_score += 15
        elif signal == "SELL":
            confidence_score -= 10
    elif rsi > 65:
        if signal == "SELL":
            confidence_score += 15
        elif signal == "BUY":
            confidence_score -= 10
    
    # 4. VOLATILIDAD (controla riesgo)
    volatility = (atr / current_price) * 100
    
    if volatility > 2.5:  # Alta volatilidad
        confidence_score -= 15
        print(f"⚠️ Alta volatilidad: {volatility:.2f}%")
    elif volatility < 0.3:  # Muy baja
        confidence_score -= 10
        print(f"⚠️ Volatilidad muy baja: {volatility:.2f}%")
    
    # 5. COHERENCIA DE PREDICCIONES
    if pred_high > current_price and pred_low > current_price:
        if signal == "BUY":
            confidence_score += 10
    elif pred_high < current_price and pred_low < current_price:
        if signal == "SELL":
            confidence_score += 10
    
    # 6. RANGO PREDICHO (sanity check)
    if pred_range_pct < 0.3:
        confidence_score -= 15
        print(f"⚠️ Rango muy pequeño: {pred_range_pct:.2f}%")
    elif pred_range_pct > 5.0:
        confidence_score -= 10
        print(f"⚠️ Rango muy grande: {pred_range_pct:.2f}%")
    
    # Límites de confianza
    confidence = max(0, min(100, confidence_score))
    
    # Umbral más alto para 5min (más selectivo)
    if confidence < 65:
        signal = "HOLD"
        confidence = 50
    
    return {
        'signal': signal,
        'confidence': confidence,
        'pred_change_%': pred_change_pct,
        'rsi': rsi,
        'atr': atr,
        'volatility_%': volatility,
        'momentum': momentum,
        'pred_range_%': pred_range_pct
    }

def main():
    print("="*70)
    print("  🔮 PREDICCIÓN 5 MINUTOS (Trading Rápido)")
    print("="*70 + "\n")
    
    # 1. Verificar señales recientes
    if check_recent_signals():
        print("✅ Skip - Señal reciente ya existe\n")
        return
    
    # 2. CARGAR MODELO (1h - pero lo usamos para tendencia general)
    model_dir = 'ADAUSD_MODELS'
    interval = '1h'
    
    if not os.path.exists(model_dir):
        print("❌ No existe modelo entrenado")
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
        model = MultiOutputLSTM(
            input_size=4,
            hidden_size=model_config.get('hidden', 192),
            num_layers=model_config.get('layers', 2),
            output_size=3,
            dropout=0.35
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print("✅ Modelo cargado\n")
        
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return
    
    # 3. DESCARGAR DATOS RECIENTES (1h para contexto)
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
        print(f"❌ Error descargando datos: {e}")
        return
    
    # 4. PREPARAR SECUENCIA Y PREDECIR
    print("🔮 Generando predicción...\n")
    
    inp = df[['open', 'high', 'low', 'close']].values[-seq_len:]
    inp_scaled = scaler_in.transform(inp)
    X = torch.FloatTensor(inp_scaled).unsqueeze(0)
    
    with torch.no_grad():
        pred = model(X)
    
    pred_denorm = scaler_out.inverse_transform(pred.numpy())
    pred_high, pred_low, pred_close = pred_denorm[0]
    
    current_price = df['close'].iloc[-1]
    
    print("="*70)
    print("  PREDICCIÓN BASE")
    print("="*70)
    print(f"Precio Actual:   ${current_price:.4f}")
    print(f"Pred High:       ${pred_high:.4f}")
    print(f"Pred Low:        ${pred_low:.4f}")
    print(f"Pred Close:      ${pred_close:.4f}")
    print("="*70 + "\n")
    
    # 5. INDICADORES TÉCNICOS
    print("📊 Calculando indicadores...")
    
    rsi = calculate_rsi(df['close'].values, period=14)
    atr = calculate_atr(df, period=14)
    momentum = detect_momentum_5min(df)
    
    print(f"RSI:       {rsi:.1f}")
    print(f"ATR:       ${atr:.4f}")
    print(f"Momentum:  {momentum}\n")
    
    # 6. GENERAR SEÑAL
    print("🎯 Generando señal de trading...\n")
    
    result = generate_signal_5min(
        pred_high, pred_low, pred_close,
        current_price, rsi, atr, momentum
    )
    
    signal = result['signal']
    confidence = result['confidence']
    
    print("="*70)
    print("  SEÑAL DE TRADING (5 MIN)")
    print("="*70)
    print(f"🚦 Señal:      {signal}")
    print(f"🎲 Confianza:  {confidence:.1f}%")
    print(f"📈 RSI:        {result['rsi']:.1f}")
    print(f"📊 Momentum:   {result['momentum']}")
    print(f"💹 Cambio Pred: {result['pred_change_%']:+.2f}%")
    print(f"📏 Rango Pred: {result['pred_range_%']:.2f}%")
    print("="*70 + "\n")
    
    # 7. GUARDAR SEÑAL
    timestamp = datetime.now()
    
    signal_data = {
        'timestamp': timestamp,
        'current_price': float(round(current_price, 4)),
        'pred_high': float(round(pred_high, 4)),
        'pred_low': float(round(pred_low, 4)),
        'pred_close': float(round(pred_close, 4)),
        'pred_change_%': float(round(result['pred_change_%'], 2)),
        'atr': float(round(result['atr'], 4)),
        'volatility': float(round(result['volatility_%'], 2)),
        'trend': str(result['momentum']),
        'signal': str(signal),
        'confidence': float(round(confidence, 1)),
        'rsi': float(round(result['rsi'], 1))
    }
    
    df_signal = pd.DataFrame([signal_data])
    
    if os.path.exists(SIGNALS_FILE):
        df_signal.to_csv(SIGNALS_FILE, mode='a', header=False, index=False)
    else:
        df_signal.to_csv(SIGNALS_FILE, index=False)
    
    print(f"✅ Señal guardada en {SIGNALS_FILE}\n")
    
    # 8. NOTIFICAR SI ES BUY/SELL
    if signal in ['BUY', 'SELL']:
        emoji = "🟢" if signal == "BUY" else "🔴"
        
        msg = f"""
{emoji} *Señal {signal} (5min)*

💰 Precio: ${current_price:.4f}
🔮 Pred Close: ${pred_close:.4f} ({result['pred_change_%']:+.2f}%)
🎲 Confianza: {confidence:.1f}%

📊 *Contexto:*
   RSI: {result['rsi']:.1f}
   Momentum: {result['momentum']}
   Volatilidad: {result['volatility_%']:.2f}%

⚡ Trading rápido - 5 min
"""
        
        send_telegram(msg)
    
    print("="*70)
    print("  ✅ PREDICCIÓN 5MIN COMPLETADA")
    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        raise
