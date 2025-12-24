"""
PREDICTOR MEJORADO - Inspirado en enfoque MQL5
✅ Normalización con min/max de 120 días
✅ Predicción de High, Low, Close para siguiente vela
✅ Clasificación de movimiento más robusta
✅ Confianza basada en coherencia de predicciones
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

# Archivos
PREDICTION_TRACKER_FILE = 'prediction_tracker.csv'
SIGNALS_FILE = 'trading_signals.csv'

# Parámetros de normalización (como MQL5)
NORMALIZATION_PERIOD_DAYS = 120  # Usar 120 días para min/max

def send_telegram(msg):
    if not TELEGRAM_API or not CHAT_ID:
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


def get_normalization_bounds(symbol="ADA-USD", days=120):
    """
    🔥 NUEVO: Obtiene min/max de 120 días como en MQL5
    
    Esto es equivalente a:
    vectorf close;
    close.CopyRates(_Symbol,PERIOD_D1,COPY_RATES_CLOSE,0,SAMPLE_SIZE);
    ExtMin=close.Min();
    ExtMax=close.Max();
    """
    print(f"\n📊 Obteniendo límites de normalización ({days} días)...")
    
    try:
        ticker = yf.Ticker(symbol)
        df_daily = ticker.history(period=f"{days}d", interval="1d")
        
        if len(df_daily) == 0:
            print("❌ No se pudieron obtener datos diarios")
            return None, None
        
        min_price = df_daily['Close'].min()
        max_price = df_daily['Close'].max()
        
        print(f"   Min (120d): ${min_price:.4f}")
        print(f"   Max (120d): ${max_price:.4f}")
        print(f"   Rango: ${max_price - min_price:.4f}")
        
        return float(min_price), float(max_price)
        
    except Exception as e:
        print(f"❌ Error obteniendo límites: {e}")
        return None, None


def normalize_prices(prices, min_val, max_val):
    """
    Normaliza precios al rango [0,1] usando min/max
    Equivalente a MinMaxScaler de sklearn
    """
    if max_val <= min_val:
        raise ValueError("max_val debe ser > min_val")
    
    normalized = (prices - min_val) / (max_val - min_val)
    return np.clip(normalized, 0, 1)  # Asegurar [0,1]


def denormalize_prices(normalized, min_val, max_val):
    """
    Desnormaliza precios del rango [0,1]
    """
    return normalized * (max_val - min_val) + min_val


def classify_price_movement(current_price, pred_high, pred_low, pred_close):
    """
    🔥 CLASIFICACIÓN MEJORADA inspirada en MQL5
    
    En MQL5:
    float delta=last_close-predicted;
    if(fabs(delta)<=0.00001)
       ExtPredictedClass=PRICE_SAME;
    else
      {
       if(delta<0)
          ExtPredictedClass=PRICE_UP;
       else
          ExtPredictedClass=PRICE_DOWN;
      }
    
    Aquí usamos múltiples factores:
    1. Predicción de Close
    2. Posición del Close predicho vs High/Low
    3. Rango de movimiento esperado
    """
    
    # 1. Cambio predicho en Close
    close_change = pred_close - current_price
    close_change_pct = (close_change / current_price) * 100
    
    # 2. Rango predicho
    pred_range = pred_high - pred_low
    pred_range_pct = (pred_range / current_price) * 100
    
    # 3. Posición relativa del close en el rango [low, high]
    if pred_range > 0:
        close_position = (pred_close - pred_low) / pred_range
    else:
        close_position = 0.5
    
    print(f"\n🎯 CLASIFICACIÓN DE MOVIMIENTO:")
    print(f"   Precio actual: ${current_price:.4f}")
    print(f"   Pred Close: ${pred_close:.4f} ({close_change_pct:+.2f}%)")
    print(f"   Pred High: ${pred_high:.4f}")
    print(f"   Pred Low: ${pred_low:.4f}")
    print(f"   Rango predicho: {pred_range_pct:.2f}%")
    print(f"   Close position: {close_position*100:.1f}% del rango")
    
    # 🔥 LÓGICA DE CLASIFICACIÓN
    
    # Umbral de movimiento significativo
    THRESHOLD_PCT = 0.5  # 0.5% mínimo para considerar movimiento
    
    signal = "HOLD"
    confidence = 50.0
    
    # Caso 1: Movimiento claro hacia arriba
    if close_change_pct > THRESHOLD_PCT:
        signal = "BUY"
        # Confianza basada en:
        # - Magnitud del cambio
        # - Posición del close en el rango (cerca del high = más confianza)
        confidence = 60 + min(close_change_pct * 5, 20) + (close_position * 15)
        
        # Si el close está muy cerca del low, reducir confianza
        if close_position < 0.3:
            confidence -= 10
            print(f"   ⚠️  Close predicho cerca del Low (posible reversión)")
    
    # Caso 2: Movimiento claro hacia abajo
    elif close_change_pct < -THRESHOLD_PCT:
        signal = "SELL"
        confidence = 60 + min(abs(close_change_pct) * 5, 20) + ((1 - close_position) * 15)
        
        # Si el close está muy cerca del high, reducir confianza
        if close_position > 0.7:
            confidence -= 10
            print(f"   ⚠️  Close predicho cerca del High (posible reversión)")
    
    # Caso 3: Sin movimiento significativo
    else:
        signal = "HOLD"
        confidence = 50.0
        print(f"   ⏸️  Movimiento insuficiente (< {THRESHOLD_PCT}%)")
    
    # Ajuste por volatilidad predicha
    if pred_range_pct > 5.0:
        confidence -= 10
        print(f"   ⚠️  Alta volatilidad predicha ({pred_range_pct:.2f}%)")
    elif pred_range_pct < 1.0:
        confidence -= 5
        print(f"   ⚠️  Baja volatilidad predicha ({pred_range_pct:.2f}%)")
    
    # Limitar confianza
    confidence = max(0, min(100, confidence))
    
    print(f"\n   📊 Señal: {signal}")
    print(f"   🎲 Confianza: {confidence:.1f}%")
    
    return {
        'signal': signal,
        'confidence': confidence,
        'close_change_%': close_change_pct,
        'pred_range_%': pred_range_pct,
        'close_position': close_position
    }


def calculate_technical_indicators(df):
    """Calcula RSI, ATR y tendencia"""
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_value = rsi.iloc[-1]
    
    # ATR
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean().iloc[-1]
    
    # Tendencia (SMA 20)
    sma = df['close'].rolling(window=20).mean()
    current_price = df['close'].iloc[-1]
    sma_value = sma.iloc[-1]
    
    if current_price > sma_value * 1.02:
        trend = "UPTREND"
    elif current_price < sma_value * 0.98:
        trend = "DOWNTREND"
    else:
        trend = "SIDEWAYS"
    
    return {
        'rsi': float(rsi_value),
        'atr': float(atr),
        'trend': trend
    }


def save_prediction(timestamp, current_price, pred_high, pred_low, pred_close,
                   signal, confidence, indicators):
    """Guarda predicción en CSV de tracking"""
    
    close_change = ((pred_close - current_price) / current_price) * 100
    high_change = ((pred_high - current_price) / current_price) * 100
    low_change = ((pred_low - current_price) / current_price) * 100
    pred_range = pred_high - pred_low
    pred_range_pct = (pred_range / current_price) * 100
    
    # Prediction tracker
    tracking_data = {
        'timestamp': timestamp,
        'current_price': float(round(current_price, 4)),
        'pred_high': float(round(pred_high, 4)),
        'pred_low': float(round(pred_low, 4)),
        'pred_close': float(round(pred_close, 4)),
        'pred_high_change_%': float(round(high_change, 2)),
        'pred_low_change_%': float(round(low_change, 2)),
        'pred_close_change_%': float(round(close_change, 2)),
        'pred_range': float(round(pred_range, 4)),
        'pred_range_%': float(round(pred_range_pct, 2)),
        'signal': str(signal),
        'confidence': float(round(confidence, 1)),
        'rsi': float(round(indicators['rsi'], 1)),
        'atr': float(round(indicators['atr'], 4)),
        'trend': str(indicators['trend']),
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
    
    # Trading signals
    signal_data = {
        'timestamp': timestamp,
        'current_price': float(round(current_price, 4)),
        'pred_high': float(round(pred_high, 4)),
        'pred_low': float(round(pred_low, 4)),
        'pred_close': float(round(pred_close, 4)),
        'pred_change_%': float(round(close_change, 2)),
        'atr': float(round(indicators['atr'], 4)),
        'volatility': float(round(pred_range_pct, 2)),
        'trend': str(indicators['trend']),
        'signal': str(signal),
        'confidence': float(round(confidence, 1)),
        'rsi': float(round(indicators['rsi'], 1))
    }
    
    df_signal = pd.DataFrame([signal_data])
    
    if os.path.exists(SIGNALS_FILE):
        df_signal.to_csv(SIGNALS_FILE, mode='a', header=False, index=False)
    else:
        df_signal.to_csv(SIGNALS_FILE, index=False)
    
    print(f"✅ Predicción guardada")


def main():
    print("="*70)
    print("  🔮 PREDICTOR MEJORADO (Enfoque MQL5)")
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
        
        checkpoint = torch.load(
            f'{model_dir}/adausd_lstm_{interval}.pth', 
            map_location=torch.device('cpu'),
            weights_only=False
        )
        
        model_config = checkpoint.get('config', {})
        hidden_size = model_config.get('hidden', 192)
        num_layers = model_config.get('layers', 2)
        
        model = MultiOutputLSTM(
            input_size=4,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=3,
            dropout=0.35
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print("✅ Modelo cargado\n")
        
    except Exception as e:
        error_msg = f"❌ Error cargando modelo: {str(e)}"
        print(error_msg)
        send_telegram(error_msg)
        return
    
    # 2. OBTENER LÍMITES DE NORMALIZACIÓN (120 días)
    min_price, max_price = get_normalization_bounds("ADA-USD", NORMALIZATION_PERIOD_DAYS)
    
    if min_price is None or max_price is None:
        error_msg = "❌ No se pudieron obtener límites de normalización"
        print(error_msg)
        send_telegram(error_msg)
        return
    
    # 3. DESCARGAR DATOS RECIENTES (H1)
    print("\n📥 Descargando datos recientes...")
    
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
    
    # 4. NORMALIZAR DATOS
    print("🔧 Normalizando datos con límites de 120 días...")
    
    current_price = df['close'].iloc[-1]
    
    # Actualizar límites si el precio actual está fuera del rango
    if current_price < min_price:
        print(f"   ⚠️  Precio actual (${current_price:.4f}) < min ({min_price:.4f})")
        min_price = current_price
    elif current_price > max_price:
        print(f"   ⚠️  Precio actual (${current_price:.4f}) > max ({max_price:.4f})")
        max_price = current_price
    
    # Normalizar secuencia
    inp = df[['open', 'high', 'low', 'close']].values[-seq_len:]
    
    inp_normalized = np.zeros_like(inp)
    for i in range(inp.shape[0]):
        for j in range(inp.shape[1]):
            inp_normalized[i, j] = normalize_prices(inp[i, j], min_price, max_price)
    
    X = torch.FloatTensor(inp_normalized).unsqueeze(0)
    
    # 5. GENERAR PREDICCIÓN
    print("\n🔮 Generando predicción...\n")
    
    with torch.no_grad():
        pred = model(X)
    
    pred_normalized = pred.numpy()[0]
    
    # Desnormalizar
    pred_high = denormalize_prices(pred_normalized[0], min_price, max_price)
    pred_low = denormalize_prices(pred_normalized[1], min_price, max_price)
    pred_close = denormalize_prices(pred_normalized[2], min_price, max_price)
    
    print("="*70)
    print("  📊 PREDICCIÓN")
    print("="*70)
    print(f"Precio Actual:   ${current_price:.4f}")
    print(f"Pred High:       ${pred_high:.4f} ({((pred_high-current_price)/current_price*100):+.2f}%)")
    print(f"Pred Low:        ${pred_low:.4f} ({((pred_low-current_price)/current_price*100):+.2f}%)")
    print(f"Pred Close:      ${pred_close:.4f} ({((pred_close-current_price)/current_price*100):+.2f}%)")
    print("="*70 + "\n")
    
    # 6. CALCULAR INDICADORES TÉCNICOS
    print("📊 Calculando indicadores técnicos...")
    indicators = calculate_technical_indicators(df)
    print(f"   RSI: {indicators['rsi']:.1f}")
    print(f"   ATR: ${indicators['atr']:.4f}")
    print(f"   Tendencia: {indicators['trend']}")
    
    # 7. CLASIFICAR MOVIMIENTO
    result = classify_price_movement(current_price, pred_high, pred_low, pred_close)
    
    signal = result['signal']
    confidence = result['confidence']
    
    # 8. GUARDAR PREDICCIÓN
    timestamp = datetime.now()
    save_prediction(
        timestamp, current_price, pred_high, pred_low, pred_close,
        signal, confidence, indicators
    )
    
    # 9. NOTIFICAR TELEGRAM
    emoji = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "⚪"
    
    msg = f"""
{emoji} *Nueva Señal: {signal}*

💰 Precio: ${current_price:.4f}
🔮 Pred Close: ${pred_close:.4f} ({result['close_change_%']:+.2f}%)
🎲 Confianza: {confidence:.1f}%

📊 *Predicciones:*
   High: ${pred_high:.4f}
   Low: ${pred_low:.4f}
   Rango: {result['pred_range_%']:.2f}%

📈 *Indicadores:*
   RSI: {indicators['rsi']:.1f}
   ATR: ${indicators['atr']:.4f}
   Tendencia: {indicators['trend']}
"""
    
    send_telegram(msg)
    
    print("\n" + "="*70)
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
