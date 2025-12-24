"""
PREDICCIÓN CADA 5 MINUTOS - Usando modelo entrenado con velas 1h
✅ Normalización con min/max de 120 días
✅ Predice High, Low, Close de la siguiente vela
✅ Clasificación multi-factor mejorada
✅ 🔥 CORRECCIÓN: Auto-swap si High < Low
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

# Parámetros
NORMALIZATION_PERIOD_DAYS = 120
MIN_SIGNAL_INTERVAL_SECONDS = 240  # 4 minutos entre señales

def send_telegram(msg):
    if not TELEGRAM_API or not CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_API}/sendMessage"
        requests.post(url, data={'chat_id': CHAT_ID, 'text': msg, 'parse_mode': 'Markdown'}, timeout=10)
    except Exception as e:
        print(f"❌ Telegram: {e}")


class MultiOutputLSTM(nn.Module):
    """Modelo LSTM sincronizado con entrenamiento"""
    def __init__(self, input_size=4, hidden_size=160, num_layers=3,
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


def check_recent_signals():
    """
    🔥 Verifica si ya hay señal reciente (< 4 min)
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
        
        if time_diff.total_seconds() < MIN_SIGNAL_INTERVAL_SECONDS:
            print(f"⏸️ Señal reciente hace {time_diff.total_seconds():.0f}s - Skip")
            return True
        
        return False
        
    except Exception as e:
        print(f"⚠️ Error verificando señales: {e}")
        return False


def get_normalization_bounds(symbol="ADA-USD", days=120):
    """
    Obtiene min/max de 120 días para normalización
    (Como en el artículo de MQL5)
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
    """Normaliza al rango [0,1]"""
    if max_val <= min_val:
        raise ValueError("max_val debe ser > min_val")
    normalized = (prices - min_val) / (max_val - min_val)
    return np.clip(normalized, 0, 1)


def denormalize_prices(normalized, min_val, max_val):
    """Desnormaliza desde [0,1]"""
    return normalized * (max_val - min_val) + min_val


def calculate_rsi(prices, period=14):
    """RSI para 5 minutos (más sensible)"""
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
    Analiza últimas 12 velas (12 horas)
    """
    if len(df) < 12:
        return "NEUTRAL"
    
    recent = df.tail(12)
    
    # Cambio de precio
    price_change = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]
    
    # Velas alcistas vs bajistas
    bullish = (recent['close'] > recent['open']).sum()
    bearish = (recent['close'] < recent['open']).sum()
    
    # SMA rápida
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


def classify_price_movement_5min(current_price, pred_high, pred_low, pred_close, rsi, momentum):
    """
    🔥 CLASIFICACIÓN MEJORADA PARA 5 MINUTOS
    
    Considera:
    1. Predicción de Close
    2. Posición del Close en rango [Low, High]
    3. RSI (sobrecompra/sobreventa)
    4. Momentum de corto plazo
    5. Coherencia de predicciones
    """
    
    # Cambios predichos
    close_change = pred_close - current_price
    close_change_pct = (close_change / current_price) * 100
    
    # Rango predicho
    pred_range = pred_high - pred_low
    pred_range_pct = (pred_range / current_price) * 100
    
    # Posición del close en el rango
    if pred_range > 0:
        close_position = (pred_close - pred_low) / pred_range
    else:
        close_position = 0.5
    
    print(f"\n🎯 CLASIFICACIÓN (5 MINUTOS):")
    print(f"   Precio actual: ${current_price:.4f}")
    print(f"   Pred Close: ${pred_close:.4f} ({close_change_pct:+.2f}%)")
    print(f"   Pred High: ${pred_high:.4f}")
    print(f"   Pred Low: ${pred_low:.4f}")
    print(f"   Rango predicho: {pred_range_pct:.2f}%")
    print(f"   Close position: {close_position*100:.1f}%")
    print(f"   RSI: {rsi:.1f}")
    print(f"   Momentum: {momentum}")
    
    # Inicializar
    signal = "HOLD"
    confidence = 50.0
    
    # 🔥 UMBRAL MÁS BAJO PARA 5 MINUTOS
    THRESHOLD_PCT = 0.3  # 0.3% mínimo (más sensible que 1h)
    
    # Caso 1: Movimiento alcista
    if close_change_pct > THRESHOLD_PCT:
        signal = "BUY"
        confidence = 60 + min(close_change_pct * 8, 25) + (close_position * 10)
        
        # Reducir si close está cerca del low
        if close_position < 0.3:
            confidence -= 10
            print(f"   ⚠️  Close predicho cerca del Low")
    
    # Caso 2: Movimiento bajista
    elif close_change_pct < -THRESHOLD_PCT:
        signal = "SELL"
        confidence = 60 + min(abs(close_change_pct) * 8, 25) + ((1 - close_position) * 10)
        
        # Reducir si close está cerca del high
        if close_position > 0.7:
            confidence -= 10
            print(f"   ⚠️  Close predicho cerca del High")
    
    # Caso 3: Sin movimiento significativo
    else:
        signal = "HOLD"
        confidence = 50.0
        print(f"   ⏸️  Movimiento insuficiente (< {THRESHOLD_PCT}%)")
    
    # 🔥 AJUSTE POR RSI (más peso en 5min)
    if rsi < 35:  # Sobreventa
        if signal == "BUY":
            confidence += 12
            print(f"   📈 RSI sobreventa favorece BUY")
        elif signal == "SELL":
            confidence -= 12
    elif rsi > 65:  # Sobrecompra
        if signal == "SELL":
            confidence += 12
            print(f"   📉 RSI sobrecompra favorece SELL")
        elif signal == "BUY":
            confidence -= 12
    
    # 🔥 AJUSTE POR MOMENTUM (crítico en 5min)
    if momentum == "STRONG_UP":
        if signal == "BUY":
            confidence += 15
            print(f"   🚀 Momentum fuerte alcista")
        elif signal == "SELL":
            confidence -= 15
    elif momentum == "UP":
        if signal == "BUY":
            confidence += 8
    elif momentum == "STRONG_DOWN":
        if signal == "SELL":
            confidence += 15
            print(f"   📉 Momentum fuerte bajista")
        elif signal == "BUY":
            confidence -= 15
    elif momentum == "DOWN":
        if signal == "SELL":
            confidence += 8
    
    # Ajuste por volatilidad
    if pred_range_pct > 3.0:
        confidence -= 12
        print(f"   ⚠️  Alta volatilidad ({pred_range_pct:.2f}%)")
    elif pred_range_pct < 0.5:
        confidence -= 8
        print(f"   ⚠️  Baja volatilidad ({pred_range_pct:.2f}%)")
    
    # Coherencia: si High, Low, Close apuntan en misma dirección
    if pred_high > current_price and pred_low > current_price and pred_close > current_price:
        if signal == "BUY":
            confidence += 10
            print(f"   ✅ Todas las predicciones alcistas")
    elif pred_high < current_price and pred_low < current_price and pred_close < current_price:
        if signal == "SELL":
            confidence += 10
            print(f"   ✅ Todas las predicciones bajistas")
    
    # Limitar confianza
    confidence = max(0, min(100, confidence))
    
    # 🔥 UMBRAL DE CONFIANZA MÁS ALTO PARA 5 MINUTOS
    if confidence < 70:  # Más selectivo
        signal = "HOLD"
        confidence = 50
        print(f"   ⏸️  Confianza insuficiente para operar")
    
    print(f"\n   📊 Señal final: {signal}")
    print(f"   🎲 Confianza: {confidence:.1f}%")
    
    return {
        'signal': signal,
        'confidence': confidence,
        'close_change_%': close_change_pct,
        'pred_range_%': pred_range_pct,
        'close_position': close_position,
        'rsi': rsi,
        'momentum': momentum
    }


def save_prediction(timestamp, current_price, pred_high, pred_low, pred_close, result):
    """Guarda predicción en CSV"""
    
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
        'signal': str(result['signal']),
        'confidence': float(round(result['confidence'], 1)),
        'rsi': float(round(result['rsi'], 1)),
        'atr': 0.0,
        'trend': str(result['momentum']),
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
        'atr': 0.0,
        'volatility': float(round(pred_range_pct, 2)),
        'trend': str(result['momentum']),
        'signal': str(result['signal']),
        'confidence': float(round(result['confidence'], 1)),
        'rsi': float(round(result['rsi'], 1))
    }
    
    df_signal = pd.DataFrame([signal_data])
    
    if os.path.exists(SIGNALS_FILE):
        df_signal.to_csv(SIGNALS_FILE, mode='a', header=False, index=False)
    else:
        df_signal.to_csv(SIGNALS_FILE, index=False)
    
    print(f"✅ Predicción guardada")


def main():
    print("="*70)
    print("  🔮 PREDICCIÓN 5 MINUTOS (Velas 1h + Normalización 120d)")
    print("="*70 + "\n")
    
    # 1. Verificar señales recientes
    if check_recent_signals():
        print("✅ Skip - Señal reciente ya existe\n")
        return
    
    # 2. CARGAR MODELO
    model_dir = 'ADAUSD_MODELS'
    model_file = 'adausd_lstm_5min.pth'
    
    if not os.path.exists(f'{model_dir}/{model_file}'):
        error_msg = f"❌ No existe {model_dir}/{model_file}. Ejecuta entrenamiento primero."
        print(error_msg)
        send_telegram(error_msg)
        return
    
    print("📂 Cargando modelo...")
    
    try:
        with open(f'{model_dir}/config_5min.json', 'r') as f:
            config = json.load(f)
        
        seq_len = config['seq_len']
        
        checkpoint = torch.load(
            f'{model_dir}/{model_file}', 
            map_location=torch.device('cpu'),
            weights_only=False
        )
        
        model_config = checkpoint.get('config', {})
        hidden_size = model_config.get('hidden', 160)
        num_layers = model_config.get('layers', 3)
        
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
    
    # 3. OBTENER LÍMITES DE NORMALIZACIÓN (120 días)
    min_price, max_price = get_normalization_bounds("ADA-USD", NORMALIZATION_PERIOD_DAYS)
    
    if min_price is None or max_price is None:
        error_msg = "❌ No se pudieron obtener límites de normalización"
        print(error_msg)
        send_telegram(error_msg)
        return
    
    # 4. DESCARGAR DATOS RECIENTES (1h)
    print("\n📥 Descargando datos recientes...")
    
    try:
        ticker = yf.Ticker("ADA-USD")
        df = ticker.history(period="5d", interval="1h")
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
    
    # 5. NORMALIZAR DATOS
    print("🔧 Normalizando datos con límites de 120 días...")
    
    current_price = df['close'].iloc[-1]
    
    # Actualizar límites si necesario
    if current_price < min_price:
        print(f"   ⚠️  Precio actual < min")
        min_price = current_price
    elif current_price > max_price:
        print(f"   ⚠️  Precio actual > max")
        max_price = current_price
    
    # Normalizar secuencia
    inp = df[['open', 'high', 'low', 'close']].values[-seq_len:]
    
    inp_normalized = np.zeros_like(inp)
    for i in range(inp.shape[0]):
        for j in range(inp.shape[1]):
            inp_normalized[i, j] = normalize_prices(inp[i, j], min_price, max_price)
    
    X = torch.FloatTensor(inp_normalized).unsqueeze(0)
    
    # 6. GENERAR PREDICCIÓN
    print("\n🔮 Generando predicción...\n")
    
    with torch.no_grad():
        pred = model(X)
    
    pred_normalized = pred.numpy()[0]
    
    # Desnormalizar
    pred_high_raw = denormalize_prices(pred_normalized[0], min_price, max_price)
    pred_low_raw = denormalize_prices(pred_normalized[1], min_price, max_price)
    pred_close_raw = denormalize_prices(pred_normalized[2], min_price, max_price)
    
    # 🔥 CORRECCIÓN: Si High < Low, intercambiar
    if pred_high_raw < pred_low_raw:
        print(f"⚠️ Predicción invertida detectada:")
        print(f"   Raw High: ${pred_high_raw:.4f}")
        print(f"   Raw Low: ${pred_low_raw:.4f}")
        print(f"   → Intercambiando valores...\n")
        
        pred_high = pred_low_raw
        pred_low = pred_high_raw
    else:
        pred_high = pred_high_raw
        pred_low = pred_low_raw
    
    # Asegurar que Close esté en el rango [Low, High]
    pred_close = np.clip(pred_close_raw, pred_low, pred_high)
    
    if pred_close != pred_close_raw:
        print(f"⚠️ Close ajustado al rango:")
        print(f"   Raw Close: ${pred_close_raw:.4f}")
        print(f"   Adjusted: ${pred_close:.4f}\n")
    
    print("="*70)
    print("  📊 PREDICCIÓN (CORREGIDA)")
    print("="*70)
    print(f"Precio Actual:   ${current_price:.4f}")
    print(f"Pred High:       ${pred_high:.4f} ({((pred_high-current_price)/current_price*100):+.2f}%)")
    print(f"Pred Low:        ${pred_low:.4f} ({((pred_low-current_price)/current_price*100):+.2f}%)")
    print(f"Pred Close:      ${pred_close:.4f} ({((pred_close-current_price)/current_price*100):+.2f}%)")
    print(f"Rango:           ${pred_high - pred_low:.4f} ({((pred_high - pred_low)/current_price*100):.2f}%)")
    print("="*70 + "\n")
    
    # Verificación final
    if pred_high < pred_low:
        error_msg = f"❌ ERROR CRÍTICO: Aún hay inversión después de corrección"
        print(error_msg)
        send_telegram(error_msg)
        return
    
    # 7. CALCULAR INDICADORES
    print("📊 Calculando indicadores...")
    
    rsi = calculate_rsi(df['close'].values, period=14)
    atr = calculate_atr(df, period=14)
    momentum = detect_momentum_5min(df)
    
    print(f"   RSI: {rsi:.1f}")
    print(f"   ATR: ${atr:.4f}")
    print(f"   Momentum: {momentum}")
    
    # 8. CLASIFICAR MOVIMIENTO
    result = classify_price_movement_5min(
        current_price, pred_high, pred_low, pred_close, rsi, momentum
    )
    
    signal = result['signal']
    confidence = result['confidence']
    
    # 9. GUARDAR PREDICCIÓN
    timestamp = datetime.now()
    save_prediction(timestamp, current_price, pred_high, pred_low, pred_close, result)
    
    # 10. NOTIFICAR TELEGRAM
    if signal in ['BUY', 'SELL']:
        emoji = "🟢" if signal == "BUY" else "🔴"
        
        msg = f"""
{emoji} *Señal {signal} (5min)*

💰 Precio: ${current_price:.4f}
🔮 Pred Close: ${pred_close:.4f} ({result['close_change_%']:+.2f}%)
🎲 Confianza: {confidence:.1f}%

📊 *Predicciones:*
   High: ${pred_high:.4f}
   Low: ${pred_low:.4f}
   Rango: {result['pred_range_%']:.2f}%

📈 *Contexto:*
   RSI: {result['rsi']:.1f}
   Momentum: {result['momentum']}

⚡ Trading rápido - 5 min
"""
        
        send_telegram(msg)
    
    print("\n" + "="*70)
    print("  ✅ PREDICCIÓN 5MIN COMPLETADA")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        print(error_msg)
        send_telegram(error_msg)
        raise
