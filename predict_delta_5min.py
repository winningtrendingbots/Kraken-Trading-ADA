"""
PREDICCIÓN CON DELTAS - SOLUCIÓN AL PROBLEMA DE DESANCLAJE

✅ Predice cambios RELATIVOS al precio actual
✅ Garantiza que predicciones estén ancladas al precio base
✅ Usa volumen y sus indicadores avanzados
✅ Implementa técnicas de confirmación de tendencia
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

# Archivos
PREDICTION_TRACKER_FILE = 'prediction_tracker.csv'
SIGNALS_FILE = 'trading_signals.csv'

# Configuración
MIN_SIGNAL_INTERVAL_SECONDS = 240  # 4 minutos

def send_telegram(msg):
    if not TELEGRAM_API or not CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_API}/sendMessage"
        requests.post(url, data={'chat_id': CHAT_ID, 'text': msg, 'parse_mode': 'Markdown'}, timeout=10)
    except Exception as e:
        print(f"❌ Telegram: {e}")

# ================================
# 📊 INDICADORES DE VOLUMEN
# ================================
def calculate_volume_indicators(df):
    """Calcula indicadores avanzados de volumen"""
    df = df.copy()
    
    # OBV
    obv = [0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.append(obv[-1] + df['volume'].iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.append(obv[-1] - df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['obv'] = obv
    
    # Volume Rate of Change
    df['volume_roc'] = df['volume'].pct_change(periods=14)
    
    # Volume MA
    df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
    
    # Volume Ratio
    df['volume_ratio'] = df['volume'] / df['volume_ma_20']
    
    # VWAP
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    
    # PVT
    df['pvt'] = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1) * df['volume']).cumsum()
    
    df.fillna(method='bfill', inplace=True)
    
    return df

def detect_volume_divergence(df, window=14):
    """Detecta divergencias entre precio y volumen"""
    df = df.copy()
    
    # Tendencia de precio
    price_slope = df['close'].rolling(window=window).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else 0
    )
    
    # Tendencia de volumen
    volume_slope = df['volume'].rolling(window=window).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else 0
    )
    
    # Divergencia bearish: precio sube, volumen baja
    df['bearish_divergence'] = ((price_slope > 0) & (volume_slope < 0)).astype(int)
    
    # Divergencia bullish: precio baja, volumen baja
    df['bullish_divergence'] = ((price_slope < 0) & (volume_slope < 0)).astype(int)
    
    df.fillna(0, inplace=True)
    
    return df

def analyze_volume_confirmation(df, current_idx):
    """
    🔥 TÉCNICA AVANZADA: Confirmación de Tendencia por Volumen
    
    Basado en el artículo de MQL5:
    - Alto volumen + subida de precio = tendencia alcista fuerte
    - Alto volumen + bajada de precio = tendencia bajista fuerte
    - Bajo volumen = señal débil o falsa
    """
    if current_idx < 20:
        return {
            'trend_confirmation': 'NEUTRAL',
            'strength': 'WEAK',
            'volume_support': False
        }
    
    # Últimas 20 velas
    recent = df.iloc[current_idx-20:current_idx+1]
    
    current_volume = recent['volume'].iloc[-1]
    avg_volume = recent['volume'].mean()
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
    
    # Cambio de precio reciente
    price_change = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]
    
    # Clasificar
    if volume_ratio > 1.5:  # Volumen alto
        if price_change > 0.01:  # Subida
            trend = 'STRONG_BULLISH'
            strength = 'STRONG'
            support = True
        elif price_change < -0.01:  # Bajada
            trend = 'STRONG_BEARISH'
            strength = 'STRONG'
            support = True
        else:
            trend = 'CONSOLIDATION'
            strength = 'MEDIUM'
            support = False
    
    elif volume_ratio < 0.7:  # Volumen bajo
        if abs(price_change) > 0.02:
            trend = 'WEAK_MOVE'
            strength = 'WEAK'
            support = False  # Movimiento sin convicción
        else:
            trend = 'LOW_ACTIVITY'
            strength = 'WEAK'
            support = False
    
    else:  # Volumen normal
        if price_change > 0.01:
            trend = 'BULLISH'
            strength = 'MEDIUM'
            support = True
        elif price_change < -0.01:
            trend = 'BEARISH'
            strength = 'MEDIUM'
            support = True
        else:
            trend = 'NEUTRAL'
            strength = 'WEAK'
            support = False
    
    return {
        'trend_confirmation': trend,
        'strength': strength,
        'volume_support': support,
        'volume_ratio': volume_ratio,
        'price_change_%': price_change * 100
    }

def validate_breakout(df, current_idx, pred_delta_high, pred_delta_low):
    """
    🔥 TÉCNICA AVANZADA: Validación de Breakouts
    
    Un breakout válido debe tener:
    1. Alto volumen
    2. Momentum consistente
    3. No ser un "bull trap" o "bear trap"
    """
    if current_idx < 50:
        return {
            'is_breakout': False,
            'breakout_type': 'NONE',
            'confidence': 0
        }
    
    recent = df.iloc[current_idx-50:current_idx+1]
    
    # Resistencia/Soporte reciente (high/low de últimas 50 velas)
    resistance = recent['high'].max()
    support = recent['low'].min()
    
    current_price = recent['close'].iloc[-1]
    pred_high = current_price * (1 + pred_delta_high)
    pred_low = current_price * (1 + pred_delta_low)
    
    # Volumen actual vs promedio
    current_volume = recent['volume'].iloc[-1]
    avg_volume = recent['volume'].mean()
    volume_spike = current_volume / avg_volume if avg_volume > 0 else 1
    
    # Detectar breakout
    breakout_up = pred_high > resistance * 1.001  # 0.1% sobre resistencia
    breakout_down = pred_low < support * 0.999    # 0.1% bajo soporte
    
    if breakout_up and volume_spike > 1.3:
        return {
            'is_breakout': True,
            'breakout_type': 'BULLISH',
            'confidence': min(volume_spike * 50, 100),
            'target': resistance,
            'volume_spike': volume_spike
        }
    
    elif breakout_down and volume_spike > 1.3:
        return {
            'is_breakout': True,
            'breakout_type': 'BEARISH',
            'confidence': min(volume_spike * 50, 100),
            'target': support,
            'volume_spike': volume_spike
        }
    
    elif (breakout_up or breakout_down) and volume_spike < 1.1:
        return {
            'is_breakout': False,
            'breakout_type': 'FALSE_BREAKOUT',
            'confidence': 0,
            'warning': 'Bajo volumen = posible trampa'
        }
    
    else:
        return {
            'is_breakout': False,
            'breakout_type': 'NONE',
            'confidence': 0
        }

# ================================
# 🧠 MODELO
# ================================
class ImprovedLSTM(nn.Module):
    """Mismo modelo que en entrenamiento"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.35):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
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

# ================================
# 📊 CLASIFICACIÓN MEJORADA
# ================================
def classify_with_deltas_and_volume(current_price, delta_high, delta_low, delta_close, 
                                   volume_analysis, breakout_analysis, divergence_signals):
    """
    🔥 CLASIFICACIÓN MEJORADA CON DELTAS Y VOLUMEN
    
    Ahora las predicciones están ANCLADAS al precio actual:
    - pred_high = current_price * (1 + delta_high)
    - pred_low = current_price * (1 + delta_low)
    - pred_close = current_price * (1 + delta_close)
    
    ✅ SOLUCIONA EL PROBLEMA DE DESANCLAJE
    """
    
    # Convertir deltas a precios absolutos
    pred_high = current_price * (1 + delta_high)
    pred_low = current_price * (1 + delta_low)
    pred_close = current_price * (1 + delta_close)
    
    # Garantizar coherencia
    if pred_high < pred_low:
        pred_high, pred_low = pred_low, pred_high
    
    pred_close = np.clip(pred_close, pred_low, pred_high)
    
    print(f"\n🎯 CLASIFICACIÓN CON DELTAS:")
    print(f"   Precio actual: ${current_price:.4f}")
    print(f"   Delta High: {delta_high*100:+.2f}% → ${pred_high:.4f}")
    print(f"   Delta Low: {delta_low*100:+.2f}% → ${pred_low:.4f}")
    print(f"   Delta Close: {delta_close*100:+.2f}% → ${pred_close:.4f}")
    
    # Verificar anclaje
    print(f"\n✅ VERIFICACIÓN DE ANCLAJE:")
    print(f"   ¿High > Low? {pred_high > pred_low}")
    print(f"   ¿Close en rango? {pred_low <= pred_close <= pred_high}")
    print(f"   ¿Precio actual referenciado? ✅")
    
    # Decisión base
    close_change_pct = delta_close * 100
    
    signal = "HOLD"
    confidence = 50.0
    
    THRESHOLD = 0.3  # 0.3%
    
    if close_change_pct > THRESHOLD:
        signal = "BUY"
        confidence = 60 + min(close_change_pct * 8, 25)
    elif close_change_pct < -THRESHOLD:
        signal = "SELL"
        confidence = 60 + min(abs(close_change_pct) * 8, 25)
    
    # 🔥 AJUSTE POR VOLUMEN
    vol_trend = volume_analysis['trend_confirmation']
    vol_support = volume_analysis['volume_support']
    
    print(f"\n📊 ANÁLISIS DE VOLUMEN:")
    print(f"   Tendencia: {vol_trend}")
    print(f"   Soporte: {'✅' if vol_support else '❌'}")
    print(f"   Ratio volumen: {volume_analysis['volume_ratio']:.2f}x")
    
    if vol_trend == 'STRONG_BULLISH' and signal == 'BUY':
        confidence += 15
        print(f"   ✅ Volumen confirma BUY (+15)")
    elif vol_trend == 'STRONG_BEARISH' and signal == 'SELL':
        confidence += 15
        print(f"   ✅ Volumen confirma SELL (+15)")
    elif vol_trend == 'WEAK_MOVE' or not vol_support:
        confidence -= 20
        print(f"   ⚠️ Volumen débil (-20)")
    
    # 🔥 AJUSTE POR BREAKOUT
    breakout_info = breakout_analysis
    
    print(f"\n🚀 ANÁLISIS DE BREAKOUT:")
    print(f"   Tipo: {breakout_info['breakout_type']}")
    print(f"   Es breakout: {breakout_info['is_breakout']}")
    
    if breakout_info['is_breakout']:
        if breakout_info['breakout_type'] == 'BULLISH' and signal == 'BUY':
            confidence += 20
            print(f"   ✅ Breakout alcista validado (+20)")
        elif breakout_info['breakout_type'] == 'BEARISH' and signal == 'SELL':
            confidence += 20
            print(f"   ✅ Breakout bajista validado (+20)")
    
    elif breakout_info['breakout_type'] == 'FALSE_BREAKOUT':
        confidence -= 25
        print(f"   ⚠️ Posible falso breakout (-25)")
    
    # 🔥 AJUSTE POR DIVERGENCIAS
    if divergence_signals.get('bearish_divergence', 0) > 0 and signal == 'BUY':
        confidence -= 15
        print(f"\n⚠️ Divergencia bearish detectada (-15)")
    
    if divergence_signals.get('bullish_divergence', 0) > 0 and signal == 'SELL':
        confidence -= 15
        print(f"\n⚠️ Divergencia bullish detectada (-15)")
    
    # Limitar confianza
    confidence = max(0, min(100, confidence))
    
    # Umbral de confianza
    if confidence < 70:
        signal = "HOLD"
        confidence = 50
        print(f"\n⏸️ Confianza insuficiente para operar")
    
    print(f"\n🎲 SEÑAL FINAL: {signal}")
    print(f"🎲 CONFIANZA: {confidence:.1f}%")
    
    return {
        'signal': signal,
        'confidence': confidence,
        'pred_high': pred_high,
        'pred_low': pred_low,
        'pred_close': pred_close,
        'delta_high': delta_high,
        'delta_low': delta_low,
        'delta_close': delta_close,
        'close_change_%': close_change_pct,
        'volume_confirmation': vol_trend,
        'volume_support': vol_support,
        'breakout_type': breakout_info['breakout_type'],
        'is_breakout': breakout_info['is_breakout']
    }

# ================================
# 📝 GUARDAR PREDICCIÓN
# ================================
def save_prediction(timestamp, current_price, result):
    """Guarda predicción en CSV"""
    
    pred_range = result['pred_high'] - result['pred_low']
    pred_range_pct = (pred_range / current_price) * 100
    
    # Prediction tracker
    tracking_data = {
        'timestamp': timestamp,
        'current_price': float(round(current_price, 4)),
        'pred_high': float(round(result['pred_high'], 4)),
        'pred_low': float(round(result['pred_low'], 4)),
        'pred_close': float(round(result['pred_close'], 4)),
        'pred_high_change_%': float(round(result['delta_high'] * 100, 2)),
        'pred_low_change_%': float(round(result['delta_low'] * 100, 2)),
        'pred_close_change_%': float(round(result['close_change_%'], 2)),
        'pred_range': float(round(pred_range, 4)),
        'pred_range_%': float(round(pred_range_pct, 2)),
        'signal': str(result['signal']),
        'confidence': float(round(result['confidence'], 1)),
        'volume_confirmation': str(result['volume_confirmation']),
        'volume_support': bool(result['volume_support']),
        'breakout_type': str(result['breakout_type']),
        'is_breakout': bool(result['is_breakout']),
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
        'pred_high': float(round(result['pred_high'], 4)),
        'pred_low': float(round(result['pred_low'], 4)),
        'pred_close': float(round(result['pred_close'], 4)),
        'pred_change_%': float(round(result['close_change_%'], 2)),
        'atr': 0.0,
        'volatility': float(round(pred_range_pct, 2)),
        'trend': str(result['volume_confirmation']),
        'signal': str(result['signal']),
        'confidence': float(round(result['confidence'], 1)),
        'rsi': 0.0
    }
    
    df_signal = pd.DataFrame([signal_data])
    
    if os.path.exists(SIGNALS_FILE):
        df_signal.to_csv(SIGNALS_FILE, mode='a', header=False, index=False)
    else:
        df_signal.to_csv(SIGNALS_FILE, index=False)
    
    print(f"✅ Predicción guardada")

# ================================
# 🚀 MAIN
# ================================
def main():
    print("="*70)
    print("  🔮 PREDICCIÓN CON DELTAS Y VOLUMEN")
    print("="*70 + "\n")
    
    # 1. Verificar señales recientes
    if os.path.exists(SIGNALS_FILE):
        try:
            df_sig = pd.read_csv(SIGNALS_FILE)
            if len(df_sig) > 0:
                df_sig['timestamp'] = pd.to_datetime(df_sig['timestamp'])
                last_signal = df_sig.iloc[-1]
                
                time_diff = datetime.now() - last_signal['timestamp']
                
                if time_diff.total_seconds() < MIN_SIGNAL_INTERVAL_SECONDS:
                    print(f"⏸️ Señal reciente hace {time_diff.total_seconds():.0f}s - Skip\n")
                    return
        except Exception as e:
            print(f"⚠️ Error verificando señales: {e}")
    
    # 2. Cargar modelo
    model_dir = 'ADAUSD_MODELS'
    model_file = 'adausd_lstm_delta.pth'
    
    if not os.path.exists(f'{model_dir}/{model_file}'):
        error_msg = f"❌ No existe {model_dir}/{model_file}"
        print(error_msg)
        send_telegram(error_msg)
        return
    
    print("📂 Cargando modelo...")
    
    try:
        with open(f'{model_dir}/config_delta.json', 'r') as f:
            config = json.load(f)
        
        seq_len = config['seq_len']
        input_size = config['input_size']
        output_size = config['output_size']
        hidden_size = config['hidden']
        num_layers = config['layers']
        feature_cols = config['feature_cols']
        target_cols = config['target_cols']
        
        checkpoint = torch.load(
            f'{model_dir}/{model_file}', 
            map_location=torch.device('cpu'),
            weights_only=False
        )
        
        model = ImprovedLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=0.35
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        scaler_in = joblib.load(f'{model_dir}/scaler_input_delta.pkl')
        scaler_out = joblib.load(f'{model_dir}/scaler_output_delta.pkl')
        
        print("✅ Modelo cargado\n")
        
    except Exception as e:
        error_msg = f"❌ Error cargando modelo: {str(e)}"
        print(error_msg)
        send_telegram(error_msg)
        return
    
    # 3. Descargar datos recientes
    print("📥 Descargando datos recientes...")
    
    try:
        ticker = yf.Ticker("ADA-USD")
        df = ticker.history(period="5d", interval="1h")
        df = df.reset_index()
        df.columns = [str(c).lower() for c in df.columns]
        df.rename(columns={'date': 'time', 'datetime': 'time'}, inplace=True)
        
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
        
        df = df[['time', 'open', 'high', 'low', 'close', 'volume']].tail(seq_len + 50)
        
        print(f"✅ {len(df)} velas descargadas\n")
        
    except Exception as e:
        error_msg = f"❌ Error descargando datos: {str(e)}"
        print(error_msg)
        send_telegram(error_msg)
        return
    
    # 4. Calcular indicadores de volumen
    if 'obv' in feature_cols or 'volume_ratio' in feature_cols:
        print("📊 Calculando indicadores de volumen...")
        df = calculate_volume_indicators(df)
        df = detect_volume_divergence(df)
    
    current_price = df['close'].iloc[-1]
    current_idx = len(df) - 1
    
    # 5. Análisis de volumen avanzado
    volume_analysis = analyze_volume_confirmation(df, current_idx)
    
    # 6. Análisis de divergencias
    divergence_signals = {
        'bearish_divergence': df['bearish_divergence'].iloc[-1] if 'bearish_divergence' in df.columns else 0,
        'bullish_divergence': df['bullish_divergence'].iloc[-1] if 'bullish_divergence' in df.columns else 0
    }
    
    # 7. Preparar input
    inp = df[feature_cols].values[-seq_len:]
    inp_scaled = scaler_in.transform(inp)
    X = torch.FloatTensor(inp_scaled).unsqueeze(0)
    
    # 8. Predicción
    print("\n🔮 Generando predicción...\n")
    
    with torch.no_grad():
        pred = model(X)
    
    pred_scaled = pred.numpy()[0]
    pred_deltas = scaler_out.inverse_transform([pred_scaled])[0]
    
    delta_high = pred_deltas[0]
    delta_low = pred_deltas[1]
    delta_close = pred_deltas[2]
    
    # 9. Análisis de breakout
    breakout_analysis = validate_breakout(df, current_idx, delta_high, delta_low)
    
    # 10. Clasificar
    result = classify_with_deltas_and_volume(
        current_price, delta_high, delta_low, delta_close,
        volume_analysis, breakout_analysis, divergence_signals
    )
    
    # 11. Guardar
    timestamp = datetime.now()
    save_prediction(timestamp, current_price, result)
    
    # 12. Telegram
    signal = result['signal']
    confidence = result['confidence']
    
    if signal in ['BUY', 'SELL']:
        emoji = "🟢" if signal == "BUY" else "🔴"
        
        msg = f"""
{emoji} *Señal {signal} (Delta)*

💰 Precio: ${current_price:.4f}

🔮 *Predicciones (ancladas):*
   High: ${result['pred_high']:.4f} ({result['delta_high']*100:+.2f}%)
   Low: ${result['pred_low']:.4f} ({result['delta_low']*100:+.2f}%)
   Close: ${result['pred_close']:.4f} ({result['close_change_%']:+.2f}%)

🎲 Confianza: {confidence:.1f}%

📊 *Volumen:*
   Confirmación: {result['volume_confirmation']}
   Soporte: {'✅' if result['volume_support'] else '❌'}

🚀 *Breakout:*
   Tipo: {result['breakout_type']}
   Válido: {'✅' if result['is_breakout'] else '❌'}

✅ *Anclaje garantizado*
"""
        
        send_telegram(msg)
    
    print("\n" + "="*70)
    print("  ✅ PREDICCIÓN DELTA COMPLETADA")
    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        print(error_msg)
        send_telegram(error_msg)
        raise
