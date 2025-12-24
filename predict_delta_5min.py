"""
PREDICCIÓN CON DELTAS HÍBRIDOS - VERSIÓN MEJORADA
✅ Usa modelo híbrido LSTM con atención
✅ Predice deltas relativos al precio actual
✅ Incluye análisis de volumen avanzado
✅ Validación de breakouts y divergencias
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
import os
import warnings
from datetime import datetime
import yfinance as yf
import requests
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')

# Telegram
TELEGRAM_API = os.environ.get('TELEGRAM_API', '')
CHAT_ID = os.environ.get('CHAT_ID', '')

# Archivos
PREDICTION_TRACKER_FILE = 'prediction_tracker.csv'
SIGNALS_FILE = 'trading_signals.csv'
MODEL_DIR = 'ADAUSD_MODELS'

# Configuración
MIN_SIGNAL_INTERVAL_SECONDS = 240  # 4 minutos
SEQ_LEN = 60

def send_telegram(msg):
    """Envía mensaje a Telegram"""
    if not TELEGRAM_API or not CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_API}/sendMessage"
        requests.post(url, data={'chat_id': CHAT_ID, 'text': msg, 'parse_mode': 'Markdown'}, timeout=10)
    except Exception as e:
        print(f"❌ Telegram: {e}")

# ================================
# 🧠 MODELO HÍBRIDO (igual que en entrenamiento)
# ================================
class HybridLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.25, bidirectional=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        fc_input_size = hidden_size * self.num_directions
        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input_size, fc_input_size // 2),
            nn.BatchNorm1d(fc_input_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_input_size // 2, fc_input_size // 4),
            nn.BatchNorm1d(fc_input_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_input_size // 4, output_size)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        return self.fc_layers(context)

# ================================
# 📊 INDICADORES AVANZADOS
# ================================
def calculate_advanced_indicators(df):
    """Calcula los mismos indicadores que en entrenamiento"""
    df = df.copy()
    
    # Derivadas de volumen
    df['volume_1st_deriv'] = df['volume'].diff()
    df['volume_2nd_deriv'] = df['volume_1st_deriv'].diff()
    
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
    
    # OBV ROC
    df['obv_roc'] = df['obv'].pct_change(periods=14)
    
    # VWAP
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    
    # PVT
    df['pvt'] = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1) * df['volume']).cumsum()
    
    # Volume Ratio
    df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma_20']
    
    # Volume RSI
    def calculate_volume_rsi(volume, period=14):
        gains = np.where(volume.diff() > 0, volume.diff(), 0)
        losses = np.where(volume.diff() < 0, -volume.diff(), 0)
        avg_gain = pd.Series(gains).rolling(window=period).mean()
        avg_loss = pd.Series(losses).rolling(window=period).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    df['volume_rsi'] = calculate_volume_rsi(df['volume'])
    
    # Derivadas de precio
    df['price_1st_deriv'] = df['close'].diff()
    df['price_2nd_deriv'] = df['price_1st_deriv'].diff()
    
    # Divergencias
    price_slope = df['close'].rolling(window=14).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 14 else 0
    )
    obv_slope = df['obv'].rolling(window=14).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 14 else 0
    )
    
    df['bullish_divergence'] = ((price_slope < 0) & (obv_slope > 0)).astype(int)
    df['bearish_divergence'] = ((price_slope > 0) & (obv_slope < 0)).astype(int)
    
    df.fillna(method='bfill', inplace=True)
    df.fillna(0, inplace=True)
    
    return df

# ================================
# 🔍 ANÁLISIS DE VOLUMEN (artículo MQL5)
# ================================
def analyze_volume_breakout(df):
    """Analiza breakouts basados en volumen"""
    if len(df) < 50:
        return {'breakout': False, 'type': 'NONE', 'confidence': 0}
    
    recent = df.iloc[-50:]
    
    # Resistencia y soporte
    resistance = recent['high'].max()
    support = recent['low'].min()
    
    current_price = recent['close'].iloc[-1]
    current_volume = recent['volume'].iloc[-1]
    avg_volume = recent['volume'].mean()
    
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
    
    # Detectar breakout
    if current_price > resistance and volume_ratio > 1.5:
        return {
            'breakout': True,
            'type': 'BULLISH',
            'confidence': min(volume_ratio * 30, 95),
            'volume_spike': volume_ratio
        }
    elif current_price < support and volume_ratio > 1.5:
        return {
            'breakout': True,
            'type': 'BEARISH',
            'confidence': min(volume_ratio * 30, 95),
            'volume_spike': volume_ratio
        }
    
    return {'breakout': False, 'type': 'NONE', 'confidence': 0}

def analyze_trend_confirmation(df):
    """Confirma tendencia usando volumen"""
    if len(df) < 20:
        return {'trend': 'NEUTRAL', 'strength': 0, 'confirmed': False}
    
    recent = df.iloc[-20:]
    
    price_change = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]
    volume_change = recent['volume'].iloc[-1] / recent['volume'].mean()
    
    # Confirmación de tendencia (artículo MQL5)
    if price_change > 0.01 and volume_change > 1.2:
        return {
            'trend': 'BULLISH',
            'strength': min(price_change * 100 + volume_change * 10, 100),
            'confirmed': True
        }
    elif price_change < -0.01 and volume_change > 1.2:
        return {
            'trend': 'BEARISH',
            'strength': min(abs(price_change) * 100 + volume_change * 10, 100),
            'confirmed': True
        }
    
    return {'trend': 'NEUTRAL', 'strength': 0, 'confirmed': False}

# ================================
# 🎯 CLASIFICACIÓN CON DELTAS
# ================================
def classify_with_deltas(current_price, delta_high, delta_low, delta_close, 
                        volume_analysis, breakout_analysis):
    """
    Clasifica señal usando deltas predichos
    """
    # Convertir deltas a precios absolutos
    pred_high = current_price * (1 + delta_high)
    pred_low = current_price * (1 + delta_low)
    pred_close = current_price * (1 + delta_close)
    
    # Garantizar coherencia
    pred_high, pred_low = max(pred_high, pred_low), min(pred_high, pred_low)
    pred_close = np.clip(pred_close, pred_low, pred_high)
    
    print(f"\n🎯 DELTAS PREDICHOS:")
    print(f"   Precio actual: ${current_price:.4f}")
    print(f"   Delta High: {delta_high*100:+.2f}% → ${pred_high:.4f}")
    print(f"   Delta Low:  {delta_low*100:+.2f}% → ${pred_low:.4f}")
    print(f"   Delta Close: {delta_close*100:+.2f}% → ${pred_close:.4f}")
    
    # Señal base del delta_close
    signal = "HOLD"
    base_confidence = 50
    
    if delta_close > 0.003:  # +0.3%
        signal = "BUY"
        base_confidence = 60 + min(delta_close * 500, 25)
    elif delta_close < -0.003:  # -0.3%
        signal = "SELL"
        base_confidence = 60 + min(abs(delta_close) * 500, 25)
    
    # Ajuste por volumen
    if volume_analysis['confirmed']:
        if volume_analysis['trend'] == 'BULLISH' and signal == 'BUY':
            base_confidence += 15
            print(f"   ✅ Volumen confirma BUY (+15)")
        elif volume_analysis['trend'] == 'BEARISH' and signal == 'SELL':
            base_confidence += 15
            print(f"   ✅ Volumen confirma SELL (+15)")
        elif volume_analysis['trend'] != signal:
            base_confidence -= 20
            print(f"   ⚠️ Volumen contradice señal (-20)")
    
    # Ajuste por breakout
    if breakout_analysis['breakout']:
        if breakout_analysis['type'] == 'BULLISH' and signal == 'BUY':
            base_confidence += breakout_analysis['confidence']
            print(f"   ✅ Breakout alcista (+{breakout_analysis['confidence']:.0f})")
        elif breakout_analysis['type'] == 'BEARISH' and signal == 'SELL':
            base_confidence += breakout_analysis['confidence']
            print(f"   ✅ Breakout bajista (+{breakout_analysis['confidence']:.0f})")
    
    # Confianza final
    confidence = max(30, min(95, base_confidence))
    
    # Umbral de confianza
    if confidence < 65:
        signal = "HOLD"
        confidence = 50
    
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
        'volume_trend': volume_analysis['trend'],
        'volume_confirmed': volume_analysis['confirmed'],
        'breakout': breakout_analysis['breakout'],
        'breakout_type': breakout_analysis['type']
    }

# ================================
# 💾 GUARDAR PREDICCIÓN
# ================================
def save_prediction(timestamp, current_price, result):
    """Guarda predicción en CSV"""
    
    # Prediction tracker
    tracking_data = {
        'timestamp': timestamp,
        'current_price': float(round(current_price, 4)),
        'pred_high': float(round(result['pred_high'], 4)),
        'pred_low': float(round(result['pred_low'], 4)),
        'pred_close': float(round(result['pred_close'], 4)),
        'delta_high_%': float(round(result['delta_high'] * 100, 3)),
        'delta_low_%': float(round(result['delta_low'] * 100, 3)),
        'delta_close_%': float(round(result['delta_close'] * 100, 3)),
        'signal': str(result['signal']),
        'confidence': float(round(result['confidence'], 1)),
        'volume_trend': str(result['volume_trend']),
        'volume_confirmed': bool(result['volume_confirmed']),
        'breakout': bool(result['breakout']),
        'breakout_type': str(result['breakout_type'])
    }
    
    df_track = pd.DataFrame([tracking_data])
    
    if os.path.exists(PREDICTION_TRACKER_FILE):
        existing = pd.read_csv(PREDICTION_TRACKER_FILE)
        df_track = pd.concat([existing, df_track], ignore_index=True)
    
    df_track.to_csv(PREDICTION_TRACKER_FILE, index=False)
    
    # Trading signals (solo para señales BUY/SELL)
    if result['signal'] in ['BUY', 'SELL']:
        signal_data = {
            'timestamp': timestamp,
            'current_price': float(round(current_price, 4)),
            'pred_high': float(round(result['pred_high'], 4)),
            'pred_low': float(round(result['pred_low'], 4)),
            'pred_close': float(round(result['pred_close'], 4)),
            'pred_change_%': float(round(result['delta_close'] * 100, 2)),
            'atr': 0.0,  # Placeholder
            'volatility': float(round((result['pred_high'] - result['pred_low']) / current_price * 100, 2)),
            'trend': str(result['volume_trend']),
            'signal': str(result['signal']),
            'confidence': float(round(result['confidence'], 1)),
            'rsi': 0.0  # Placeholder
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
    print("  🔮 PREDICCIÓN CON DELTAS HÍBRIDOS")
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
    model_path = f'{MODEL_DIR}/adausd_hybrid_lstm.pth'
    config_path = f'{MODEL_DIR}/config_hybrid.json'
    
    if not os.path.exists(model_path):
        error_msg = f"❌ No existe modelo híbrido entrenado"
        print(error_msg)
        send_telegram(error_msg)
        return
    
    print("📂 Cargando modelo híbrido...")
    
    try:
        # Cargar configuración
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        input_size = config['input_size']
        output_size = config['output_size']
        hidden_size = config['hidden_size']
        num_layers = config['num_layers']
        seq_len = config['seq_len']
        feature_cols = config['feature_cols']
        target_cols = config['target_cols']
        
        # Cargar modelo
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        model = HybridLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=0.25,
            bidirectional=True
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Cargar scalers
        scaler_in = joblib.load(f'{MODEL_DIR}/scaler_input_hybrid.pkl')
        scaler_out = joblib.load(f'{MODEL_DIR}/scaler_output_hybrid.pkl')
        
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
        
        # Renombrar columna de tiempo
        time_col = next((col for col in df.columns if 'date' in col or 'time' in col), None)
        if time_col:
            df = df.rename(columns={time_col: 'time'})
        
        # Asegurar columnas OHLCV
        required_cols = {'time', 'open', 'high', 'low', 'close', 'volume'}
        df = df[[col for col in df.columns if col in required_cols]]
        
        # Convertir tiempo
        df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
        
        print(f"✅ {len(df)} velas descargadas\n")
        
    except Exception as e:
        error_msg = f"❌ Error descargando datos: {str(e)}"
        print(error_msg)
        send_telegram(error_msg)
        return
    
    # 4. Calcular indicadores avanzados
    print("📊 Calculando indicadores avanzados...")
    df = calculate_advanced_indicators(df)
    
    # 5. Preparar input para el modelo
    if len(df) < seq_len + 10:
        error_msg = f"❌ Datos insuficientes: {len(df)} < {seq_len + 10}"
        print(error_msg)
        send_telegram(error_msg)
        return
    
    # Tomar las últimas secuencias
    current_data = df[feature_cols].tail(seq_len).values
    
    # Verificar que tenemos todas las columnas
    if current_data.shape[1] != len(feature_cols):
        error_msg = f"❌ Columnas inconsistentes: esperadas {len(feature_cols)}, obtenidas {current_data.shape[1]}"
        print(error_msg)
        send_telegram(error_msg)
        return
    
    # Escalar
    current_scaled = scaler_in.transform(current_data)
    
    # Convertir a tensor
    X = torch.FloatTensor(current_scaled).unsqueeze(0)  # [1, seq_len, features]
    
    # 6. Predicción
    print("\n🔮 Generando predicción...\n")
    
    with torch.no_grad():
        pred_scaled = model(X)
    
    # Desescalar predicción
    pred = scaler_out.inverse_transform(pred_scaled.numpy())[0]
    
    delta_high, delta_low, delta_close = pred
    
    # 7. Análisis de volumen y breakouts
    volume_analysis = analyze_trend_confirmation(df)
    breakout_analysis = analyze_volume_breakout(df)
    
    current_price = df['close'].iloc[-1]
    
    # 8. Clasificar señal
    result = classify_with_deltas(
        current_price,
        delta_high,
        delta_low,
        delta_close,
        volume_analysis,
        breakout_analysis
    )
    
    # 9. Guardar predicción
    timestamp = datetime.now()
    save_prediction(timestamp, current_price, result)
    
    # 10. Enviar Telegram si es señal
    if result['signal'] in ['BUY', 'SELL']:
        emoji = "🟢" if result['signal'] == "BUY" else "🔴"
        
        msg = f"""
{emoji} *Señal {result['signal']} (Híbrido)*

💰 Precio: ${current_price:.4f}

🔮 *Predicciones (deltas):*
   High: ${result['pred_high']:.4f} ({result['delta_high']*100:+.2f}%)
   Low: ${result['pred_low']:.4f} ({result['delta_low']*100:+.2f}%)
   Close: ${result['pred_close']:.4f} ({result['delta_close']*100:+.2f}%)

🎲 Confianza: {result['confidence']:.1f}%

📊 *Volumen:*
   Tendencia: {result['volume_trend']}
   Confirmado: {'✅' if result['volume_confirmed'] else '❌'}

🚀 *Breakout:*
   {'✅' if result['breakout'] else '❌'} {result['breakout_type']}

✅ *Modelo Híbrido LSTM con Atención*
"""
        
        send_telegram(msg)
    
    print("\n" + "="*70)
    print("  ✅ PREDICCIÓN HÍBRIDA COMPLETADA")
    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        send_telegram(error_msg)
        raise
