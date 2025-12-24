"""
ENTRENAMIENTO LSTM HÍBRIDO CON DELTAS Y VOLUMEN MEJORADO
✅ Predice DELTAS RELATIVOS con métricas optimizadas
✅ Incluye indicadores de volumen avanzados (OBV, VWAP, PVT)
✅ Sistema híbrido con derivadas de volumen (1ra y 2da)
✅ Normalización inteligente por ventana
✅ Pérdida personalizada con restricciones de coherencia
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
import time
import json
import joblib
import yfinance as yf
from tqdm.auto import tqdm
import requests
from datetime import datetime, timedelta
import warnings

from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore')

# ================================
# 🎛️ CONFIGURACIÓN MEJORADA
# ================================
# ================================
# 🎛️ CONFIGURACIÓN OPTIMIZADA
# ================================
class Config:
    # Características
    USE_VOLUME = True
    USE_VOLUME_DERIVATIVES = False  # Menos ruido
    USE_VOLUME_INDICATORS = True
    USE_PRICE_DERIVATIVES = False
    
    # Targets
    PREDICT_ABSOLUTE = True         # Precios absolutos
    PREDICT_DELTAS = False
    
    # Normalización
    SCALER_TYPE = 'robust'
    
    # Arquitectura
    SEQ_LEN = 30                    # Ventana más corta
    HIDDEN_SIZE = 64                # Reducido
    NUM_LAYERS = 2
    DROPOUT = 0.3
    
    # Entrenamiento
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    PATIENCE = 10
    
    # Datos
    TRAIN_SIZE = 0.70
    VAL_SIZE = 0.15
    TEST_SIZE = 0.15

# Telegram
TELEGRAM_API = os.environ.get('TELEGRAM_API', '')
CHAT_ID = os.environ.get('CHAT_ID', '')

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
# 📊 PREPARACIÓN DE DATOS MEJORADA
# ================================
def calculate_basic_indicators(df):
    """Indicadores básicos y confiables"""
    df = df.copy()
    
    # Volumen
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # Precio
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=20).std()
    
    # RSI simple
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Rellenar NaN
    df.fillna(method='bfill', inplace=True)
    df.fillna(0, inplace=True)
    
    return df

# ================================
# 📊 INDICADORES AVANZADOS
# ================================
def calculate_advanced_indicators(df):
    """
    Calcula indicadores avanzados basados en el artículo de MQL5
    Incluye derivadas de volumen y osciladores
    """
    df = df.copy()
    
    # 1. Derivadas de volumen (1ra y 2da)
    if Config.USE_VOLUME_DERIVATIVES:
        df['volume_1st_deriv'] = df['volume'].diff()
        df['volume_2nd_deriv'] = df['volume_1st_deriv'].diff()
    
    # 2. On-Balance Volume (OBV)
    obv = [0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.append(obv[-1] + df['volume'].iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.append(obv[-1] - df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['obv'] = obv
    
    # 3. OBV Rate of Change
    df['obv_roc'] = df['obv'].pct_change(periods=14)
    
    # 4. Volume Weighted Average Price (VWAP)
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    
    # 5. Price-Volume Trend (PVT)
    df['pvt'] = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1) * df['volume']).cumsum()
    
    # 6. Volume Ratio (actual vs promedio 20)
    df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma_20']
    
    # 7. Derivadas de precio
    if Config.USE_PRICE_DERIVATIVES:
        df['price_1st_deriv'] = df['close'].diff()
        df['price_2nd_deriv'] = df['price_1st_deriv'].diff()

     # ==== SOLUCIÓN 2: Cálculo robusto de derivadas ====
    if 'volume' in df.columns:
        # Primera derivada del volumen
        volume_diff = df['volume'].diff()
        df['volume_1st_deriv'] = volume_diff
        
        # Segunda derivada del volumen
        df['volume_2nd_deriv'] = volume_diff.diff()
        
        # Normalizar derivadas si hay suficiente variación
        for deriv_col in ['volume_1st_deriv', 'volume_2nd_deriv']:
            if df[deriv_col].std() > 1e-10:  # Evitar división por cero
                # Normalizar usando puntuación Z, pero recortando outliers
                mean_val = df[deriv_col].mean()
                std_val = df[deriv_col].std()
                
                # Calcular puntuación Z
                z_scores = (df[deriv_col] - mean_val) / std_val
                
                # Recortar valores extremos (más de 5 desviaciones estándar)
                z_scores_clipped = np.clip(z_scores, -5, 5)
                
                # Convertir de vuelta a escala original pero sin outliers extremos
                df[deriv_col] = z_scores_clipped * std_val + mean_val
    
    # Derivada del precio con manejo robusto
    price_diff = df['close'].diff()
    df['price_1st_deriv'] = price_diff
    
    if df['price_1st_deriv'].std() > 1e-10:
        mean_price_deriv = df['price_1st_deriv'].mean()
        std_price_deriv = df['price_1st_deriv'].std()
        z_scores_price = (df['price_1st_deriv'] - mean_price_deriv) / std_price_deriv
        z_scores_price_clipped = np.clip(z_scores_price, -5, 5)
        df['price_1st_deriv'] = z_scores_price_clipped * std_price_deriv + mean_price_deriv

    # ==== SOLUCIÓN 1: Limpieza de NaN/Inf después de cálculos ====
    # Reemplazar infinitos con NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Eliminar filas con NaN
    initial_rows = len(df)
    df = df.dropna()
    rows_removed = initial_rows - len(df)
    
    if rows_removed > 0:
        print(f"⚠️  Se eliminaron {rows_removed} filas con valores NaN/Inf")
        print(f"📊 Filas restantes: {len(df)}")
    
    # Al final, aplicar limpieza completa
    df = clean_financial_data(df, max_abs_value=1e6, fill_method='ffill')
    
    return df
    
    # 8. RSI de volumen
    def calculate_volume_rsi(volume, period=14):
        gains = np.where(volume.diff() > 0, volume.diff(), 0)
        losses = np.where(volume.diff() < 0, -volume.diff(), 0)
        
        avg_gain = pd.Series(gains).rolling(window=period).mean()
        avg_loss = pd.Series(losses).rolling(window=period).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    df['volume_rsi'] = calculate_volume_rsi(df['volume'])
    
    # 9. Divergencias
    # Precio vs OBV
    price_slope = df['close'].rolling(window=14).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 14 else 0
    )
    obv_slope = df['obv'].rolling(window=14).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 14 else 0
    )
    
    df['bullish_divergence'] = ((price_slope < 0) & (obv_slope > 0)).astype(int)
    df['bearish_divergence'] = ((price_slope > 0) & (obv_slope < 0)).astype(int)
    
    # Rellenar NaNs
    df.fillna(method='bfill', inplace=True)
    df.fillna(0, inplace=True)
    
    return df

# ================================
# 🧠 MODELO SIMPLIFICADO
# ================================
class PricePredictorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=3, dropout=0.3):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Atención simple
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        # Salida
        return self.fc(context)

class CoherentDeltaLoss(nn.Module):
    """
    Pérdida personalizada que garantiza coherencia:
    1. delta_high >= delta_low
    2. delta_close entre delta_low y delta_high
    3. Deltas realistas (no extremos)
    """
    def __init__(self, mse_weight=1.0, constraint_weight=5.0, realism_weight=2.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.mse_weight = mse_weight
        self.constraint_weight = constraint_weight
        self.realism_weight = realism_weight
    
    def forward(self, predictions, targets):
        # MSE base
        mse_loss = self.mse(predictions, targets)
        
        # Constraint 1: delta_high >= delta_low
        delta_high = predictions[:, 0]
        delta_low = predictions[:, 1]
        delta_close = predictions[:, 2]
        
        high_low_violation = torch.clamp(delta_low - delta_high, min=0)
        constraint_loss_1 = high_low_violation.mean()
        
        # Constraint 2: close entre high y low
        close_below_low = torch.clamp(delta_low - delta_close, min=0)
        close_above_high = torch.clamp(delta_close - delta_high, min=0)
        constraint_loss_2 = (close_below_low + close_above_high).mean()
        
        # Realism: deltas razonables (-10% a +10%)
        max_delta = 0.10
        extreme_high = torch.clamp(torch.abs(delta_high) - max_delta, min=0)
        extreme_low = torch.clamp(torch.abs(delta_low) - max_delta, min=0)
        extreme_close = torch.clamp(torch.abs(delta_close) - max_delta, min=0)
        realism_loss = (extreme_high + extreme_low + extreme_close).mean()
        
        # Pérdida total
        total_loss = (
            self.mse_weight * mse_loss +
            self.constraint_weight * (constraint_loss_1 + constraint_loss_2) +
            self.realism_weight * realism_loss
        )
        
        return total_loss, {
            'mse': mse_loss.item(),
            'constraint': (constraint_loss_1 + constraint_loss_2).item(),
            'realism': realism_loss.item()
        }

# ================================
# 📦 PREPARACIÓN DE DATOS
# ================================
def download_and_prepare_data(symbol="ADA-USD", interval='1h', path='ADAUSD_1h_data.csv'):
    """Descarga y prepara datos OHLCV"""
    print("="*70)
    print(f"  📥 DESCARGA DE DATOS OHLCV - {interval.upper()}")
    print("="*70 + "\n")
    
    # Verificar si hay datos recientes
    if os.path.exists(path):
        try:
            df_exist = pd.read_csv(path, nrows=5)
            if len(df_exist) > 0:
                print(f"✅ Datos existentes en {path}")
                return pd.read_csv(path)
        except:
            pass
    
    # Descargar nuevos datos
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="730d", interval=interval)  # 2 años
    
    if len(df) == 0:
        raise ValueError(f"No se pudieron descargar datos para {symbol}")
    
    # Formatear
    df = df.reset_index()
    df.columns = [str(c).lower() for c in df.columns]
    
    # Renombrar columnas
    rename_dict = {}
    for col in df.columns:
        if 'date' in col or 'datetime' in col:
            rename_dict[col] = 'time'
    df.rename(columns=rename_dict, inplace=True)
    
    # Asegurar columnas OHLCV
    required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
    df = df[required_cols]
    
    # Convertir tiempo
    df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
    
    # Guardar
    df.to_csv(path, index=False)
    print(f"✅ Datos guardados: {len(df):,} velas")
    return df

def prepare_improved_dataset(df):
    """Dataset mejorado con targets absolutos"""
    print("\n" + "="*70)
    print("  🔧 PREPARACIÓN MEJORADA DE DATOS")
    print("="*70)
    
    # 1. Indicadores básicos
    df = calculate_basic_indicators(df)
    
    # 2. Targets: precios absolutos de la próxima vela
    df['high_next'] = df['high'].shift(-1)
    df['low_next'] = df['low'].shift(-1)
    df['close_next'] = df['close'].shift(-1)
    
    # 3. Features básicas
    basic_cols = ['open', 'high', 'low', 'close', 'volume']
    indicator_cols = ['volume_ma', 'volume_ratio', 'returns', 'volatility', 'rsi']
    
    feature_cols = basic_cols + indicator_cols
    target_cols = ['high_next', 'low_next', 'close_next']
    
    # 4. Eliminar NaN
    df = df.dropna()
    print(f"📊 Datos totales: {len(df):,} velas")
    
    # 5. Dividir temporalmente
    train_size = Config.TRAIN_SIZE
    val_size = Config.VAL_SIZE
    
    train_end = int(len(df) * train_size)
    val_end = int(len(df) * (train_size + val_size))
    
    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()
    
    # 6. Normalización separada
    scaler_in = RobustScaler()
    scaler_out = MinMaxScaler()
    
    X_train = scaler_in.fit_transform(df_train[feature_cols])
    X_val = scaler_in.transform(df_val[feature_cols])
    X_test = scaler_in.transform(df_test[feature_cols])
    
    y_train = scaler_out.fit_transform(df_train[target_cols])
    y_val = scaler_out.transform(df_val[target_cols])
    y_test = scaler_out.transform(df_test[target_cols])
    
    # 7. Crear secuencias
    def create_sequences(X, y, seq_len):
        X_seq, y_seq = [], []
        for i in range(seq_len, len(X)):
            X_seq.append(X[i-seq_len:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)
    
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, Config.SEQ_LEN)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, Config.SEQ_LEN)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, Config.SEQ_LEN)
    
    print(f"\n📊 Conjuntos de datos:")
    print(f"   Train: X{X_train_seq.shape}, y{y_train_seq.shape}")
    print(f"   Val:   X{X_val_seq.shape}, y{y_val_seq.shape}")
    print(f"   Test:  X{X_test_seq.shape}, y{y_test_seq.shape}")
    
    return (X_train_seq, y_train_seq), (X_val_seq, y_val_seq), (X_test_seq, y_test_seq), \
           scaler_in, scaler_out, feature_cols, target_cols

def prepare_delta_dataset(df):
    """
    🔥 PREPARA DATOS CON DELTAS MEJORADOS
    """
    print("\n" + "="*70)
    print("  🔧 PREPARACIÓN DE DATOS CON DELTAS MEJORADOS")
    print("="*70)
    
    # 1. Calcular indicadores avanzados
    df = calculate_advanced_indicators(df)

    # ==== CORRECCIÓN: Verificar solo columnas numéricas ====
    # Aplicar limpieza adicional si es necesario
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if df[numeric_cols].isnull().any().any() or np.isinf(df[numeric_cols].values).any():
        print("Aplicando limpieza adicional...")
        df = clean_financial_data(df, max_abs_value=1e6, fill_method='ffill')
    
    # 2. Calcular deltas para la próxima vela
    df['delta_high'] = (df['high'].shift(-1) - df['close']) / df['close']
    df['delta_low'] = (df['low'].shift(-1) - df['close']) / df['close']
    df['delta_close'] = (df['close'].shift(-1) - df['close']) / df['close']
    
    # 3. Eliminar NaNs
    initial_len = len(df)
    df = df.dropna()
    print(f"📊 Datos después de limpieza: {len(df):,} de {initial_len:,} velas")
    
    # 4. Seleccionar features (solo columnas numéricas relevantes)
    # Primero obtener todas las columnas numéricas
    all_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Quitar las columnas que no queremos como features
    exclude_cols = ['delta_high', 'delta_low', 'delta_close']
    feature_cols = [col for col in all_numeric_cols if col not in exclude_cols]
    
    # Asegurar que tenemos las columnas básicas
    basic_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in basic_cols:
        if col in df.columns and col not in feature_cols:
            feature_cols.append(col)
    
    # Verificar indicadores específicos si están configurados
    if Config.USE_VOLUME_INDICATORS:
        volume_indicators = ['obv', 'obv_roc', 'vwap', 'pvt', 'volume_ratio', 'volume_rsi']
        for indicator in volume_indicators:
            if indicator in df.columns and indicator not in feature_cols:
                feature_cols.append(indicator)
    
    if Config.USE_VOLUME_DERIVATIVES:
        volume_derivs = ['volume_1st_deriv', 'volume_2nd_deriv']
        for deriv in volume_derivs:
            if deriv in df.columns and deriv not in feature_cols:
                feature_cols.append(deriv)
    
    if Config.USE_PRICE_DERIVATIVES:
        price_derivs = ['price_1st_deriv', 'price_2nd_deriv']
        for deriv in price_derivs:
            if deriv in df.columns and deriv not in feature_cols:
                feature_cols.append(deriv)
    
    # Divergencias
    divergence_cols = ['bullish_divergence', 'bearish_divergence']
    for col in divergence_cols:
        if col in df.columns and col not in feature_cols:
            feature_cols.append(col)
    
    # Targets
    target_cols = ['delta_high', 'delta_low', 'delta_close']
    
    print(f"\n🎯 {len(feature_cols)} Features: {feature_cols}")
    print(f"🎯 {len(target_cols)} Targets: {target_cols}")
    
    # ==== CORRECCIÓN: Llamar a debug_data_issues con datos correctos ====
    # Llamar a la función de depuración
    problematic_cols = debug_data_issues(df, feature_cols)
    
    if problematic_cols:
        print("⚠️  Problemas detectados. Aplicando limpieza...")
        df = clean_financial_data(df, max_abs_value=1e6, fill_method='ffill')
    
    # 5. Dividir datos (temporal, no aleatorio)
    train_size = Config.TRAIN_SIZE
    val_size = Config.VAL_SIZE
    
    train_end = int(len(df) * train_size)
    val_end = int(len(df) * (train_size + val_size))
    
    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()
    
    print(f"\n📊 División temporal:")
    print(f"   Train: {len(df_train):,} ({train_size*100:.0f}%)")
    print(f"   Val:   {len(df_val):,} ({val_size*100:.0f}%)")
    print(f"   Test:  {len(df_test):,} ({Config.TEST_SIZE*100:.0f}%)")
    
    # 6. Normalización
    print(f"\n🔧 Normalización con {Config.SCALER_TYPE.upper()}...")
    
    # Usar RobustScaler para outliers
    scaler_in = RobustScaler(quantile_range=(25, 75))
    scaler_out = RobustScaler(quantile_range=(25, 75))
    
    print(f"📏 Usando scaler: {scaler_in.__class__.__name__}")
    
    # ==== CORRECCIÓN: Validación antes del escalado ====
    print("🔍 Validando datos antes del escalado...")
    
    # Verificar que solo estamos usando columnas numéricas
    for df_part in [df_train, df_val, df_test]:
        for col in feature_cols:
            if col not in df_part.select_dtypes(include=[np.number]).columns:
                print(f"❌ ERROR: Columna {col} no es numérica o no existe")
                raise ValueError(f"Columna {col} no es numérica")
    
    # 1. Verificar NaN solo en columnas numéricas
    nan_check_train = df_train[feature_cols].isnull().sum()
    if nan_check_train.any():
        print("❌ ERROR: Se encontraron valores NaN en train:")
        for col, count in nan_check_train[nan_check_train > 0].items():
            print(f"   - {col}: {count} NaN")
        raise ValueError("Datos train contienen NaN")
    
    # 2. Verificar infinitos solo en columnas numéricas
    inf_mask_train = np.isinf(df_train[feature_cols].values)
    if inf_mask_train.any():
        print("❌ ERROR: Se encontraron valores infinitos en train")
        inf_cols = []
        for i, col in enumerate(feature_cols):
            if inf_mask_train[:, i].any():
                inf_cols.append(col)
        print(f"   Columnas con infinitos: {inf_cols}")
        raise ValueError("Datos train contienen valores infinitos")
    
    # 3. Verificar valores extremos
    print("📊 Estadísticas de características (train):")
    for col in feature_cols:
        col_min = df_train[col].min()
        col_max = df_train[col].max()
        col_mean = df_train[col].mean()
        col_std = df_train[col].std()
        
        print(f"   {col}: min={col_min:.6f}, max={col_max:.6f}, "
              f"mean={col_mean:.6f}, std={col_std:.6f}")
        
        # Detectar valores sospechosamente grandes
        if abs(col_max) > 1e6 or abs(col_min) > 1e6:
            print(f"   ⚠️  Advertencia: {col} tiene valores > 1e6")
            # Recortar valores extremos
            df_train[col] = np.clip(df_train[col], -1e6, 1e6)
            df_val[col] = np.clip(df_val[col], -1e6, 1e6)
            df_test[col] = np.clip(df_test[col], -1e6, 1e6)
    
    # Ahora proceder con el escalado
    print("✅ Validación completada. Aplicando escalado...")
    
    # Fit en train
    X_train = scaler_in.fit_transform(df_train[feature_cols])
    y_train = scaler_out.fit_transform(df_train[target_cols])
    
    X_val = scaler_in.transform(df_val[feature_cols])
    y_val = scaler_out.transform(df_val[target_cols])
    
    X_test = scaler_in.transform(df_test[feature_cols])
    y_test = scaler_out.transform(df_test[target_cols])
    
    # 7. Crear secuencias
    def create_sequences(X, y, seq_len):
        X_seq, y_seq = [], []
        for i in range(seq_len, len(X)):
            X_seq.append(X[i-seq_len:i])
            y_seq.append(y[i-1])  # Target es la vela después de la secuencia
        return np.array(X_seq), np.array(y_seq)
    
    print("\n🔄 Creando secuencias...")
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, Config.SEQ_LEN)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, Config.SEQ_LEN)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, Config.SEQ_LEN)
    
    print(f"✅ Train: X{X_train_seq.shape}, y{y_train_seq.shape}")
    print(f"✅ Val:   X{X_val_seq.shape}, y{y_val_seq.shape}")
    print(f"✅ Test:  X{X_test_seq.shape}, y{y_test_seq.shape}")
    
    return (X_train_seq, y_train_seq), (X_val_seq, y_val_seq), (X_test_seq, y_test_seq), \
           scaler_in, scaler_out, feature_cols, target_cols

def prepare_improved_dataset(df):
    """Dataset mejorado con targets absolutos"""
    print("\n" + "="*70)
    print("  🔧 PREPARACIÓN MEJORADA DE DATOS")
    print("="*70)
    
    # 1. Indicadores básicos
    df = calculate_basic_indicators(df)
    
    # 2. Targets: precios absolutos de la próxima vela
    df['high_next'] = df['high'].shift(-1)
    df['low_next'] = df['low'].shift(-1)
    df['close_next'] = df['close'].shift(-1)
    
    # 3. Features básicas
    basic_cols = ['open', 'high', 'low', 'close', 'volume']
    indicator_cols = ['volume_ma', 'volume_ratio', 'returns', 'volatility', 'rsi']
    
    feature_cols = basic_cols + indicator_cols
    target_cols = ['high_next', 'low_next', 'close_next']
    
    # 4. Eliminar NaN
    df = df.dropna()
    print(f"📊 Datos totales: {len(df):,} velas")
    
    # 5. Dividir temporalmente
    train_size = Config.TRAIN_SIZE
    val_size = Config.VAL_SIZE
    
    train_end = int(len(df) * train_size)
    val_end = int(len(df) * (train_size + val_size))
    
    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()
    
    # 6. Normalización separada
    scaler_in = RobustScaler()
    scaler_out = MinMaxScaler()
    
    X_train = scaler_in.fit_transform(df_train[feature_cols])
    X_val = scaler_in.transform(df_val[feature_cols])
    X_test = scaler_in.transform(df_test[feature_cols])
    
    y_train = scaler_out.fit_transform(df_train[target_cols])
    y_val = scaler_out.transform(df_val[target_cols])
    y_test = scaler_out.transform(df_test[target_cols])
    
    # 7. Crear secuencias
    def create_sequences(X, y, seq_len):
        X_seq, y_seq = [], []
        for i in range(seq_len, len(X)):
            X_seq.append(X[i-seq_len:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)
    
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, Config.SEQ_LEN)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, Config.SEQ_LEN)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, Config.SEQ_LEN)
    
    print(f"\n📊 Conjuntos de datos:")
    print(f"   Train: X{X_train_seq.shape}, y{y_train_seq.shape}")
    print(f"   Val:   X{X_val_seq.shape}, y{y_val_seq.shape}")
    print(f"   Test:  X{X_test_seq.shape}, y{y_test_seq.shape}")
    
    return (X_train_seq, y_train_seq), (X_val_seq, y_val_seq), (X_test_seq, y_test_seq), \
           scaler_in, scaler_out, feature_cols, target_cols

def clean_financial_data(df, max_abs_value=1e6, fill_method='ffill'):
    """
    Pipeline completo para limpiar datos financieros
    
    Parámetros:
    - df: DataFrame con datos financieros
    - max_abs_value: valor máximo absoluto permitido (valores más grandes se recortan)
    - fill_method: método para rellenar NaN ('ffill', 'bfill', 'mean', 'median', 'zero')
    
    Retorna:
    - DataFrame limpio
    """
    print("\n🧹 INICIANDO LIMPIEZA DE DATOS FINANCIEROS")
    initial_shape = df.shape
    
    # 1. Hacer copia para no modificar el original
    df_clean = df.copy()
    
    # 2. Identificar columnas numéricas
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    print(f"   Columnas numéricas detectadas: {len(numeric_cols)}")
    
    # 3. Paso 1: Reemplazar infinitos con NaN
    inf_count_before = np.isinf(df_clean[numeric_cols].values).sum()
    df_clean[numeric_cols] = df_clean[numeric_cols].replace([np.inf, -np.inf], np.nan)
    print(f"   Paso 1: Reemplazados {inf_count_before} valores infinitos con NaN")
    
    # 4. Paso 2: Rellenar NaN según el método especificado
    nan_count_before = df_clean[numeric_cols].isnull().sum().sum()
    
    if fill_method == 'ffill':
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(method='ffill')
    elif fill_method == 'bfill':
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(method='bfill')
    elif fill_method == 'mean':
        for col in numeric_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
    elif fill_method == 'median':
        for col in numeric_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    elif fill_method == 'zero':
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(0)
    else:  # Por defecto: ffill + bfill
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(method='ffill').fillna(method='bfill')
    
    nan_count_after = df_clean[numeric_cols].isnull().sum().sum()
    print(f"   Paso 2: Rellenados {nan_count_before - nan_count_after} NaN usando método '{fill_method}'")
    
    # 5. Paso 3: Si aún hay NaN (todas las columnas NaN en una fila), usar 0
    if nan_count_after > 0:
        df_clean = df_clean.fillna(0)
        print(f"   Paso 3: Rellenados {nan_count_after} NaN restantes con 0")
    
    # 6. Paso 4: Recortar valores extremos
    outliers_count = 0
    for col in numeric_cols:
        # Detectar outliers extremos
        col_abs_max = df_clean[col].abs().max()
        if col_abs_max > max_abs_value:
            outliers_before = (df_clean[col].abs() > max_abs_value).sum()
            df_clean[col] = np.clip(df_clean[col], -max_abs_value, max_abs_value)
            outliers_count += outliers_before
    
    if outliers_count > 0:
        print(f"   Paso 4: Recortados {outliers_count} valores extremos (> {max_abs_value})")
    
    # 7. Paso 5: Normalizar si hay valores muy pequeños (evitar underflow)
    tiny_values_count = 0
    for col in numeric_cols:
        col_max_abs = df_clean[col].abs().max()
        # Si todos los valores son muy pequeños pero no cero
        if 0 < col_max_abs < 1e-10:
            df_clean[col] = df_clean[col] * 1e10  # Escalar para evitar underflow
            tiny_values_count += 1
    
    if tiny_values_count > 0:
        print(f"   Paso 5: Escaladas {tiny_values_count} columnas con valores muy pequeños")
    
    # 8. Paso 6: Verificación final
    final_nan = df_clean[numeric_cols].isnull().sum().sum()
    final_inf = np.isinf(df_clean[numeric_cols].values).sum()
    
    if final_nan == 0 and final_inf == 0:
        print(f"   ✅ LIMPIEZA COMPLETADA EXITOSAMENTE")
    else:
        print(f"   ⚠️  LIMPIEZA PARCIAL: {final_nan} NaN y {final_inf} Inf restantes")
    
    print(f"   📊 Forma original: {initial_shape} → Forma final: {df_clean.shape}")
    print(f"   📈 Filas eliminadas: {initial_shape[0] - df_clean.shape[0]}")
    print(f"   🎯 Columnas procesadas: {len(numeric_cols)}\n")
    
    return df_clean


def debug_data_issues(df, feature_cols):
    """
    Función de depuración para identificar problemas específicos en los datos
    """
    print("\n" + "="*60)
    print("🔧 DIAGNÓSTICO DETALLADO DE DATOS")
    print("="*60)
    
    # Filtrar solo columnas numéricas que existen en el DataFrame
    available_feature_cols = [col for col in feature_cols if col in df.columns]
    print(f"📊 Resumen del DataFrame:")
    print(f"   - Filas totales: {len(df)}")
    print(f"   - Columnas totales: {len(df.columns)}")
    print(f"   - Columnas de características disponibles: {len(available_feature_cols)}/{len(feature_cols)}")
    
    # Columnas faltantes
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"⚠️  Columnas faltantes: {missing_cols}")
    
    # 2. Verificar cada columna disponible
    print(f"\n📈 Estadísticas por columna:")
    problematic_cols = []
    
    for i, col in enumerate(available_feature_cols, 1):
        print(f"\n   {i:2d}. {col}:")
        
        # Verificar si es columna numérica
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"      ❌ No es numérica, tipo: {df[col].dtype}")
            problematic_cols.append((col, 'Non-numeric'))
            continue
        
        # Valores NaN
        nan_count = df[col].isnull().sum()
        if nan_count > 0:
            print(f"      ❌ NaN: {nan_count}")
            problematic_cols.append((col, 'NaN'))
        
        # Valores infinitos (solo para numéricas)
        try:
            inf_mask = np.isinf(df[col].values)
            inf_count = inf_mask.sum()
            if inf_count > 0:
                print(f"      ❌ Infinitos: {inf_count}")
                problematic_cols.append((col, 'Inf'))
        except TypeError:
            print(f"      ⚠️  No se puede verificar infinitos (tipo: {df[col].dtype})")
            problematic_cols.append((col, 'TypeError'))
        
        # Estadísticas básicas
        if nan_count == 0 and inf_count == 0:
            try:
                print(f"      ✅ Min: {df[col].min():.6f}")
                print(f"      ✅ Max: {df[col].max():.6f}")
                print(f"      ✅ Media: {df[col].mean():.6f}")
                print(f"      ✅ Std: {df[col].std():.6f}")
                
                # Detectar outliers (más de 5 std de la media)
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    outliers = ((df[col] - mean).abs() > 5 * std).sum()
                    if outliers > 0:
                        print(f"      ⚠️  Outliers (>5σ): {outliers}")
                        problematic_cols.append((col, 'Outliers'))
            except TypeError:
                print(f"      ⚠️  No se pueden calcular estadísticas (tipo: {df[col].dtype})")
    
    # 3. Identificar filas problemáticas
    print(f"\n🔍 Buscando filas problemáticas...")
    
    # Usar solo columnas numéricas disponibles
    numeric_available_cols = [col for col in available_feature_cols 
                             if pd.api.types.is_numeric_dtype(df[col])]
    
    if numeric_available_cols:
        # Filas con cualquier NaN
        rows_with_nan = df[numeric_available_cols].isnull().any(axis=1)
        if rows_with_nan.any():
            print(f"   Filas con NaN: {rows_with_nan.sum()}")
            print("   Índices:", df[rows_with_nan].index.tolist()[:10])
        
        # Filas con cualquier Inf
        try:
            rows_with_inf = np.isinf(df[numeric_available_cols].values).any(axis=1)
            if rows_with_inf.any():
                print(f"   Filas con Inf: {rows_with_inf.sum()}")
                print("   Índices:", df[rows_with_inf].index.tolist()[:10])
        except TypeError:
            print("   ⚠️  No se puede verificar infinitos en todas las columnas")
    
    # 4. Recomendaciones
    print(f"\n💡 RECOMENDACIONES:")
    if problematic_cols:
        print("   Se detectaron problemas en las siguientes columnas:")
        for col, issue in problematic_cols:
            print(f"   - {col}: {issue}")
        
        print("\n   Acciones sugeridas:")
        print("   1. Revisar calculate_advanced_indicators()")
        print("   2. Aplicar clean_financial_data()")
        print("   3. Considerar eliminar columnas problemáticas")
    else:
        print("   ✅ No se detectaron problemas graves en los datos")
    
    print("="*60 + "\n")
    
    return problematic_cols
# ================================
# 🏋️ ENTRENAMIENTO
# ================================
class ForexDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# ================================
# 🏋️ ENTRENAMIENTO - VERSIÓN CORREGIDA
# ================================
def train_model(model, train_loader, val_loader, device):
    """Entrena el modelo con early stopping"""
    print("\n" + "="*70)
    print("  🏋️ ENTRENANDO MODELO HÍBRIDO")
    print("="*70)
    
    criterion = CoherentDeltaLoss(
        mse_weight=1.0,
        constraint_weight=5.0,
        realism_weight=2.0
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # 🔥 CORRECCIÓN: Eliminar 'verbose' - no compatible con todas las versiones
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
        # ❌ ELIMINADO: verbose=True
    )
    
    early_stopping = EarlyStopping(patience=Config.PATIENCE)
    
    train_losses, val_losses = [], []
    best_model_state = None
    best_val_loss = float('inf')
    
    for epoch in tqdm(range(Config.EPOCHS), desc="Progreso"):
        # Entrenamiento
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss, loss_components = criterion(predictions, y_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validación
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch)
                loss, _ = criterion(predictions, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Guardar mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            torch.save(model.state_dict(), 'best_model_temp.pth')
            print(f"\n💾 Mejor modelo guardado (val loss: {val_loss:.6f})")
        
        # Actualizar scheduler
        scheduler.step(val_loss)
        
        # Mostrar LR actual cada 10 épocas
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\n📊 Epoch {epoch+1}/{Config.EPOCHS}")
            print(f"  LR actual: {current_lr:.6f}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss:   {val_loss:.6f}")
            print(f"  Best Val:   {best_val_loss:.6f}")
        
        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"\n🛑 Early stopping en epoch {epoch+1}")
            break
    
    # Cargar mejor modelo
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    print(f"\n✅ Entrenamiento completado")
    print(f"   Mejor val loss: {best_val_loss:.6f}")
    print(f"   Épocas totales: {len(train_losses)}")
    
    return train_losses, val_losses

# ================================
# 🏋️ ENTRENAMIENTO OPTIMIZADO
# ================================
def train_improved_model(model, train_loader, val_loader, device):
    print("\n" + "="*70)
    print("  🏋️ ENTRENAMIENTO OPTIMIZADO")
    print("="*70)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    
    for epoch in range(Config.EPOCHS):
        # Entrenamiento
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validación
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Scheduler
        scheduler.step(val_loss)
        
        # Log
        if (epoch + 1) % 5 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
            print(f"  LR: {current_lr:.6f}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss:   {val_loss:.6f}")
            print(f"  Best Val:   {best_val_loss:.6f}")
        
        # Guardar mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
    
    return train_losses, val_losses

# ================================
# 📈 EVALUACIÓN MEJORADA
# ================================
def evaluate_improved_model(model, test_loader, scaler_out, close_prices, device):
    print("\n" + "="*70)
    print("  📊 EVALUACIÓN MEJORADA")
    print("="*70)
    
    model.eval()
    predictions, targets = [], []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            predictions.extend(pred.cpu().numpy())
            targets.extend(y_batch.cpu().numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Desnormalizar
    predictions_abs = scaler_out.inverse_transform(predictions)
    targets_abs = scaler_out.inverse_transform(targets)
    
    # Calcular deltas
    predictions_delta = (predictions_abs - close_prices) / close_prices
    targets_delta = (targets_abs - close_prices) / close_prices
    
    # Métricas
    metrics = {}
    for i, name in enumerate(['delta_high', 'delta_low', 'delta_close']):
        mae = mean_absolute_error(targets_delta[:, i], predictions_delta[:, i])
        rmse = np.sqrt(mean_squared_error(targets_delta[:, i], predictions_delta[:, i]))
        r2 = r2_score(targets_delta[:, i], predictions_delta[:, i])
        
        # Accuracy direccional
        direction_true = np.sign(targets_delta[:, i])
        direction_pred = np.sign(predictions_delta[:, i])
        accuracy = np.mean(direction_true == direction_pred) * 100
        
        metrics[name] = {
            'MAE': float(mae),
            'RMSE': float(rmse),
            'R2': float(r2),
            'Direction_Accuracy': float(accuracy)
        }
        
        print(f"\n📊 {name}:")
        print(f"   MAE:  {mae:.6f}")
        print(f"   RMSE: {rmse:.6f}")
        print(f"   R²:   {r2:.4f}")
        print(f"   Accuracy Direccional: {accuracy:.2f}%")
    
    return predictions_abs, targets_abs, metrics


# ================================
# 📊 EVALUACIÓN
# ================================
def evaluate_model(model, test_loader, scaler_out, target_cols, device):
    """Evalúa el modelo en test set"""
    print("\n" + "="*70)
    print("  📊 EVALUACIÓN DEL MODELO")
    print("="*70)
    
    model.eval()
    predictions, targets = [], []
    
    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader, desc="Evaluando"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            predictions.extend(pred.cpu().numpy())
            targets.extend(y_batch.cpu().numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Desnormalizar
    predictions_denorm = scaler_out.inverse_transform(predictions)
    targets_denorm = scaler_out.inverse_transform(targets)
    
    # Métricas por target
    metrics = {}
    for i, col in enumerate(target_cols):
        mae = mean_absolute_error(targets_denorm[:, i], predictions_denorm[:, i])
        rmse = np.sqrt(mean_squared_error(targets_denorm[:, i], predictions_denorm[:, i]))
        r2 = r2_score(targets_denorm[:, i], predictions_denorm[:, i])
        
        # MAPE (evitar división por cero)
        y_true = targets_denorm[:, i]
        y_pred = predictions_denorm[:, i]
        non_zero_mask = np.abs(y_true) > 1e-8
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
        else:
            mape = np.nan
        
        metrics[col] = {
            'MAE': float(mae),
            'RMSE': float(rmse),
            'R2': float(r2),
            'MAPE': float(mape)
        }
        
        print(f"\n📊 {col}:")
        print(f"   MAE:  {mae:.6f}")
        print(f"   RMSE: {rmse:.6f}")
        print(f"   R²:   {r2:.4f}")
        print(f"   MAPE: {mape:.2f}%")
    
    return predictions_denorm, targets_denorm, metrics

# ================================
# 🎨 VISUALIZACIÓN
# ================================
def plot_training_history(train_losses, val_losses, metrics, predictions, targets):
    """Crea gráficas del entrenamiento y resultados"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('ADAUSD LSTM Híbrido - Entrenamiento y Resultados', fontsize=16, fontweight='bold')
    
    # Pérdidas
    ax1 = axes[0, 0]
    ax1.plot(train_losses, label='Train', linewidth=2)
    ax1.plot(val_losses, label='Val', linewidth=2)
    ax1.set_title('Pérdida durante Entrenamiento')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Pérdida')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # R² scores
    ax2 = axes[0, 1]
    targets_names = list(metrics.keys())
    r2_scores = [metrics[t]['R2'] for t in targets_names]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax2.bar(targets_names, r2_scores, color=colors)
    ax2.set_title('R² Score por Target')
    ax2.set_ylabel('R²')
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Añadir valores en las barras
    for bar, score in zip(bars, r2_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}', ha='center', va='bottom')
    
    # MAE
    ax3 = axes[0, 2]
    mae_scores = [metrics[t]['MAE'] for t in targets_names]
    bars = ax3.bar(targets_names, mae_scores, color=colors)
    ax3.set_title('MAE por Target')
    ax3.set_ylabel('MAE')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Predicciones vs Real (delta_high)
    ax4 = axes[1, 0]
    sample_size = min(200, len(predictions))
    ax4.scatter(targets[:sample_size, 0], predictions[:sample_size, 0], 
                alpha=0.6, s=20, color=colors[0])
    ax4.plot([targets[:, 0].min(), targets[:, 0].max()],
             [targets[:, 0].min(), targets[:, 0].max()], 'r--', alpha=0.7)
    ax4.set_title('delta_high: Predicciones vs Real')
    ax4.set_xlabel('Real')
    ax4.set_ylabel('Predicción')
    ax4.grid(True, alpha=0.3)
    
    # Predicciones vs Real (delta_low)
    ax5 = axes[1, 1]
    ax5.scatter(targets[:sample_size, 1], predictions[:sample_size, 1], 
                alpha=0.6, s=20, color=colors[1])
    ax5.plot([targets[:, 1].min(), targets[:, 1].max()],
             [targets[:, 1].min(), targets[:, 1].max()], 'r--', alpha=0.7)
    ax5.set_title('delta_low: Predicciones vs Real')
    ax5.set_xlabel('Real')
    ax5.set_ylabel('Predicción')
    ax5.grid(True, alpha=0.3)
    
    # Predicciones vs Real (delta_close)
    ax6 = axes[1, 2]
    ax6.scatter(targets[:sample_size, 2], predictions[:sample_size, 2], 
                alpha=0.6, s=20, color=colors[2])
    ax6.plot([targets[:, 2].min(), targets[:, 2].max()],
             [targets[:, 2].min(), targets[:, 2].max()], 'r--', alpha=0.7)
    ax6.set_title('delta_close: Predicciones vs Real')
    ax6.set_xlabel('Real')
    ax6.set_ylabel('Predicción')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('adausd_hybrid_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n📈 Gráficas guardadas en 'adausd_hybrid_results.png'")

# ================================
# 🚀 MAIN
# ================================
# ================================
# 🚀 MAIN MEJORADO
# ================================
def main():
    try:
        print("\n" + "="*70)
        print("  🚀 LSTM OPTIMIZADO PARA ADAUSD")
        print("="*70)
        
        # Configuración
        print("\n⚙️  CONFIGURACIÓN OPTIMIZADA:")
        print(f"   Sequence Length: {Config.SEQ_LEN}")
        print(f"   Hidden Size: {Config.HIDDEN_SIZE}")
        print(f"   Layers: {Config.NUM_LAYERS}")
        print(f"   Dropout: {Config.DROPOUT}")
        
        # Device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n🖥️  Device: {device}")
        
        # 1. Descargar datos
        df = download_and_prepare_data(symbol="ADA-USD", interval='1h')
        
        # 2. Preparar dataset mejorado
        (X_train_seq, y_train_seq), (X_val_seq, y_val_seq), (X_test_seq, y_test_seq), \
        scaler_in, scaler_out, feature_cols, target_cols = prepare_improved_dataset(df)
        
        input_size = len(feature_cols)
        output_size = len(target_cols)
        
        # 3. Dataloaders
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train_seq), 
            torch.FloatTensor(y_train_seq)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val_seq), 
            torch.FloatTensor(y_val_seq)
        )
        test_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_test_seq), 
            torch.FloatTensor(y_test_seq)
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False
        )
        
        # 4. Modelo
        model = PricePredictorLSTM(
            input_size=input_size,
            hidden_size=Config.HIDDEN_SIZE,
            num_layers=Config.NUM_LAYERS,
            output_size=output_size,
            dropout=Config.DROPOUT
        ).to(device)
        
        # 5. Entrenar
        start_time = time.time()
        train_losses, val_losses = train_improved_model(model, train_loader, val_loader, device)
        training_time = time.time() - start_time
        
        # 6. Evaluar
        predictions, targets, metrics = evaluate_improved_model(
            model, test_loader, scaler_out, close_prices, device
        )
        
        print(f"\n✅ Entrenamiento completado en {training_time/60:.1f} minutos")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
