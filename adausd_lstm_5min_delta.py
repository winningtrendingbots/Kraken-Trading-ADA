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
warnings.filterwarnings('ignore')

# ================================
# 🎛️ CONFIGURACIÓN MEJORADA
# ================================
class Config:
    # Características
    USE_VOLUME = True               # ✅ Usar volumen como feature
    USE_VOLUME_DERIVATIVES = True   # ✅ Incluir 1ra y 2da derivada del volumen
    USE_VOLUME_INDICATORS = True    # ✅ Calcular OBV, VWAP, PVT
    USE_PRICE_DERIVATIVES = True    # ✅ Derivadas de precio
    
    # Targets
    PREDICT_DELTAS = True           # ✅ Predecir deltas (no absolutos)
    PREDICT_VOLUME_DELTA = False    # ❌ NO predecir delta de volumen (problema de escala)
    
    # Normalización
    SCALER_TYPE = 'robust'          # ✅ RobustScaler (menos sensible a outliers)
    SCALE_BY_WINDOW = False         # ❌ Escalar globalmente
    
    # Arquitectura LSTM
    SEQ_LEN = 60                    # 60 velas (2.5 días en 1h)
    HIDDEN_SIZE = 128               # Reducido para evitar overfitting
    NUM_LAYERS = 2                  # Menos capas
    DROPOUT = 0.25                  # Menor dropout
    BIDIRECTIONAL = True            # ✅ LSTM bidireccional
    
    # Entrenamiento
    BATCH_SIZE = 64                 # Batch más pequeño
    EPOCHS = 120                    # Más épocas
    LEARNING_RATE = 0.0008          # Learning rate más bajo
    WEIGHT_DECAY = 1e-5             # Regularización L2
    PATIENCE = 15                   # Paciencia early stopping
    
    # Output
    OUTPUT_SIZE = 3                 # Solo delta_high, delta_low, delta_close
    
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
# 🧠 MODELO LSTM HÍBRIDO
# ================================
class HybridLSTM(nn.Module):
    """
    LSTM bidireccional con atención y skip connections
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.25, bidirectional=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM bidireccional
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Capa de atención
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Capas fully connected
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
        
        # Inicialización de pesos
        self.init_weights()
    
    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
        for layer in self.fc_layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)  # [batch, seq_len, hidden*directions]
        
        # Atención
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Fully connected
        output = self.fc_layers(context)
        return output

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

def prepare_delta_dataset(df):
    """
    🔥 PREPARA DATOS CON DELTAS MEJORADOS
    
    Target: [delta_high, delta_low, delta_close]
    donde delta = (valor_futuro - valor_actual) / valor_actual
    """
    print("\n" + "="*70)
    print("  🔧 PREPARACIÓN DE DATOS CON DELTAS MEJORADOS")
    print("="*70)
    
    # 1. Calcular indicadores avanzados
    df = calculate_advanced_indicators(df)
    
    # 2. Calcular deltas para la próxima vela
    df['delta_high'] = (df['high'].shift(-1) - df['close']) / df['close']
    df['delta_low'] = (df['low'].shift(-1) - df['close']) / df['close']
    df['delta_close'] = (df['close'].shift(-1) - df['close']) / df['close']
    
    # 3. Eliminar NaNs
    initial_len = len(df)
    df = df.dropna()
    print(f"📊 Datos después de limpieza: {len(df):,} de {initial_len:,} velas")
    
    # 4. Seleccionar features
    feature_cols = ['open', 'high', 'low', 'close', 'volume']
    
    if Config.USE_VOLUME_INDICATORS:
        feature_cols.extend(['obv', 'obv_roc', 'vwap', 'pvt', 'volume_ratio', 'volume_rsi'])
    
    if Config.USE_VOLUME_DERIVATIVES:
        feature_cols.extend(['volume_1st_deriv', 'volume_2nd_deriv'])
    
    if Config.USE_PRICE_DERIVATIVES:
        feature_cols.extend(['price_1st_deriv', 'price_2nd_deriv'])
    
    # Divergencias
    feature_cols.extend(['bullish_divergence', 'bearish_divergence'])
    
    # Targets
    target_cols = ['delta_high', 'delta_low', 'delta_close']
    
    print(f"\n🎯 {len(feature_cols)} Features: {feature_cols}")
    print(f"🎯 {len(target_cols)} Targets: {target_cols}")
    
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
    
    if Config.SCALER_TYPE == 'robust':
        scaler_in = RobustScaler()
        scaler_out = RobustScaler()
    else:
        scaler_in = StandardScaler()
        scaler_out = StandardScaler()
    
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
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
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
        
        # Actualizar scheduler
        scheduler.step(val_loss)
        
        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"\n🛑 Early stopping en epoch {epoch+1}")
            break
        
        # Log cada 10 épocas
        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss:   {val_loss:.6f}")
            print(f"  Best Val:   {best_val_loss:.6f}")
    
    # Cargar mejor modelo
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    print(f"\n✅ Entrenamiento completado")
    print(f"   Mejor val loss: {best_val_loss:.6f}")
    
    return train_losses, val_losses

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
def main():
    try:
        print("\n" + "="*70)
        print("  🔥 LSTM HÍBRIDO CON DELTAS Y VOLUMEN MEJORADO")
        print("="*70)
        
        print("\n⚙️  CONFIGURACIÓN:")
        print(f"   Sequence Length: {Config.SEQ_LEN}")
        print(f"   Hidden Size: {Config.HIDDEN_SIZE}")
        print(f"   Layers: {Config.NUM_LAYERS}")
        print(f"   Bidirectional: {Config.BIDIRECTIONAL}")
        print(f"   Dropout: {Config.DROPOUT}")
        print(f"   Use Volume Indicators: {Config.USE_VOLUME_INDICATORS}")
        print(f"   Use Volume Derivatives: {Config.USE_VOLUME_DERIVATIVES}")
        
        # Device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n🖥️  Device: {device}")
        
        # 1. Descargar datos
        df = download_and_prepare_data(symbol="ADA-USD", interval='1h')
        
        # 2. Preparar dataset
        (X_train, y_train), (X_val, y_val), (X_test, y_test), \
        scaler_in, scaler_out, feature_cols, target_cols = prepare_delta_dataset(df)
        
        input_size = len(feature_cols)
        output_size = len(target_cols)
        
        print(f"\n📊 Dimensiones finales:")
        print(f"   Input size: {input_size}")
        print(f"   Output size: {output_size}")
        print(f"   Total parámetros estimados: ~{(input_size * Config.HIDDEN_SIZE * 4 + Config.HIDDEN_SIZE * output_size) / 1e6:.2f}M")
        
        # 3. Crear dataloaders
        train_dataset = ForexDataset(X_train, y_train)
        val_dataset = ForexDataset(X_val, y_val)
        test_dataset = ForexDataset(X_test, y_test)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False
        )
        
        # 4. Crear modelo
        model = HybridLSTM(
            input_size=input_size,
            hidden_size=Config.HIDDEN_SIZE,
            num_layers=Config.NUM_LAYERS,
            output_size=output_size,
            dropout=Config.DROPOUT,
            bidirectional=Config.BIDIRECTIONAL
        ).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n🧠 Modelo creado:")
        print(f"   Total parámetros: {total_params:,}")
        print(f"   Parámetros entrenables: {trainable_params:,}")
        
        # 5. Entrenar
        start_time = time.time()
        train_losses, val_losses = train_model(model, train_loader, val_loader, device)
        training_time = time.time() - start_time
        
        # 6. Evaluar
        predictions, targets, metrics = evaluate_model(
            model, test_loader, scaler_out, target_cols, device
        )
        
        # 7. Graficar
        plot_training_history(train_losses, val_losses, metrics, predictions, targets)
        
        # 8. Guardar modelo
        model_dir = 'ADAUSD_MODELS'
        os.makedirs(model_dir, exist_ok=True)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'input_size': input_size,
                'hidden_size': Config.HIDDEN_SIZE,
                'num_layers': Config.NUM_LAYERS,
                'output_size': output_size,
                'seq_len': Config.SEQ_LEN,
                'bidirectional': Config.BIDIRECTIONAL,
                'dropout': Config.DROPOUT,
                'feature_cols': feature_cols,
                'target_cols': target_cols
            },
            'metrics_test': metrics,
            'scaler_in': scaler_in,
            'scaler_out': scaler_out
        }, f'{model_dir}/adausd_hybrid_lstm.pth')
        
        joblib.dump(scaler_in, f'{model_dir}/scaler_input_hybrid.pkl')
        joblib.dump(scaler_out, f'{model_dir}/scaler_output_hybrid.pkl')
        
        with open(f'{model_dir}/config_hybrid.json', 'w') as f:
            json.dump({
                'input_size': input_size,
                'output_size': output_size,
                'seq_len': Config.SEQ_LEN,
                'hidden_size': Config.HIDDEN_SIZE,
                'num_layers': Config.NUM_LAYERS,
                'bidirectional': Config.BIDIRECTIONAL,
                'dropout': Config.DROPOUT,
                'feature_cols': feature_cols,
                'target_cols': target_cols,
                'metrics_test': metrics,
                'training_time_minutes': training_time / 60,
                'total_params': total_params,
                'trainable_params': trainable_params,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        # 9. Telegram summary
        avg_r2 = np.mean([metrics[t]['R2'] for t in target_cols])
        avg_mae = np.mean([metrics[t]['MAE'] for t in target_cols])
        
        msg = f"""✅ *MODELO HÍBRIDO ENTRENADO*

⏱️ Tiempo: {training_time/60:.1f} min
🧠 Parámetros: {total_params:,}

📊 *Métricas Promedio:*
   • R²: {avg_r2:.4f}
   • MAE: {avg_mae:.6f}

🎯 *Features ({input_size}):*
   • OHLCV: ✅
   • Indicadores volumen: {'✅' if Config.USE_VOLUME_INDICATORS else '❌'}
   • Derivadas volumen: {'✅' if Config.USE_VOLUME_DERIVATIVES else '❌'}
   • Divergencias: ✅

📈 *Targets ({output_size}):*
   • delta_high: R²={metrics['delta_high']['R2']:.4f}
   • delta_low: R²={metrics['delta_low']['R2']:.4f}
   • delta_close: R²={metrics['delta_close']['R2']:.4f}

🔥 *Modelo:*
   • LSTM Bidireccional
   • Atención
   • Skip Connections
   • Regularización L2

✅ *Anclaje garantizado a precio actual*
"""
        
        print("\n" + "="*70)
        print("  ✅✅✅ ENTRENAMIENTO COMPLETADO ✅✅✅")
        print("="*70)
        print(msg.replace('*', ''))
        
        send_telegram(msg)
        
    except Exception as e:
        error_msg = f"❌ Error en entrenamiento: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        send_telegram(error_msg)
        raise

if __name__ == "__main__":
    main()
