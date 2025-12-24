"""
ENTRENAMIENTO LSTM CON DELTAS Y VOLUMEN
✅ Predice CAMBIOS RELATIVOS (no valores absolutos)
✅ Incluye volumen como feature crítico
✅ Sistema configurable con flags
✅ Basado en mejores prácticas de MQL5 + Volume Analysis
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
import json
import joblib
import yfinance as yf
from tqdm.auto import tqdm
import requests
from datetime import datetime

# ================================
# 🎛️ CONFIGURACIÓN
# ================================
class Config:
    # Features
    USE_VOLUME = True              # ✅ Usar volumen como feature
    USE_DELTA_PREDICTION = True    # ✅ Predecir deltas en lugar de absolutos
    USE_PERCENTAGE_TARGETS = False # ❌ Si True, predice % change en lugar de deltas
    
    # Normalización
    NORMALIZE_BY_WINDOW = True     # ✅ Normalizar por ventana local (no global)
    WINDOW_NORM_DAYS = 120         # Ventana para normalización si no es local
    
    # Volumen
    PREDICT_VOLUME = True          # ✅ Predecir volumen futuro
    VOLUME_INDICATORS = True       # ✅ Calcular OBV, VWAP, etc.
    
    # Arquitectura
    SEQ_LEN = 72                   # 3 días de velas 1h
    HIDDEN_SIZE = 160
    NUM_LAYERS = 3
    DROPOUT = 0.35
    
    # Entrenamiento
    BATCH_SIZE = 128
    EPOCHS = 100
    LEARNING_RATE = 0.0012
    PATIENCE = 8
    
    # Output
    OUTPUT_SIZE = 4 if PREDICT_VOLUME else 3  # High, Low, Close, (Volume)

# Telegram
TELEGRAM_API = os.environ.get('TELEGRAM_API', '')
CHAT_ID = os.environ.get('CHAT_ID', '')

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
    """
    Calcula indicadores avanzados de volumen
    Basado en el artículo de MQL5
    """
    df = df.copy()
    
    # 1. On-Balance Volume (OBV)
    obv = [0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.append(obv[-1] + df['volume'].iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.append(obv[-1] - df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['obv'] = obv
    
    # 2. Volume Rate of Change
    df['volume_roc'] = df['volume'].pct_change(periods=14)
    
    # 3. Volume Moving Average
    df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
    
    # 4. Volume Ratio (actual vs average)
    df['volume_ratio'] = df['volume'] / df['volume_ma_20']
    
    # 5. VWAP (Volume Weighted Average Price)
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    
    # 6. Price-Volume Trend
    df['pvt'] = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1) * df['volume']).cumsum()
    
    # Rellenar NaNs
    df.fillna(method='bfill', inplace=True)
    
    return df

def detect_volume_divergence(df, window=14):
    """
    Detecta divergencias entre precio y volumen
    """
    df = df.copy()
    
    # Tendencia de precio (MA slope)
    price_slope = df['close'].rolling(window=window).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0]
    )
    
    # Tendencia de volumen (MA slope)
    volume_slope = df['volume'].rolling(window=window).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0]
    )
    
    # Divergencia: precio sube pero volumen baja (bearish)
    df['bearish_divergence'] = ((price_slope > 0) & (volume_slope < 0)).astype(int)
    
    # Divergencia: precio baja pero volumen baja (bullish)
    df['bullish_divergence'] = ((price_slope < 0) & (volume_slope < 0)).astype(int)
    
    df.fillna(0, inplace=True)
    
    return df

# ================================
# 🧠 MODELO LSTM MEJORADO
# ================================
class ImprovedLSTM(nn.Module):
    """
    LSTM mejorado que:
    - Acepta OHLCV (+ indicadores de volumen opcionales)
    - Predice deltas relativos al precio actual
    - Puede predecir volumen futuro
    """
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

class DeltaConstrainedLoss(nn.Module):
    """
    Loss mejorada que:
    1. Penaliza si delta_high < delta_low
    2. Penaliza predicciones irrealistas
    """
    def __init__(self, constraint_weight=10.0, realism_weight=5.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.constraint_weight = constraint_weight
        self.realism_weight = realism_weight
    
    def forward(self, predictions, targets):
        # MSE base
        mse_loss = self.mse(predictions, targets)
        
        # Constraint: delta_high debe ser >= delta_low
        pred_delta_high = predictions[:, 0]
        pred_delta_low = predictions[:, 1]
        
        violation = torch.clamp(pred_delta_low - pred_delta_high, min=0)
        constraint_loss = violation.mean()
        
        # Realism: los deltas no deben ser extremos (>10%)
        max_reasonable_delta = 0.10  # 10%
        extreme_high = torch.clamp(torch.abs(pred_delta_high) - max_reasonable_delta, min=0)
        extreme_low = torch.clamp(torch.abs(pred_delta_low) - max_reasonable_delta, min=0)
        realism_loss = (extreme_high + extreme_low).mean()
        
        total_loss = mse_loss + \
                    self.constraint_weight * constraint_loss + \
                    self.realism_weight * realism_loss
        
        return total_loss

# ================================
# 📦 PREPARACIÓN DE DATOS
# ================================
def download_data_with_volume(symbol="ADA-USD", interval='1h', path='ADAUSD_1h_data.csv'):
    """Descarga datos incluyendo volumen"""
    print("="*70)
    print(f"  📥 DESCARGA DE DATOS CON VOLUMEN - {interval.upper()}")
    print("="*70 + "\n")
    
    if os.path.exists(path):
        df_exist = pd.read_csv(path)
        df_exist['time'] = pd.to_datetime(df_exist['time'])
        
        if df_exist['time'].dt.tz is not None:
            now = pd.Timestamp.now(tz='UTC')
        else:
            now = pd.Timestamp.now()
        
        diff_h = (now - df_exist['time'].max()).total_seconds() / 3600
        
        if diff_h < 1:
            print(f"✅ Datos actualizados (hace {diff_h:.1f}h)\n")
            return df_exist
    
    ticker = yf.Ticker(symbol)
    df_new = ticker.history(period="max", interval=interval)
    df_new = df_new.reset_index()
    df_new.columns = [str(c).lower() for c in df_new.columns]
    df_new.rename(columns={'date': 'time', 'datetime': 'time'}, inplace=True)
    
    if 'time' in df_new.columns:
        df_new['time'] = pd.to_datetime(df_new['time']).dt.tz_localize(None)
    
    # ✅ INCLUIR VOLUMEN
    df_new = df_new[['time', 'open', 'high', 'low', 'close', 'volume']]
    
    if 'df_exist' in locals():
        df_exist['time'] = df_exist['time'].dt.tz_localize(None) if df_exist['time'].dt.tz is not None else df_exist['time']
        df = pd.concat([df_exist, df_new], ignore_index=True)
        df.sort_values('time', inplace=True)
        df.drop_duplicates('time', keep='last', inplace=True)
        df.reset_index(drop=True, inplace=True)
    else:
        df = df_new
    
    df.to_csv(path, index=False)
    print(f"\n✅ Guardado: {len(df):,} velas con volumen")
    return df

def prepare_delta_data(df, seq_len=72, train_size=0.75, val_size=0.15):
    """
    🔥 PREPARACIÓN CON DELTAS
    
    En lugar de predecir [High, Low, Close] absolutos,
    predice [Delta_High, Delta_Low, Delta_Close] relativos al Close actual.
    
    Delta_High = (High_next - Close_current) / Close_current
    """
    print(f"\n{'='*70}")
    print("  🔧 PREPARACIÓN DE DATOS CON DELTAS")
    print("="*70)
    
    # 1. Añadir indicadores de volumen
    if Config.VOLUME_INDICATORS:
        print("\n📊 Calculando indicadores de volumen...")
        df = calculate_volume_indicators(df)
        df = detect_volume_divergence(df)
    
    # 2. Crear deltas de precio
    print("\n🔄 Calculando deltas relativos...")
    
    df['delta_high'] = (df['high'].shift(-1) - df['close']) / df['close']
    df['delta_low'] = (df['low'].shift(-1) - df['close']) / df['close']
    df['delta_close'] = (df['close'].shift(-1) - df['close']) / df['close']
    
    if Config.PREDICT_VOLUME:
        df['delta_volume'] = (df['volume'].shift(-1) - df['volume']) / (df['volume'] + 1e-8)
    
    # Eliminar NaNs
    df = df.dropna().reset_index(drop=True)
    
    # 3. Dividir temporalmente
    total = len(df)
    train_end = int(total * train_size)
    val_end = int(total * (train_size + val_size))
    
    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()
    
    print(f"\n📊 División temporal:")
    print(f"   Train: {len(df_train):,} velas")
    print(f"   Val:   {len(df_val):,} velas")
    print(f"   Test:  {len(df_test):,} velas")
    
    # 4. Seleccionar features
    if Config.USE_VOLUME and Config.VOLUME_INDICATORS:
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 
                       'obv', 'volume_ratio', 'vwap', 'pvt',
                       'bearish_divergence', 'bullish_divergence']
    elif Config.USE_VOLUME:
        feature_cols = ['open', 'high', 'low', 'close', 'volume']
    else:
        feature_cols = ['open', 'high', 'low', 'close']
    
    # Targets (deltas)
    if Config.PREDICT_VOLUME:
        target_cols = ['delta_high', 'delta_low', 'delta_close', 'delta_volume']
    else:
        target_cols = ['delta_high', 'delta_low', 'delta_close']
    
    print(f"\n🎯 Features: {feature_cols}")
    print(f"🎯 Targets: {target_cols}")
    
    # 5. Normalización
    if Config.NORMALIZE_BY_WINDOW:
        print("\n🔧 Normalización por ventana local...")
        scaler_in = StandardScaler()  # Z-score mejor para ventanas
        scaler_out = StandardScaler()
    else:
        print("\n🔧 Normalización global (MinMax)...")
        scaler_in = MinMaxScaler()
        scaler_out = MinMaxScaler()
    
    inp_train = df_train[feature_cols].values
    out_train = df_train[target_cols].values
    
    inp_val = df_val[feature_cols].values
    out_val = df_val[target_cols].values
    
    inp_test = df_test[feature_cols].values
    out_test = df_test[target_cols].values
    
    # Fit solo en train
    inp_train_scaled = scaler_in.fit_transform(inp_train)
    out_train_scaled = scaler_out.fit_transform(out_train)
    
    inp_val_scaled = scaler_in.transform(inp_val)
    out_val_scaled = scaler_out.transform(out_val)
    
    inp_test_scaled = scaler_in.transform(inp_test)
    out_test_scaled = scaler_out.transform(out_test)
    
    # 6. Crear secuencias
    def create_sequences(inp_scaled, out_scaled, seq_len):
        X, y = [], []
        for i in range(seq_len, len(inp_scaled)):
            X.append(inp_scaled[i-seq_len:i])
            y.append(out_scaled[i, :])
        return np.array(X), np.array(y)
    
    print("\n🔄 Creando secuencias...")
    X_train, y_train = create_sequences(inp_train_scaled, out_train_scaled, seq_len)
    X_val, y_val = create_sequences(inp_val_scaled, out_val_scaled, seq_len)
    X_test, y_test = create_sequences(inp_test_scaled, out_test_scaled, seq_len)
    
    print(f"✅ Train: X={X_train.shape}, y={y_train.shape}")
    print(f"✅ Val:   X={X_val.shape}, y={y_val.shape}")
    print(f"✅ Test:  X={X_test.shape}, y={y_test.shape}\n")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler_in, scaler_out, feature_cols, target_cols

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
    def __init__(self, patience=15):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def train_model(model, train_loader, val_loader, epochs, lr, device, patience):
    print(f"\n{'='*70}")
    print("  🏋️ ENTRENANDO MODELO")
    print("="*70)
    
    criterion = DeltaConstrainedLoss(constraint_weight=10.0, realism_weight=5.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    early_stop = EarlyStopping(patience)
    
    train_losses, val_losses, lrs = [], [], []
    best_state = None
    best_val = float('inf')
    
    model.to(device)
    for epoch in tqdm(range(epochs), desc="Progreso"):
        lrs.append(optimizer.param_groups[0]['lr'])
        
        # Train
        model.train()
        t_loss = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(X_b)
            loss = criterion(pred, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()
        t_loss /= len(train_loader)
        train_losses.append(t_loss)
        
        # Val
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                pred = model(X_b)
                v_loss += criterion(pred, y_b).item()
        v_loss /= len(val_loader)
        val_losses.append(v_loss)
        
        if v_loss < best_val:
            best_val = v_loss
            best_state = model.state_dict().copy()
        
        scheduler.step(v_loss)
        early_stop(v_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch+1}: Train={t_loss:.6f} | Val={v_loss:.6f}")
        
        if early_stop.early_stop:
            print(f"\n🛑 Early stop en epoch {epoch+1}")
            break
    
    if best_state:
        model.load_state_dict(best_state)
    
    print(f"\n✅ Entrenamiento completado (Best Val: {best_val:.6f})\n")
    return train_losses, val_losses, lrs

# ================================
# 📊 EVALUACIÓN
# ================================
def evaluate(model, test_loader, scaler_out, device, target_cols):
    print(f"\n{'='*70}")
    print("  📊 EVALUANDO MODELO")
    print("="*70 + "\n")
    
    model.eval()
    preds, acts = [], []
    with torch.no_grad():
        for X_b, y_b in tqdm(test_loader, desc="Test"):
            preds.extend(model(X_b.to(device)).cpu().numpy())
            acts.extend(y_b.numpy())
    
    preds, acts = np.array(preds), np.array(acts)
    pred_denorm = scaler_out.inverse_transform(preds)
    act_denorm = scaler_out.inverse_transform(acts)
    
    # Métricas
    metrics = {}
    for i, label in enumerate(target_cols):
        mae = mean_absolute_error(act_denorm[:, i], pred_denorm[:, i])
        rmse = np.sqrt(mean_squared_error(act_denorm[:, i], pred_denorm[:, i]))
        r2 = r2_score(act_denorm[:, i], pred_denorm[:, i])
        mape = np.mean(np.abs((act_denorm[:, i] - pred_denorm[:, i]) / (np.abs(act_denorm[:, i]) + 1e-8))) * 100
        
        metrics[label] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}
        
        print(f"   {label}:")
        print(f"      MAE: {mae:.6f} | RMSE: {rmse:.6f}")
        print(f"      R²: {r2:.4f} | MAPE: {mape:.2f}%\n")
    
    return preds, acts, metrics, pred_denorm, act_denorm

# ================================
# 🎨 VISUALIZACIÓN
# ================================
def plot_results(train_l, val_l, lrs, pred_d, act_d, metrics, target_cols, path):
    """Gráficas de resultados"""
    def smooth(data, w=0.85):
        s, last = [], data[0]
        for p in data:
            val = last * w + (1 - w) * p
            s.append(val)
            last = val
        return s
    
    n_targets = len(target_cols)
    fig = plt.figure(figsize=(24, 4 * n_targets + 4))
    fig.suptitle('ADAUSD LSTM - Delta Prediction with Volume', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFA07A']
    
    # Training History
    ax1 = plt.subplot(n_targets + 1, 3, 1)
    ax1.plot(smooth(train_l), 'b-', linewidth=2, label='Train', alpha=0.8)
    ax1.plot(smooth(val_l), 'r-', linewidth=2, label='Val', alpha=0.8)
    ax1.set_title('Training History', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # LR
    ax2 = plt.subplot(n_targets + 1, 3, 2)
    ax2.plot(lrs, color='purple', linewidth=2)
    ax2.set_yscale('log')
    ax2.set_title('Learning Rate', fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # Loss Log
    ax3 = plt.subplot(n_targets + 1, 3, 3)
    ax3.plot(smooth(train_l), 'b-', linewidth=2, label='Train', alpha=0.8)
    ax3.plot(smooth(val_l), 'r-', linewidth=2, label='Val', alpha=0.8)
    ax3.set_yscale('log')
    ax3.set_title('Loss (Log)', fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    n = min(500, len(pred_d))
    
    # Predicciones por target
    for i, (lbl, col) in enumerate(zip(target_cols, colors)):
        # Time series
        ax = plt.subplot(n_targets + 1, 3, 4 + i * 3)
        ax.plot(act_d[:n, i], 'k-', linewidth=1.5, alpha=0.7, label='Real')
        ax.plot(pred_d[:n, i], color=col, linewidth=1.5, label='Pred', alpha=0.8)
        ax.fill_between(range(n), act_d[:n, i], pred_d[:n, i], alpha=0.2, color=col)
        ax.set_title(f'{lbl} - Predictions', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        m = metrics[lbl]
        metrics_text = f"MAE: {m['MAE']:.6f}\nR²: {m['R2']:.4f}"
        ax.text(0.98, 0.02, metrics_text, transform=ax.transAxes, 
                fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor=col, alpha=0.3))
        
        # Scatter
        ax = plt.subplot(n_targets + 1, 3, 5 + i * 3)
        ax.scatter(act_d[:, i], pred_d[:, i], alpha=0.5, s=10, c=col, edgecolors='none')
        mn, mx = act_d[:, i].min(), act_d[:, i].max()
        ax.plot([mn, mx], [mn, mx], 'r--', linewidth=2, alpha=0.7)
        ax.set_title(f'{lbl} - Scatter', fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Error dist
        ax = plt.subplot(n_targets + 1, 3, 6 + i * 3)
        err = pred_d[:, i] - act_d[:, i]
        ax.hist(err, bins=50, alpha=0.7, color=col, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2)
        ax.set_title(f'{lbl} - Error Distribution', fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"📈 Gráficas: {path}\n")

# ================================
# 🚀 MAIN
# ================================
if __name__ == "__main__":
    try:
        print("\n" + "="*70)
        print("  🔥 ENTRENAMIENTO LSTM CON DELTAS Y VOLUMEN")
        print("="*70 + "\n")
        
        print("⚙️  CONFIGURACIÓN:")
        print(f"   Use Volume: {Config.USE_VOLUME}")
        print(f"   Predict Deltas: {Config.USE_DELTA_PREDICTION}")
        print(f"   Volume Indicators: {Config.VOLUME_INDICATORS}")
        print(f"   Predict Volume: {Config.PREDICT_VOLUME}")
        print(f"   Normalize by Window: {Config.NORMALIZE_BY_WINDOW}\n")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Device: {device}\n")
        
        # 1. Descargar datos
        df = download_data_with_volume(symbol="ADA-USD", interval='1h', path='ADAUSD_1h_data.csv')
        
        # 2. Preparar datos
        (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler_in, scaler_out, feature_cols, target_cols = \
            prepare_delta_data(df, seq_len=Config.SEQ_LEN)
        
        input_size = len(feature_cols)
        output_size = len(target_cols)
        
        print(f"📊 Dimensiones:")
        print(f"   Input size: {input_size}")
        print(f"   Output size: {output_size}")
        
        # 3. Loaders
        train_loader = torch.utils.data.DataLoader(
            ForexDataset(X_train, y_train), Config.BATCH_SIZE, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            ForexDataset(X_val, y_val), Config.BATCH_SIZE, shuffle=False
        )
        test_loader = torch.utils.data.DataLoader(
            ForexDataset(X_test, y_test), Config.BATCH_SIZE, shuffle=False
        )
        
        # 4. Modelo
        model = ImprovedLSTM(
            input_size=input_size,
            hidden_size=Config.HIDDEN_SIZE,
            num_layers=Config.NUM_LAYERS,
            output_size=output_size,
            dropout=Config.DROPOUT
        )
        
        params = sum(p.numel() for p in model.parameters())
        print(f"\n🧠 Modelo: {params:,} parámetros\n")
        
        # 5. Entrenar
        start = time.time()
        train_l, val_l, lrs = train_model(
            model, train_loader, val_loader, 
            Config.EPOCHS, Config.LEARNING_RATE, device, Config.PATIENCE
        )
        
        # 6. Evaluar
        preds, acts, metrics_test, pred_d, act_d = evaluate(
            model, test_loader, scaler_out, device, target_cols
        )
        
        # 7. Graficar
        plot_results(train_l, val_l, lrs, pred_d, act_d, metrics_test, target_cols, 
                    'adausd_delta_results.png')
        
        # 8. Guardar
        model_dir = 'ADAUSD_MODELS'
        os.makedirs(model_dir, exist_ok=True)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'metrics_test': metrics_test,
            'config': {
                'seq_len': Config.SEQ_LEN,
                'hidden': Config.HIDDEN_SIZE,
                'layers': Config.NUM_LAYERS,
                'input_size': input_size,
                'output_size': output_size,
                'use_volume': Config.USE_VOLUME,
                'use_delta': Config.USE_DELTA_PREDICTION,
                'volume_indicators': Config.VOLUME_INDICATORS,
                'predict_volume': Config.PREDICT_VOLUME,
                'feature_cols': feature_cols,
                'target_cols': target_cols
            }
        }, f'{model_dir}/adausd_lstm_delta.pth')
        
        joblib.dump(scaler_in, f'{model_dir}/scaler_input_delta.pkl')
        joblib.dump(scaler_out, f'{model_dir}/scaler_output_delta.pkl')
        
        with open(f'{model_dir}/config_delta.json', 'w') as f:
            json.dump({
                'seq_len': Config.SEQ_LEN,
                'hidden': Config.HIDDEN_SIZE,
                'layers': Config.NUM_LAYERS,
                'input_size': input_size,
                'output_size': output_size,
                'use_volume': Config.USE_VOLUME,
                'use_delta': Config.USE_DELTA_PREDICTION,
                'volume_indicators': Config.VOLUME_INDICATORS,
                'predict_volume': Config.PREDICT_VOLUME,
                'feature_cols': feature_cols,
                'target_cols': target_cols,
                'metrics_test': {k: {mk: float(mv) for mk, mv in v.items()}
                                for k, v in metrics_test.items()}
            }, f, indent=2)
        
        total_time = time.time() - start
        
        # Mensaje Telegram
        msg = f"""✅ *Modelo Delta+Volume Entrenado*

⏱️ Tiempo: {total_time/60:.1f} min
🧠 Parámetros: {params:,}

🎯 *Features:* {input_size}
   • Volume: {'✅' if Config.USE_VOLUME else '❌'}
   • Indicators: {'✅' if Config.VOLUME_INDICATORS else '❌'}

📊 *Outputs:* {output_size}
   • Deltas: ✅
   • Volume Pred: {'✅' if Config.PREDICT_VOLUME else '❌'}

📈 *Métricas Test:*
"""
        
        for target, m in metrics_test.items():
            msg += f"\n*{target}:*\n"
            msg += f"  • R²: {m['R2']:.4f}\n"
            msg += f"  • MAE: {m['MAE']:.6f}\n"
        
        print("\n" + "="*70)
        print("✅✅✅  ENTRENAMIENTO COMPLETADO  ✅✅✅")
        print("="*70)
        print(msg.replace('*', ''))
        
        send_telegram(msg)
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        print(error_msg)
        send_telegram(error_msg)
        raise
