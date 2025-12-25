"""
LSTM HÍBRIDO MEJORADO - PREDICCIÓN OHLCV
✅ Volumen con normalización logarítmica separada
✅ Arquitectura más robusta (más capacidad)
✅ Loss balanceado entre precio y volumen
✅ Feature engineering mejorado
✅ Hiperparámetros optimizados
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
import time
import json
import joblib
import yfinance as yf
from tqdm.auto import tqdm
import requests
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ================================
# 🎛️ CONFIGURACIÓN MEJORADA
# ================================
class Config:
    # Características
    USE_VOLUME = True
    USE_VOLUME_LOG = True  # ✅ Nuevo: log-transform para volumen
    USE_VOLUME_DERIVATIVES = True
    USE_VOLUME_INDICATORS = True
    USE_PRICE_DERIVATIVES = True
    PREDICT_DELTAS = True
    
    # Arquitectura más robusta
    SEQ_LEN = 90  # ✅ Más contexto
    HIDDEN_SIZE = 128  # ✅ Más capacidad
    NUM_LAYERS = 3  # ✅ Más profundidad
    DROPOUT = 0.3  # ✅ Menos dropout (modelo más grande puede manejar)
    BIDIRECTIONAL = True
    
    # Entrenamiento ajustado
    BATCH_SIZE = 64  # ✅ Menor batch para mejor convergencia
    EPOCHS = 200
    LEARNING_RATE = 0.0001  # ✅ LR más bajo para estabilidad
    WEIGHT_DECAY = 5e-5
    PATIENCE = 25
    MIN_DELTA = 1e-6
    GRAD_CLIP = 1.0
    
    # Loss weights
    PRICE_WEIGHT = 1.0
    VOLUME_WEIGHT = 0.1  # ✅ Reducir peso del volumen
    
    # Datos
    TRAIN_SIZE = 0.70
    VAL_SIZE = 0.15
    TEST_SIZE = 0.15

# ================================
# 📊 INDICADORES MEJORADOS
# ================================
def calculate_enhanced_indicators(df):
    """Calcula indicadores técnicos avanzados"""
    df = df.copy()
    
    # === INDICADORES DE VOLUMEN ===
    
    # Log volumen (normaliza mejor)
    df['log_volume'] = np.log1p(df['volume'])
    df['volume_ma_5'] = df['volume'].rolling(5).mean()
    df['volume_ma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma_20'] + 1e-10)
    
    # Derivadas de volumen
    df['volume_1st_deriv'] = df['log_volume'].diff()
    df['volume_2nd_deriv'] = df['volume_1st_deriv'].diff()
    df['volume_acceleration'] = df['volume_1st_deriv'].diff()
    
    # OBV mejorado
    obv = [0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.append(obv[-1] + df['volume'].iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.append(obv[-1] - df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['obv'] = obv
    df['obv_ma'] = pd.Series(obv).rolling(20).mean()
    df['obv_signal'] = (df['obv'] > df['obv_ma']).astype(int)
    
    # === INDICADORES DE PRECIO ===
    
    # Medias móviles
    for period in [5, 10, 20, 50]:
        df[f'sma_{period}'] = df['close'].rolling(period).mean()
        df[f'close_sma_{period}_ratio'] = df['close'] / (df[f'sma_{period}'] + 1e-10)
    
    # EMAs
    for period in [9, 21]:
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    
    # VWAP
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (df['typical_price'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['vwap_distance'] = (df['close'] - df['vwap']) / (df['vwap'] + 1e-10)
    
    # Bandas de Bollinger
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    
    # RSI
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['rsi_ma'] = df['rsi'].rolling(5).mean()
    
    # MACD
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # ATR (Average True Range)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift())
    df['tr3'] = abs(df['low'] - df['close'].shift())
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['true_range'].rolling(14).mean()
    df['atr_percent'] = df['atr'] / (df['close'] + 1e-10)
    
    # Volatilidad
    df['returns'] = df['close'].pct_change()
    df['volatility_5'] = df['returns'].rolling(5).std()
    df['volatility_20'] = df['returns'].rolling(20).std()
    
    # Rangos intraday
    df['high_low_range'] = (df['high'] - df['low']) / (df['close'] + 1e-10)
    df['open_close_range'] = abs(df['open'] - df['close']) / (df['close'] + 1e-10)
    
    # Momentum
    df['roc_5'] = df['close'].pct_change(periods=5)
    df['roc_10'] = df['close'].pct_change(periods=10)
    df['roc_20'] = df['close'].pct_change(periods=20)
    
    # === INDICADORES DE TIEMPO ===
    
    # Extracción temporal (si disponible)
    if 'time' in df.columns and pd.api.types.is_datetime64_any_dtype(df['time']):
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.dayofweek
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Limpiar
    df = clean_financial_data(df)
    
    return df

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def clean_financial_data(df, max_abs_value=1e6, fill_method='ffill'):
    df_clean = df.copy()
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_clean[col] = np.clip(df_clean[col], -max_abs_value, max_abs_value)
    
    return df_clean

# ================================
# 🧠 MODELO MEJORADO
# ================================
class EnhancedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, 
                 output_size=5, dropout=0.3, bidirectional=True):
        super().__init__()
        
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM con más capacidad
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mejorado
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size, 1)
        )
        
        # Cabezas separadas para precio y volumen
        fc_input = hidden_size * self.num_directions
        
        # Cabeza de precio (OHLC)
        self.price_head = nn.Sequential(
            nn.Linear(fc_input, fc_input // 2),
            nn.LayerNorm(fc_input // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_input // 2, fc_input // 4),
            nn.LayerNorm(fc_input // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(fc_input // 4, 4)  # Open, High, Low, Close
        )
        
        # Cabeza de volumen
        self.volume_head = nn.Sequential(
            nn.Linear(fc_input, fc_input // 4),
            nn.LayerNorm(fc_input // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_input // 4, 1)  # Volume
        )
        
        self.output_activation = nn.Tanh()
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        # Predicciones separadas
        price_pred = self.price_head(context)
        volume_pred = self.volume_head(context)
        
        # Combinar
        output = torch.cat([price_pred, volume_pred], dim=1)
        output = self.output_activation(output) * 0.05
        
        return output

# ================================
# 📦 PREPARACIÓN MEJORADA
# ================================
def prepare_enhanced_dataset(df):
    print("\n" + "="*70)
    print("  🔧 PREPARACIÓN DE DATOS MEJORADOS - OHLCV")
    print("="*70)
    
    df = calculate_enhanced_indicators(df)
    
    # Targets con normalización especial para volumen
    df['delta_open'] = (df['open'].shift(-1) - df['close']) / (df['close'] + 1e-10)
    df['delta_high'] = (df['high'].shift(-1) - df['close']) / (df['close'] + 1e-10)
    df['delta_low'] = (df['low'].shift(-1) - df['close']) / (df['close'] + 1e-10)
    df['delta_close'] = (df['close'].shift(-1) - df['close']) / (df['close'] + 1e-10)
    
    # ✅ Volumen con log-transform
    df['delta_volume'] = np.log1p(df['volume'].shift(-1)) - np.log1p(df['volume'])
    
    initial_len = len(df)
    df = df.dropna()
    print(f"📊 Datos después de limpieza: {len(df):,} de {initial_len:,} velas")
    
    all_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['delta_open', 'delta_high', 'delta_low', 'delta_close', 'delta_volume']
    feature_cols = [col for col in all_numeric_cols if col not in exclude_cols]
    target_cols = ['delta_open', 'delta_high', 'delta_low', 'delta_close', 'delta_volume']
    
    print(f"\n🎯 {len(feature_cols)} Características")
    print(f"🎯 {len(target_cols)} Targets (OHLCV con volumen log-normalizado)")
    
    train_end = int(len(df) * Config.TRAIN_SIZE)
    val_end = int(len(df) * (Config.TRAIN_SIZE + Config.VAL_SIZE))
    
    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()
    
    print(f"\n📊 División:")
    print(f"   Train: {len(df_train):,} ({Config.TRAIN_SIZE*100:.0f}%)")
    print(f"   Val:   {len(df_val):,} ({Config.VAL_SIZE*100:.0f}%)")
    print(f"   Test:  {len(df_test):,} ({Config.TEST_SIZE*100:.0f}%)")
    
    # Scalers separados para mejor normalización
    scaler_in = RobustScaler(quantile_range=(10, 90))  # ✅ Más robusto a outliers
    scaler_out_price = RobustScaler(quantile_range=(10, 90))
    scaler_out_volume = RobustScaler(quantile_range=(10, 90))
    
    # Features
    X_train = scaler_in.fit_transform(df_train[feature_cols])
    X_val = scaler_in.transform(df_val[feature_cols])
    X_test = scaler_in.transform(df_test[feature_cols])
    
    # Targets: precio y volumen separados
    y_train_price = scaler_out_price.fit_transform(df_train[target_cols[:4]])  # OHLC
    y_train_volume = scaler_out_volume.fit_transform(df_train[[target_cols[4]]])  # Volume
    y_train = np.hstack([y_train_price, y_train_volume])
    
    y_val_price = scaler_out_price.transform(df_val[target_cols[:4]])
    y_val_volume = scaler_out_volume.transform(df_val[[target_cols[4]]])
    y_val = np.hstack([y_val_price, y_val_volume])
    
    y_test_price = scaler_out_price.transform(df_test[target_cols[:4]])
    y_test_volume = scaler_out_volume.transform(df_test[[target_cols[4]]])
    y_test = np.hstack([y_test_price, y_test_volume])
    
    def create_sequences(X, y, seq_len):
        X_seq, y_seq = [], []
        for i in range(seq_len, len(X)):
            X_seq.append(X[i-seq_len:i])
            y_seq.append(y[i-1])
        return np.array(X_seq), np.array(y_seq)
    
    print(f"\n🔄 Creando secuencias (seq_len={Config.SEQ_LEN})...")
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, Config.SEQ_LEN)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, Config.SEQ_LEN)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, Config.SEQ_LEN)
    
    print(f"✅ Train: X{X_train_seq.shape}, y{y_train_seq.shape}")
    print(f"✅ Val:   X{X_val_seq.shape}, y{y_val_seq.shape}")
    print(f"✅ Test:  X{X_test_seq.shape}, y{y_test_seq.shape}")
    
    scalers = {
        'input': scaler_in,
        'output_price': scaler_out_price,
        'output_volume': scaler_out_volume
    }
    
    return (X_train_seq, y_train_seq), (X_val_seq, y_val_seq), (X_test_seq, y_test_seq), \
           scalers, feature_cols, target_cols

# ================================
# 🏋️ LOSS MEJORADO
# ================================
class BalancedLoss(nn.Module):
    def __init__(self, price_weight=1.0, volume_weight=0.1, 
                 constraint_weight=1.0, realism_weight=0.5):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.price_weight = price_weight
        self.volume_weight = volume_weight
        self.constraint_weight = constraint_weight
        self.realism_weight = realism_weight
    
    def forward(self, predictions, targets):
        # Separar precio y volumen
        pred_price = predictions[:, :4]
        pred_volume = predictions[:, 4:]
        target_price = targets[:, :4]
        target_volume = targets[:, 4:]
        
        # MSE separado
        mse_price = self.mse(pred_price, target_price).mean()
        mse_volume = self.mse(pred_volume, target_volume).mean()
        
        # Restricciones OHLC
        delta_open = predictions[:, 0]
        delta_high = predictions[:, 1]
        delta_low = predictions[:, 2]
        delta_close = predictions[:, 3]
        
        high_low_violation = torch.clamp(delta_low - delta_high, min=0)
        close_below_low = torch.clamp(delta_low - delta_close, min=0)
        close_above_high = torch.clamp(delta_close - delta_high, min=0)
        constraint_loss = (high_low_violation + close_below_low + close_above_high).mean()
        
        # Restricciones de realismo
        max_delta_price = 0.05
        extreme_price = torch.clamp(torch.abs(pred_price) - max_delta_price, min=0).mean()
        extreme_volume = torch.clamp(torch.abs(pred_volume) - 0.5, min=0).mean()
        realism_loss = extreme_price + extreme_volume
        
        # Loss total balanceado
        total_loss = (
            self.price_weight * mse_price +
            self.volume_weight * mse_volume +
            self.constraint_weight * constraint_loss +
            self.realism_weight * realism_loss
        )
        
        return total_loss, {
            'mse_price': mse_price.item(),
            'mse_volume': mse_volume.item(),
            'constraint': constraint_loss.item(),
            'realism': realism_loss.item()
        }

# ================================
# 🏋️ ENTRENAMIENTO
# ================================
def train_enhanced_model(model, train_loader, val_loader, device):
    print("\n" + "="*70)
    print("  🏋️ ENTRENAMIENTO MEJORADO")
    print("="*70)
    
    criterion = BalancedLoss(
        price_weight=Config.PRICE_WEIGHT,
        volume_weight=Config.VOLUME_WEIGHT,
        constraint_weight=1.0,
        realism_weight=0.5
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )
    
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    print(f"\n⚙️  Configuración:")
    print(f"   Learning Rate: {Config.LEARNING_RATE}")
    print(f"   Batch Size: {Config.BATCH_SIZE}")
    print(f"   Price Weight: {Config.PRICE_WEIGHT}")
    print(f"   Volume Weight: {Config.VOLUME_WEIGHT}")
    print(f"   Grad Clip: {Config.GRAD_CLIP}")
    print()
    
    epoch_bar = tqdm(range(Config.EPOCHS), desc="Entrenando", unit="epoch")
    
    for epoch in epoch_bar:
        # ENTRENAMIENTO
        model.train()
        train_loss = 0
        train_components = {'mse_price': 0, 'mse_volume': 0, 'constraint': 0, 'realism': 0}
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss, components = criterion(predictions, y_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
            optimizer.step()
            
            train_loss += loss.item()
            for key in train_components:
                train_components[key] += components[key]
        
        train_loss /= len(train_loader)
        for key in train_components:
            train_components[key] /= len(train_loader)
        train_losses.append(train_loss)
        
        # VALIDACIÓN
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
        
        scheduler.step(val_loss)
        
        # Early stopping
        improvement = best_val_loss - val_loss
        if improvement > Config.MIN_DELTA:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            torch.save(best_model_state, 'best_enhanced_model_ohlcv.pth')
        else:
            patience_counter += 1
        
        # Actualizar barra
        current_lr = optimizer.param_groups[0]['lr']
        epoch_bar.set_postfix({
            'train': f'{train_loss:.4f}',
            'val': f'{val_loss:.4f}',
            'best': f'{best_val_loss:.4f}',
            'lr': f'{current_lr:.6f}',
            'patience': f'{patience_counter}/{Config.PATIENCE}'
        })
        
        # Log detallado
        if (epoch + 1) % 10 == 0:
            print(f"\n📊 Época {epoch+1}/{Config.EPOCHS}")
            print(f"   Train: {train_loss:.6f} (Price: {train_components['mse_price']:.6f}, "
                  f"Vol: {train_components['mse_volume']:.6f})")
            print(f"   Val: {val_loss:.6f} | Best: {best_val_loss:.6f}")
            print(f"   LR: {current_lr:.6f} | Patience: {patience_counter}/{Config.PATIENCE}")
        
        if patience_counter >= Config.PATIENCE:
            print(f"\n⏹️  Early stopping en época {epoch+1}")
            break
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    print(f"\n✅ Entrenamiento completado")
    print(f"   Mejor val loss: {best_val_loss:.6f}")
    print(f"   Épocas totales: {len(train_losses)}")
    
    return train_losses, val_losses

# ================================
# 📊 EVALUACIÓN
# ================================
def evaluate_enhanced_model(model, test_loader, scalers, target_cols, device):
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
    
    # Desnormalizar separadamente
    pred_price = scalers['output_price'].inverse_transform(predictions[:, :4])
    pred_volume = scalers['output_volume'].inverse_transform(predictions[:, 4:])
    predictions_denorm = np.hstack([pred_price, pred_volume])
    
    target_price = scalers['output_price'].inverse_transform(targets[:, :4])
    target_volume = scalers['output_volume'].inverse_transform(targets[:, 4:])
    targets_denorm = np.hstack([target_price, target_volume])
    
    metrics = {}
    print()
    for i, col in enumerate(target_cols):
        mae = mean_absolute_error(targets_denorm[:, i], predictions_denorm[:, i])
        rmse = np.sqrt(mean_squared_error(targets_denorm[:, i], predictions_denorm[:, i]))
        r2 = r2_score(targets_denorm[:, i], predictions_denorm[:, i])
        
        direction_true = np.sign(targets_denorm[:, i])
        direction_pred = np.sign(predictions_denorm[:, i])
        accuracy = np.mean(direction_true == direction_pred) * 100
        
        metrics[col] = {
            'MAE': float(mae),
            'RMSE': float(rmse),
            'R2': float(r2),
            'Direction_Accuracy': float(accuracy)
        }
        
        print(f"📊 {col}:")
        print(f"   MAE:  {mae:.6f} ({mae*100:.4f}%)")
        print(f"   RMSE: {rmse:.6f} ({rmse*100:.4f}%)")
        print(f"   R²:   {r2:.4f}")
        print(f"   Accuracy Direccional: {accuracy:.2f}%")
        print()
    
    return predictions_denorm, targets_denorm, metrics

# ================================
# 🎨 VISUALIZACIÓN
# ================================
def plot_enhanced_results(train_losses, val_losses, metrics, predictions, targets):
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    fig.suptitle('ADAUSD LSTM Mejorado - Predicción OHLCV', fontsize=16, fontweight='bold')
    
    # Pérdidas
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(train_losses, label='Train', linewidth=2, color='blue', alpha=0.7)
    ax1.plot(val_losses, label='Val', linewidth=2, color='orange', alpha=0.7)
    ax1.set_title('Loss durante Entrenamiento')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # R² comparación
    ax2 = fig.add_subplot(gs[0, 1])
    targets_names = list(metrics.keys())
    r2_scores = [metrics[t]['R2'] for t in targets_names]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    bars = ax2.bar(targets_names, r2_scores, color=colors)
    ax2.set_title('R² Score por Target')
    ax2.set_ylabel('R²')
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, score in zip(bars, r2_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}', ha='center', va='bottom' if score > 0 else 'top', fontsize=9)
    
    # Accuracy
    ax3 = fig.add_subplot(gs[0, 2])
    acc_scores = [metrics[t]['Direction_Accuracy'] for t in targets_names]
    bars = ax3.bar(targets_names, acc_scores, color=colors)
    ax3.set_title('Accuracy Direccional')
    ax3.set_ylabel('Accuracy (%)')
    ax3.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Random')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend()
    ax3.tick_params(axis='x', rotation=45)
    
    for bar, acc in zip(bars, acc_scores):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Scatterplots
    sample_size = min(300, len(predictions))
    titles = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    for idx, (title, color) in enumerate(zip(titles, colors)):
        row = 1 + idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        ax.scatter(targets[:sample_size, idx], predictions[:sample_size, idx], 
                  alpha=0.5, s=15, color=color, edgecolors='black', linewidth=0.3)
        
        # Línea ideal
        min_val, max_val = targets[:, idx].min(), targets[:, idx].max()
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, linewidth=2)
        
        r2 = metrics[f'delta_{title.lower()}']['R2']
        acc = metrics[f'delta_{title.lower()}']['Direction_Accuracy']
        ax.set_title(f'{title} (R²={r2:.3f}, Acc={acc:.1f}%)')
        ax.set_xlabel('Real')
        ax.set_ylabel('Predicción')
        ax.grid(True, alpha=0.3)
    
    plt.savefig('adausd_enhanced_ohlcv_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("📈 Gráficas guardadas en 'adausd_enhanced_ohlcv_results.png'")

# ================================
# 🚀 MAIN
# ================================
def main():
    try:
        print("\n" + "="*70)
        print("  🚀 LSTM MEJORADO - PREDICCIÓN OHLCV")
        print("="*70)
        
        print("\n⚙️  MEJORAS IMPLEMENTADAS:")
        print("   ✅ Volumen con log-normalización")
        print("   ✅ Scalers separados para precio/volumen")
        print("   ✅ Arquitectura más robusta (128 hidden, 3 layers)")
        print("   ✅ Cabezas separadas para precio y volumen")
        print("   ✅ Feature engineering mejorado (+20 indicadores)")
        print("   ✅ Loss balanceado (precio 1.0, volumen 0.1)")
        print("   ✅ SEQ_LEN aumentado a 90")
        print("   ✅ Learning rate reducido (0.0001)")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n🖥️  Device: {device}")
        
        print("\n" + "="*70)
        print("  📥 DESCARGANDO DATOS")
        print("="*70)
        
        ticker = yf.Ticker("ADA-USD")
        df = ticker.history(period="2y", interval="1h")
        
        if len(df) == 0:
            raise ValueError("No se pudieron descargar datos")
        
        df = df.reset_index()
        df.columns = [col.lower() for col in df.columns]
        
        rename_dict = {}
        for col in df.columns:
            if 'date' in col or 'time' in col:
                rename_dict[col] = 'time'
        df.rename(columns=rename_dict, inplace=True)
        
        required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        df = df[required_cols]
        
        print(f"✅ Datos descargados: {len(df):,} velas")
        
        (X_train_seq, y_train_seq), (X_val_seq, y_val_seq), (X_test_seq, y_test_seq), \
        scalers, feature_cols, target_cols = prepare_enhanced_dataset(df)
        
        input_size = len(feature_cols)
        output_size = len(target_cols)
        
        print(f"\n📊 Dimensiones:")
        print(f"   Input: {input_size} features")
        print(f"   Output: {output_size} targets")
        
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
        
        model = EnhancedLSTM(
            input_size=input_size,
            hidden_size=Config.HIDDEN_SIZE,
            num_layers=Config.NUM_LAYERS,
            output_size=output_size,
            dropout=Config.DROPOUT,
            bidirectional=Config.BIDIRECTIONAL
        ).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n🧠 Modelo: {total_params:,} parámetros")
        
        start_time = time.time()
        train_losses, val_losses = train_enhanced_model(model, train_loader, val_loader, device)
        training_time = time.time() - start_time
        
        print(f"\n⏱️  Tiempo: {training_time/60:.1f} min")
        
        predictions, targets, metrics = evaluate_enhanced_model(
            model, test_loader, scalers, target_cols, device
        )
        
        print("="*70)
        print("  📈 RESULTADOS FINALES")
        print("="*70)
        
        # Separar métricas de precio y volumen
        price_metrics = {k: v for k, v in metrics.items() if 'volume' not in k}
        volume_metrics = {k: v for k, v in metrics.items() if 'volume' in k}
        
        avg_r2_price = np.mean([m['R2'] for m in price_metrics.values()])
        avg_acc_price = np.mean([m['Direction_Accuracy'] for m in price_metrics.values()])
        
        print(f"\n📊 Precio (OHLC):")
        print(f"   R² promedio: {avg_r2_price:.4f}")
        print(f"   Accuracy promedio: {avg_acc_price:.2f}%")
        
        if volume_metrics:
            vol_r2 = volume_metrics['delta_volume']['R2']
            vol_acc = volume_metrics['delta_volume']['Direction_Accuracy']
            print(f"\n📊 Volumen:")
            print(f"   R²: {vol_r2:.4f}")
            print(f"   Accuracy: {vol_acc:.2f}%")
        
        plot_enhanced_results(train_losses, val_losses, metrics, predictions, targets)
        
        model_dir = 'ADAUSD_MODELS'
        os.makedirs(model_dir, exist_ok=True)
        
        model_config = {
            'model_state_dict': model.state_dict(),
            'input_size': input_size,
            'hidden_size': Config.HIDDEN_SIZE,
            'num_layers': Config.NUM_LAYERS,
            'output_size': output_size,
            'seq_len': Config.SEQ_LEN,
            'bidirectional': Config.BIDIRECTIONAL,
            'dropout': Config.DROPOUT,
            'feature_cols': feature_cols,
            'target_cols': target_cols,
            'metrics_test': metrics,
            'timestamp': datetime.now().isoformat(),
            'training_time_minutes': training_time / 60,
            'total_epochs': len(train_losses),
            'best_val_loss': min(val_losses)
        }
        
        torch.save(model_config, f'{model_dir}/adausd_enhanced_lstm_ohlcv.pth')
        
        # Guardar scalers
        joblib.dump(scalers['input'], f'{model_dir}/scaler_input_enhanced.pkl')
        joblib.dump(scalers['output_price'], f'{model_dir}/scaler_output_price_enhanced.pkl')
        joblib.dump(scalers['output_volume'], f'{model_dir}/scaler_output_volume_enhanced.pkl')
        
        json_config = {
            'input_size': input_size,
            'hidden_size': Config.HIDDEN_SIZE,
            'num_layers': Config.NUM_LAYERS,
            'output_size': output_size,
            'seq_len': Config.SEQ_LEN,
            'feature_cols': feature_cols,
            'target_cols': target_cols,
            'metrics_test': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(f'{model_dir}/config_enhanced_ohlcv.json', 'w') as f:
            json.dump(json_config, f, indent=2)
        
        print(f"\n💾 Modelo guardado en '{model_dir}/'")
        
        msg = f"""
✅ *LSTM Mejorado OHLCV*

📊 *Precio (OHLC):*
   • R² promedio: {avg_r2_price:.4f}
   • Accuracy: {avg_acc_price:.2f}%

📊 *Volumen:*
   • R²: {vol_r2:.4f}
   • Accuracy: {vol_acc:.2f}%

⚙️ *Mejoras:*
   • Log-normalización volumen
   • Cabezas separadas precio/volumen
   • {total_params:,} parámetros
   • {len(feature_cols)} features

⏱️ *Tiempo:* {training_time/60:.1f} min
"""
        send_telegram(msg)
        
        print("\n" + "="*70)
        print("  ✅ PROCESO COMPLETADO")
        print("="*70 + "\n")
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        print(f"\n{error_msg}")
        import traceback
        traceback.print_exc()
        send_telegram(error_msg)
        raise

def send_telegram(msg):
    TELEGRAM_API = os.environ.get('TELEGRAM_API', '')
    CHAT_ID = os.environ.get('CHAT_ID', '')
    
    if not TELEGRAM_API or not CHAT_ID:
        return
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_API}/sendMessage"
        requests.post(url, data={'chat_id': CHAT_ID, 'text': msg, 'parse_mode': 'Markdown'}, timeout=10)
    except Exception as e:
        print(f"❌ Telegram: {e}")

if __name__ == "__main__":
    main()
