"""
LSTM HÍBRIDO CON DERIVADAS DE VOLUMEN MEJORADO
✅ Implementación basada en el artículo MQL5
✅ LSTM bidireccional con atención
✅ Derivadas de volumen (1ra y 2da) optimizadas
✅ Indicadores avanzados: OBV, VWAP, PVT, divergencias
✅ Sistema híbrido precio-volumen
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
# 🎛️ CONFIGURACIÓN HÍBRIDA
# ================================
class Config:
    # Características híbridas (precio + volumen)
    USE_VOLUME = True
    USE_VOLUME_DERIVATIVES = True  # Derivadas 1ra y 2da
    USE_VOLUME_INDICATORS = True   # OBV, VWAP, PVT
    USE_PRICE_DERIVATIVES = True   # Derivadas de precio
    
    # Targets
    PREDICT_ABSOLUTE = False       # Predecir deltas relativos
    PREDICT_DELTAS = True          # Predicción principal
    
    # Normalización
    SCALER_TYPE = 'robust'
    
    # Arquitectura LSTM híbrida
    SEQ_LEN = 30
    HIDDEN_SIZE = 128              # Mayor capacidad
    NUM_LAYERS = 2
    DROPOUT = 0.3
    BIDIRECTIONAL = True           # LSTM bidireccional
    
    # Entrenamiento
    BATCH_SIZE = 64
    EPOCHS = 100
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    PATIENCE = 15
    
    # Datos
    TRAIN_SIZE = 0.70
    VAL_SIZE = 0.15
    TEST_SIZE = 0.15

# ================================
# 📊 INDICADORES HÍBRIDOS (Precio + Volumen)
# ================================
def calculate_hybrid_indicators(df):
    """
    Calcula indicadores híbridos basados en el artículo MQL5
    Combinación de precio y volumen para detectar divergencias
    """
    df = df.copy()
    
    # 1. Indicadores de volumen mejorados
    # Derivadas robustas del volumen
    df['volume_1st_deriv'] = df['volume'].diff()
    df['volume_2nd_deriv'] = df['volume_1st_deriv'].diff()
    
    # Suavizar derivadas
    df['volume_1st_deriv_smooth'] = df['volume_1st_deriv'].rolling(window=5, center=True).mean()
    df['volume_2nd_deriv_smooth'] = df['volume_2nd_deriv'].rolling(window=5, center=True).mean()
    
    # 2. On-Balance Volume (OBV) optimizado
    obv = [0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.append(obv[-1] + df['volume'].iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.append(obv[-1] - df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['obv'] = obv
    
    # OBV normalizado
    df['obv_norm'] = (df['obv'] - df['obv'].rolling(window=50).mean()) / df['obv'].rolling(window=50).std()
    df['obv_roc'] = df['obv'].pct_change(periods=14)
    
    # 3. Volume Weighted Average Price (VWAP)
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (df['typical_price'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Desviación del VWAP
    df['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap'] * 100
    
    # 4. Price-Volume Trend (PVT)
    df['pvt'] = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1) * df['volume']).cumsum()
    df['pvt_ma'] = df['pvt'].rolling(window=20).mean()
    df['pvt_signal'] = (df['pvt'] > df['pvt_ma']).astype(int)
    
    # 5. Relación volumen/precio
    df['volume_price_ratio'] = df['volume'] / df['close'].rolling(window=20).mean()
    df['volume_volatility_ratio'] = df['volume'] / df['volume'].rolling(window=20).std()
    
    # 6. Divergencias precio-volumen (como en MQL5)
    # Pendiente del precio (14 periodos)
    def calculate_slope(series, window=14):
        slopes = []
        for i in range(len(series)):
            if i < window:
                slopes.append(0)
            else:
                x = np.arange(window)
                y = series.iloc[i-window:i].values
                slope = np.polyfit(x, y, 1)[0]
                slopes.append(slope)
        return pd.Series(slopes, index=series.index)
    
    df['price_slope_14'] = calculate_slope(df['close'], 14)
    df['obv_slope_14'] = calculate_slope(df['obv'], 14)
    
    # Detectar divergencias
    df['bullish_divergence'] = ((df['price_slope_14'] < 0) & (df['obv_slope_14'] > 0)).astype(int)
    df['bearish_divergence'] = ((df['price_slope_14'] > 0) & (df['obv_slope_14'] < 0)).astype(int)
    
    # 7. Indicadores de momentum con volumen
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['volume_rsi'] = calculate_rsi(df['volume'], 14)
    
    # 8. Aceleración/desaceleración del volumen
    df['volume_acceleration'] = df['volume_1st_deriv'].diff()
    
    # 9. Relación entre derivadas de precio y volumen
    df['price_1st_deriv'] = df['close'].diff()
    df['price_2nd_deriv'] = df['price_1st_deriv'].diff()
    
    df['price_volume_correlation'] = df['price_1st_deriv'].rolling(window=20).corr(df['volume_1st_deriv'])
    
    # 10. Volatilidad ajustada por volumen
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=20).std()
    df['volume_adjusted_volatility'] = df['volatility'] * (df['volume'] / df['volume'].rolling(window=20).mean())
    
    # Rellenar valores NaN
    df = clean_financial_data(df, max_abs_value=1e6, fill_method='ffill')
    
    return df

def calculate_rsi(series, period=14):
    """Calcula RSI robusto"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def clean_financial_data(df, max_abs_value=1e6, fill_method='ffill'):
    """Limpieza completa de datos financieros"""
    df_clean = df.copy()
    
    # Reemplazar infinitos
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    
    # Rellenar NaN
    if fill_method == 'ffill':
        df_clean = df_clean.fillna(method='ffill')
    elif fill_method == 'bfill':
        df_clean = df_clean.fillna(method='bfill')
    
    df_clean = df_clean.fillna(0)
    
    # Recortar valores extremos
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_clean[col] = np.clip(df_clean[col], -max_abs_value, max_abs_value)
    
    return df_clean

# ================================
# 🧠 MODELO LSTM HÍBRIDO BIDIRECCIONAL
# ================================
class HybridLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, 
                 output_size=3, dropout=0.3, bidirectional=True):
        super().__init__()
        
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM bidireccional
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Mecanismo de atención
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Capas fully connected
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size // 2, output_size)
        )
        
        # Capa de salida con activación tanh para deltas (-1 a 1)
        self.output_activation = nn.Tanh()
    
    def forward(self, x):
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Atención
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        # Fully connected
        output = self.fc(context)
        
        # Activación de salida (deltas entre -10% y +10%)
        output = self.output_activation(output) * 0.10
        
        return output

# ================================
# 📦 PREPARACIÓN DE DATOS HÍBRIDOS
# ================================
def prepare_hybrid_dataset(df):
    """
    Prepara dataset con indicadores híbridos y deltas
    """
    print("\n" + "="*70)
    print("  🔧 PREPARACIÓN DE DATOS HÍBRIDOS")
    print("="*70)
    
    # 1. Calcular indicadores híbridos
    df = calculate_hybrid_indicators(df)
    
    # 2. Calcular deltas para la próxima vela
    df['delta_high'] = (df['high'].shift(-1) - df['close']) / df['close']
    df['delta_low'] = (df['low'].shift(-1) - df['close']) / df['close']
    df['delta_close'] = (df['close'].shift(-1) - df['close']) / df['close']
    
    # 3. Eliminar NaN
    initial_len = len(df)
    df = df.dropna()
    print(f"📊 Datos después de limpieza: {len(df):,} de {initial_len:,} velas")
    
    # 4. Seleccionar características (todas las numéricas excepto targets)
    all_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Excluir targets y columnas futuras
    exclude_cols = ['delta_high', 'delta_low', 'delta_close']
    feature_cols = [col for col in all_numeric_cols if col not in exclude_cols]
    
    # Asegurar columnas básicas
    basic_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in basic_cols:
        if col in df.columns and col not in feature_cols:
            feature_cols.append(col)
    
    # Targets
    target_cols = ['delta_high', 'delta_low', 'delta_close']
    
    print(f"\n🎯 {len(feature_cols)} Características híbridas")
    print(f"🎯 {len(target_cols)} Targets (deltas)")
    
    # 5. Dividir datos temporalmente
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
    
    # 6. Normalización robusta
    scaler_in = RobustScaler(quantile_range=(25, 75))
    scaler_out = RobustScaler(quantile_range=(25, 75))
    
    print(f"\n🔧 Normalización con {scaler_in.__class__.__name__}...")
    
    # Validar datos antes de escalar
    for df_part, name in [(df_train, 'train'), (df_val, 'val'), (df_test, 'test')]:
        for col in feature_cols:
            if col not in df_part.columns:
                print(f"❌ ERROR: Columna {col} no existe en {name}")
                raise ValueError(f"Columna {col} no existe")
    
    # Escalar características
    X_train = scaler_in.fit_transform(df_train[feature_cols])
    X_val = scaler_in.transform(df_val[feature_cols])
    X_test = scaler_in.transform(df_test[feature_cols])
    
    # Escalar targets
    y_train = scaler_out.fit_transform(df_train[target_cols])
    y_val = scaler_out.transform(df_val[target_cols])
    y_test = scaler_out.transform(df_test[target_cols])
    
    # 7. Crear secuencias
    def create_sequences(X, y, seq_len):
        X_seq, y_seq = [], []
        for i in range(seq_len, len(X)):
            X_seq.append(X[i-seq_len:i])
            y_seq.append(y[i-1])
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
# 🏋️ ENTRENAMIENTO CON PÉRDIDA PERSONALIZADA
# ================================
class HybridLoss(nn.Module):
    """
    Pérdida híbrida para LSTM bidireccional
    Combina MSE con restricciones de coherencia
    """
    def __init__(self, mse_weight=1.0, constraint_weight=2.0, realism_weight=1.0):
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

def train_hybrid_model(model, train_loader, val_loader, device):
    """Entrena el modelo LSTM híbrido"""
    print("\n" + "="*70)
    print("  🏋️ ENTRENANDO LSTM HÍBRIDO BIDIRECCIONAL")
    print("="*70)
    
    criterion = HybridLoss(
        mse_weight=1.0,
        constraint_weight=2.0,
        realism_weight=1.0
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(Config.EPOCHS):
        # Entrenamiento
        model.train()
        train_loss = 0
        train_components = {'mse': 0, 'constraint': 0, 'realism': 0}
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss, components = criterion(predictions, y_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            for key in train_components:
                train_components[key] += components[key]
        
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
        
        # Scheduler
        scheduler.step(val_loss)
        
        # Guardar mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            torch.save(model.state_dict(), 'best_hybrid_model.pth')
        
        # Log cada 10 épocas
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\n📊 Epoch {epoch+1}/{Config.EPOCHS}")
            print(f"  LR: {current_lr:.6f}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss:   {val_loss:.6f}")
            print(f"  Best Val:   {best_val_loss:.6f}")
    
    # Cargar mejor modelo
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    print(f"\n✅ Entrenamiento completado")
    print(f"   Mejor val loss: {best_val_loss:.6f}")
    print(f"   Épocas totales: {len(train_losses)}")
    
    return train_losses, val_losses

# ================================
# 📈 EVALUACIÓN HÍBRIDA
# ================================
def evaluate_hybrid_model(model, test_loader, scaler_out, target_cols, device):
    """Evalúa el modelo LSTM híbrido"""
    print("\n" + "="*70)
    print("  📊 EVALUACIÓN LSTM HÍBRIDO")
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
        
        # Accuracy direccional
        direction_true = np.sign(targets_denorm[:, i])
        direction_pred = np.sign(predictions_denorm[:, i])
        accuracy = np.mean(direction_true == direction_pred) * 100
        
        metrics[col] = {
            'MAE': float(mae),
            'RMSE': float(rmse),
            'R2': float(r2),
            'Direction_Accuracy': float(accuracy)
        }
        
        print(f"\n📊 {col}:")
        print(f"   MAE:  {mae:.6f} ({mae*100:.4f}%)")
        print(f"   RMSE: {rmse:.6f} ({rmse*100:.4f}%)")
        print(f"   R²:   {r2:.4f}")
        print(f"   Accuracy Direccional: {accuracy:.2f}%")
    
    return predictions_denorm, targets_denorm, metrics

# ================================
# 🎨 VISUALIZACIÓN HÍBRIDA
# ================================
def plot_hybrid_results(train_losses, val_losses, metrics, predictions, targets):
    """Crea gráficas para el modelo híbrido"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('ADAUSD LSTM Híbrido Bidireccional - Resultados', fontsize=16, fontweight='bold')
    
    # Pérdidas
    ax1 = axes[0, 0]
    ax1.plot(train_losses, label='Train', linewidth=2, color='blue')
    ax1.plot(val_losses, label='Val', linewidth=2, color='orange')
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
    
    for bar, score in zip(bars, r2_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}', ha='center', va='bottom')
    
    # Accuracy direccional
    ax3 = axes[0, 2]
    acc_scores = [metrics[t]['Direction_Accuracy'] for t in targets_names]
    bars = ax3.bar(targets_names, acc_scores, color=colors)
    ax3.set_title('Accuracy Direccional')
    ax3.set_ylabel('Accuracy (%)')
    ax3.axhline(y=50, color='r', linestyle='--', alpha=0.5)
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, acc in zip(bars, acc_scores):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%', ha='center', va='bottom')
    
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
    plt.savefig('adausd_hybrid_lstm_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n📈 Gráficas guardadas en 'adausd_hybrid_lstm_results.png'")

# ================================
# 🚀 MAIN CORREGIDO
# ================================
def main():
    try:
        print("\n" + "="*70)
        print("  🚀 LSTM HÍBRIDO BIDIRECCIONAL PARA ADAUSD")
        print("="*70)
        
        # Configuración
        print("\n⚙️  CONFIGURACIÓN HÍBRIDA:")
        print(f"   Sequence Length: {Config.SEQ_LEN}")
        print(f"   Hidden Size: {Config.HIDDEN_SIZE}")
        print(f"   Layers: {Config.NUM_LAYERS}")
        print(f"   Dropout: {Config.DROPOUT}")
        print(f"   Bidireccional: {Config.BIDIRECTIONAL}")
        print(f"   Batch Size: {Config.BATCH_SIZE}")
        
        # Device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n🖥️  Device: {device}")
        
        # 1. Descargar datos
        print("\n" + "="*70)
        print("  📥 DESCARGANDO DATOS")
        print("="*70)
        
        # Usar yfinance para descargar datos
        ticker = yf.Ticker("ADA-USD")
        df = ticker.history(period="2y", interval="1h")
        
        if len(df) == 0:
            raise ValueError("No se pudieron descargar datos")
        
        # Formatear DataFrame
        df = df.reset_index()
        df.columns = [col.lower() for col in df.columns]
        
        # Renombrar columnas
        rename_dict = {}
        for col in df.columns:
            if 'date' in col or 'time' in col:
                rename_dict[col] = 'time'
        df.rename(columns=rename_dict, inplace=True)
        
        # Seleccionar columnas necesarias
        required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        df = df[required_cols]
        
        print(f"✅ Datos descargados: {len(df):,} velas")
        
        # 2. Preparar dataset híbrido
        (X_train_seq, y_train_seq), (X_val_seq, y_val_seq), (X_test_seq, y_test_seq), \
        scaler_in, scaler_out, feature_cols, target_cols = prepare_hybrid_dataset(df)
        
        input_size = len(feature_cols)
        output_size = len(target_cols)
        
        print(f"\n📊 Dimensiones del modelo:")
        print(f"   Input size: {input_size}")
        print(f"   Output size: {output_size}")
        print(f"   Características: {feature_cols}")
        
        # 3. Crear DataLoaders
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
        
        # 4. Crear modelo LSTM híbrido
        model = HybridLSTM(
            input_size=input_size,
            hidden_size=Config.HIDDEN_SIZE,
            num_layers=Config.NUM_LAYERS,
            output_size=output_size,
            dropout=Config.DROPOUT,
            bidirectional=Config.BIDIRECTIONAL
        ).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n🧠 Modelo LSTM híbrido creado: {total_params:,} parámetros")
        
        # 5. Entrenar modelo
        start_time = time.time()
        train_losses, val_losses = train_hybrid_model(model, train_loader, val_loader, device)
        training_time = time.time() - start_time
        
        print(f"\n⏱️  Tiempo de entrenamiento: {training_time/60:.1f} minutos")
        print(f"📉 Mejor val loss: {min(val_losses):.6f}")
        
        # 6. Evaluar modelo
        predictions, targets, metrics = evaluate_hybrid_model(
            model, test_loader, scaler_out, target_cols, device
        )
        
        # 7. Resultados finales
        print("\n" + "="*70)
        print("  📈 RESULTADOS FINALES")
        print("="*70)
        
        avg_r2 = np.mean([metrics[t]['R2'] for t in metrics.keys()])
        avg_accuracy = np.mean([metrics[t]['Direction_Accuracy'] for t in metrics.keys()])
        
        print(f"\n📊 Métricas promedio:")
        print(f"   R² promedio: {avg_r2:.4f}")
        print(f"   Accuracy direccional promedio: {avg_accuracy:.2f}%")
        
        # 8. Visualizar resultados
        plot_hybrid_results(train_losses, val_losses, metrics, predictions, targets)
        
        # 9. Guardar modelo
        model_dir = 'ADAUSD_HYBRID_MODELS'
        os.makedirs(model_dir, exist_ok=True)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'input_size': input_size,
                'hidden_size': Config.HIDDEN_SIZE,
                'num_layers': Config.NUM_LAYERS,
                'output_size': output_size,
                'seq_len': Config.SEQ_LEN,
                'bidirectional': Config.BIDIRECTIONAL
            },
            'metrics': metrics,
            'scaler_in': scaler_in,
            'scaler_out': scaler_out,
            'feature_cols': feature_cols,
            'target_cols': target_cols
        }, f'{model_dir}/adausd_hybrid_lstm.pth')
        
        print(f"\n💾 Modelo guardado en '{model_dir}/adausd_hybrid_lstm.pth'")
        
        # 10. Telegram notification
        msg = f"""
✅ LSTM Híbrido entrenado exitosamente
📊 Resultados:
   • R² promedio: {avg_r2:.4f}
   • Accuracy direccional: {avg_accuracy:.2f}%
   • Tiempo de entrenamiento: {training_time/60:.1f} min
   • Mejor val loss: {min(val_losses):.6f}
"""
        send_telegram(msg)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Telegram error notification
        send_telegram(f"❌ Error en LSTM Híbrido: {str(e)}")

# Función Telegram (debes configurar tus variables de entorno)
def send_telegram(msg):
    """Envía mensaje a Telegram"""
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
