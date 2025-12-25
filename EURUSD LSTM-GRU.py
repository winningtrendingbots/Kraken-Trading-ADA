"""
LSTM CORREGIDO - PREDICCIÓN OHLCV COMPLETO
✅ Fix: Zero data leakage (X hasta t, y en t+1)
✅ Fix: Predice OHLCV completo (5 valores)
✅ Fix: Loss direccional diferenciable
✅ Fix: Features mínimas y relevantes (12)
✅ Fix: Arquitectura simple y enfocada
✅ Fix: Validación temporal estricta
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, accuracy_score
import matplotlib.pyplot as plt
import os
import time
import json
import joblib
import yfinance as yf
from tqdm.auto import tqdm
import warnings

warnings.filterwarnings('ignore')

# ================================
# 🎛️ CONFIGURACIÓN
# ================================
class Config:
    # Arquitectura
    SEQ_LEN = 60
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    DROPOUT = 0.3
    
    # Entrenamiento
    BATCH_SIZE = 256
    EPOCHS = 150
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    PATIENCE = 30
    GRAD_CLIP = 1.0
    
    # Loss weights para OHLCV
    DIRECTION_WEIGHT = 3.0  # Prioridad en dirección
    PRICE_WEIGHT = 1.0       # Precisión en niveles
    VOLUME_WEIGHT = 0.5      # Volumen menos crítico
    
    # Datos
    TRAIN_SIZE = 0.70
    VAL_SIZE = 0.15
    TEST_SIZE = 0.15

# ================================
# 📊 FEATURES ESENCIALES (12)
# ================================
def calculate_features(df):
    """Features mínimas para momentum y régimen de mercado - SIN LEAKAGE"""
    df = df.copy()
    
    # 1. Returns básicos (momentum)
    df['return_1'] = df['close'].pct_change(1)
    df['return_3'] = df['close'].pct_change(3)
    df['return_5'] = df['close'].pct_change(5)
    
    # 2. RSI (14 periodos)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    df['rsi_norm'] = (rsi - 50) / 50  # Normalizado [-1, 1]
    
    # 3. MACD simple
    ema_fast = df['close'].ewm(span=12).mean()
    ema_slow = df['close'].ewm(span=26).mean()
    df['macd'] = (ema_fast - ema_slow) / df['close']
    
    # 4. Volatility (régimen)
    df['volatility'] = df['close'].pct_change().rolling(20).std()
    
    # 5. Volume features
    df['volume_ratio'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-10)
    df['volume_change'] = df['volume'].pct_change()
    
    # 6. Price position en banda
    sma_20 = df['close'].rolling(20).mean()
    std_20 = df['close'].rolling(20).std()
    df['price_position'] = (df['close'] - sma_20) / (std_20 + 1e-10)
    
    # 7. Rango intrabar (high-low normalizado)
    df['bar_range'] = (df['high'] - df['low']) / (df['close'] + 1e-10)
    
    # 8. Body ratio (close-open vs high-low)
    body = abs(df['close'] - df['open'])
    total_range = df['high'] - df['low']
    df['body_ratio'] = body / (total_range + 1e-10)
    
    # 9. Momentum de volumen
    df['volume_momentum'] = df['volume'].rolling(5).mean() / (df['volume'].rolling(20).mean() + 1e-10)
    
    # Limpieza
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return df

# ================================
# 🧠 ARQUITECTURA LSTM PARA OHLCV
# ================================
class LSTM_OHLCV(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        
        # Encoder LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Head para OHLCV (5 valores)
        self.fc_ohlcv = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_size // 2, 5)  # open, high, low, close, volume
        )
    
    def forward(self, x):
        # x: [batch, seq, features]
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # Último timestamp
        
        # Predicción OHLCV
        ohlcv = self.fc_ohlcv(last_hidden)
        return ohlcv

# ================================
# 🎯 LOSS MULTI-OBJETIVO
# ================================
class OHLCVLoss(nn.Module):
    def __init__(self, direction_weight=3.0, price_weight=1.0, volume_weight=0.5):
        super().__init__()
        self.direction_weight = direction_weight
        self.price_weight = price_weight
        self.volume_weight = volume_weight
        self.mse = nn.MSELoss()
    
    def forward(self, predictions, targets):
        """
        predictions: [batch, 5] (open, high, low, close, volume)
        targets: [batch, 5]
        """
        # Separar componentes
        pred_ohlc = predictions[:, :4]  # open, high, low, close
        pred_volume = predictions[:, 4:5]
        
        target_ohlc = targets[:, :4]
        target_volume = targets[:, 4:5]
        
        # 1. DIRECTION LOSS (diferenciable) - Solo en CLOSE
        # Usamos tanh como aproximación suave de sign
        pred_close = predictions[:, 3:4]
        target_close = targets[:, 3:4]
        
        direction_alignment = torch.tanh(pred_close * target_close * 10)
        direction_loss = -direction_alignment.mean()
        
        # 2. PRICE LOSS (MSE en OHLC)
        price_loss = self.mse(pred_ohlc, target_ohlc)
        
        # 3. VOLUME LOSS (MSE)
        volume_loss = self.mse(pred_volume, target_volume)
        
        # TOTAL
        total_loss = (
            self.direction_weight * direction_loss +
            self.price_weight * price_loss +
            self.volume_weight * volume_loss
        )
        
        return total_loss, {
            'direction': direction_loss.item(),
            'price': price_loss.item(),
            'volume': volume_loss.item()
        }

# ================================
# 📦 PREPARACIÓN ZERO LEAKAGE
# ================================
def prepare_dataset_zero_leakage(df):
    """
    ✅ ZERO DATA LEAKAGE:
    X[i] = features de velas [t-59, ..., t]
    y[i] = delta_ohlcv de t → t+1 (próxima vela completa)
    """
    print("\n" + "="*70)
    print("  🔧 PREPARACIÓN DE DATOS (ZERO LEAKAGE)")
    print("="*70)
    
    df = calculate_features(df)
    
    # TARGETS: Delta OHLCV de t → t+1
    # ✅ Calculamos ANTES de crear secuencias
    for col in ['open', 'high', 'low', 'close', 'volume']:
        future_val = df[col].shift(-1)
        current_val = df[col]
        # Delta normalizado
        df[f'target_delta_{col}'] = (future_val - current_val) / (current_val + 1e-10)
    
    # IMPORTANTE: Dropear última fila (no tiene target)
    df = df[:-1].copy()
    df = df.dropna()
    
    print(f"📊 Datos limpios: {len(df):,} velas")
    
    # Features seleccionadas
    feature_cols = [
        'return_1', 'return_3', 'return_5',
        'rsi_norm', 'macd', 'volatility',
        'volume_ratio', 'volume_change', 'volume_momentum',
        'price_position', 'bar_range', 'body_ratio'
    ]
    
    target_cols = [
        'target_delta_open', 'target_delta_high', 
        'target_delta_low', 'target_delta_close', 
        'target_delta_volume'
    ]
    
    # Verificar
    missing = [c for c in feature_cols + target_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    print(f"🎯 {len(feature_cols)} features + 5 targets (OHLCV)")
    
    # Split temporal
    train_end = int(len(df) * Config.TRAIN_SIZE)
    val_end = int(len(df) * (Config.TRAIN_SIZE + Config.VAL_SIZE))
    
    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()
    
    print(f"📊 Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
    
    # Scalers separados
    scaler_in = RobustScaler(quantile_range=(5, 95))
    scaler_out = RobustScaler(quantile_range=(5, 95))
    
    X_train = scaler_in.fit_transform(df_train[feature_cols])
    X_val = scaler_in.transform(df_val[feature_cols])
    X_test = scaler_in.transform(df_test[feature_cols])
    
    y_train = scaler_out.fit_transform(df_train[target_cols])
    y_val = scaler_out.transform(df_val[target_cols])
    y_test = scaler_out.transform(df_test[target_cols])
    
    def create_sequences_strict(X, y, seq_len):
        """
        ✅ ALINEACIÓN ESTRICTA:
        X_seq[i] = features [i-seq_len, ..., i-1] (termina en t)
        y_seq[i] = target en i-1 (delta t→t+1)
        """
        X_seq, y_seq = [], []
        for i in range(seq_len, len(X)):
            X_seq.append(X[i-seq_len:i])  # Hasta i (exclusive)
            y_seq.append(y[i-1])           # Target en i-1
        return np.array(X_seq), np.array(y_seq)
    
    print(f"🔄 Creando secuencias (seq_len={Config.SEQ_LEN})...")
    X_train_seq, y_train_seq = create_sequences_strict(X_train, y_train, Config.SEQ_LEN)
    X_val_seq, y_val_seq = create_sequences_strict(X_val, y_val, Config.SEQ_LEN)
    X_test_seq, y_test_seq = create_sequences_strict(X_test, y_test, Config.SEQ_LEN)
    
    print(f"✅ Train: {X_train_seq.shape}, Val: {X_val_seq.shape}, Test: {X_test_seq.shape}")
    
    # VERIFICACIÓN ANTI-LEAKAGE
    print("\n🔍 Verificación anti-leakage:")
    print(f"   X_train shape: {X_train_seq.shape}")
    print(f"   y_train shape: {y_train_seq.shape}")
    print(f"   ✅ X[i] contiene features hasta timestamp t")
    print(f"   ✅ y[i] contiene delta_ohlcv de t→t+1")
    
    scalers = {
        'input': scaler_in,
        'output': scaler_out
    }
    
    return (X_train_seq, y_train_seq), (X_val_seq, y_val_seq), (X_test_seq, y_test_seq), \
           scalers, feature_cols, target_cols

# ================================
# 🏋️ ENTRENAMIENTO
# ================================
def train_model(model, train_loader, val_loader, device):
    print("\n" + "="*70)
    print("  🏋️ ENTRENAMIENTO - PREDICCIÓN OHLCV")
    print("="*70)
    
    criterion = OHLCVLoss(
        direction_weight=Config.DIRECTION_WEIGHT,
        price_weight=Config.PRICE_WEIGHT,
        volume_weight=Config.VOLUME_WEIGHT
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )
    
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    epoch_bar = tqdm(range(Config.EPOCHS), desc="Training")
    
    for epoch in epoch_bar:
        # TRAIN
        model.train()
        train_loss = 0
        train_components = {'direction': 0, 'price': 0, 'volume': 0}
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss, components = criterion(predictions, y_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
            optimizer.step()
            
            train_loss += loss.item()
            for key in components:
                train_components[key] += components[key]
        
        train_loss /= len(train_loader)
        for key in train_components:
            train_components[key] /= len(train_loader)
        train_losses.append(train_loss)
        
        # VAL
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
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        epoch_bar.set_postfix({
            'train': f'{train_loss:.4f}',
            'val': f'{val_loss:.4f}',
            'best': f'{best_val_loss:.4f}',
            'dir': f'{train_components["direction"]:.3f}'
        })
        
        if patience_counter >= Config.PATIENCE:
            print(f"\n⏹️ Early stopping at epoch {epoch+1}")
            break
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return train_losses, val_losses

# ================================
# 📊 EVALUACIÓN COMPLETA
# ================================
def evaluate_model(model, test_loader, scaler_out, device):
    print("\n" + "="*70)
    print("  📊 EVALUACIÓN OHLCV")
    print("="*70)
    
    model.eval()
    predictions, targets = [], []
    
    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader, desc="Evaluating"):
            X_batch = X_batch.to(device)
            pred = model(X_batch)
            predictions.extend(pred.cpu().numpy())
            targets.extend(y_batch.numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Desnormalizar
    predictions_denorm = scaler_out.inverse_transform(predictions)
    targets_denorm = scaler_out.inverse_transform(targets)
    
    # Métricas por componente
    components = ['Open', 'High', 'Low', 'Close', 'Volume']
    metrics = {}
    
    print("\n" + "="*70)
    for i, comp in enumerate(components):
        pred_comp = predictions_denorm[:, i]
        target_comp = targets_denorm[:, i]
        
        mae = mean_absolute_error(target_comp, pred_comp)
        rmse = np.sqrt(np.mean((target_comp - pred_comp)**2))
        
        # Direction accuracy (para OHLC)
        if i < 4:  # No calculamos dirección para volumen
            dir_true = np.sign(target_comp)
            dir_pred = np.sign(pred_comp)
            accuracy = accuracy_score(dir_true, dir_pred) * 100
            
            print(f"\n🎯 {comp}:")
            print(f"   Direction Accuracy: {accuracy:.2f}%")
            print(f"   MAE: {mae:.6f}")
            print(f"   RMSE: {rmse:.6f}")
            
            metrics[comp.lower()] = {
                'direction_accuracy': float(accuracy),
                'mae': float(mae),
                'rmse': float(rmse)
            }
        else:
            print(f"\n📊 {comp}:")
            print(f"   MAE: {mae:.6f}")
            print(f"   RMSE: {rmse:.6f}")
            
            metrics[comp.lower()] = {
                'mae': float(mae),
                'rmse': float(rmse)
            }
    
    # Métrica global (enfoque en Close)
    close_accuracy = metrics['close']['direction_accuracy']
    if close_accuracy > 55:
        status = "✅ BUENO" if close_accuracy > 60 else "⚠️ MARGINAL"
    else:
        status = "❌ POBRE"
    
    print(f"\n{'='*70}")
    print(f"📌 CLOSE Direction: {close_accuracy:.2f}% {status}")
    print(f"{'='*70}")
    
    return predictions_denorm, targets_denorm, metrics

def plot_results(train_losses, val_losses, metrics):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('LSTM OHLCV Predictor - Zero Leakage', 
                 fontsize=14, fontweight='bold')
    
    # Loss
    axes[0, 0].plot(train_losses, label='Train', linewidth=2, alpha=0.8)
    axes[0, 0].plot(val_losses, label='Val', linewidth=2, alpha=0.8)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # Direction Accuracy para OHLC
    ohlc_comps = ['open', 'high', 'low', 'close']
    accuracies = [metrics[c]['direction_accuracy'] for c in ohlc_comps]
    colors = ['green' if a > 60 else 'orange' if a > 55 else 'red' for a in accuracies]
    
    axes[0, 1].bar(ohlc_comps, accuracies, color=colors, alpha=0.7)
    axes[0, 1].set_title('Direction Accuracy (OHLC)')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].axhline(y=50, color='red', linestyle='--', alpha=0.3)
    axes[0, 1].axhline(y=55, color='orange', linestyle='--', alpha=0.3)
    axes[0, 1].axhline(y=60, color='green', linestyle='--', alpha=0.3)
    axes[0, 1].set_ylim([45, 70])
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # MAE por componente
    all_comps = ['open', 'high', 'low', 'close', 'volume']
    maes = [metrics[c]['mae'] for c in all_comps]
    
    axes[0, 2].bar(all_comps, maes, alpha=0.7, color='steelblue')
    axes[0, 2].set_title('MAE por Componente')
    axes[0, 2].set_ylabel('MAE')
    axes[0, 2].tick_params(axis='x', rotation=45)
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    
    # RMSE
    rmses = [metrics[c]['rmse'] for c in all_comps]
    axes[1, 0].bar(all_comps, rmses, alpha=0.7, color='coral')
    axes[1, 0].set_title('RMSE por Componente')
    axes[1, 0].set_ylabel('RMSE')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Close focus
    close_acc = metrics['close']['direction_accuracy']
    close_mae = metrics['close']['mae']
    color = 'green' if close_acc > 60 else 'orange' if close_acc > 55 else 'red'
    
    axes[1, 1].text(0.5, 0.7, f"{close_acc:.1f}%", 
                    ha='center', va='center', fontsize=40, 
                    color=color, weight='bold')
    axes[1, 1].text(0.5, 0.3, f"MAE: {close_mae:.6f}", 
                    ha='center', va='center', fontsize=12)
    axes[1, 1].set_title('CLOSE Direction Accuracy')
    axes[1, 1].axis('off')
    
    # Info
    info_text = (
        "✅ Zero Data Leakage\n"
        "✅ OHLCV completo\n"
        "✅ Loss direccional diferenciable\n"
        "✅ 12 features esenciales\n"
        f"✅ Target: Próxima vela"
    )
    axes[1, 2].text(0.1, 0.5, info_text, 
                    ha='left', va='center', fontsize=10,
                    family='monospace')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('lstm_ohlcv_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("📈 Plot saved: lstm_ohlcv_results.png")

# ================================
# 🚀 MAIN
# ================================
def main():
    try:
        print("\n" + "="*70)
        print("  🚀 LSTM OHLCV PREDICTOR - ZERO LEAKAGE")
        print("="*70)
        print("\n✅ Mejoras:")
        print("   1. Zero data leakage (validación temporal estricta)")
        print("   2. Predicción OHLCV completo (5 valores)")
        print("   3. Loss direccional diferenciable")
        print("   4. Features mínimas (12) y relevantes")
        print("   5. Arquitectura simple y enfocada")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n🖥️ Device: {device}")
        
        print("\n📥 Downloading EURUSD data...")
        ticker = yf.Ticker("EURUSD=X")
        df = ticker.history(period="2y", interval="1h")
        
        if len(df) == 0:
            raise ValueError("Failed to download data")
        
        df = df.reset_index()
        df.columns = [col.lower() for col in df.columns]
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        df = df[required_cols]
        
        print(f"✅ Downloaded: {len(df):,} candles")
        
        (X_train, y_train), (X_val, y_val), (X_test, y_test), \
        scalers, feature_cols, target_cols = prepare_dataset_zero_leakage(df)
        
        input_size = len(feature_cols)
        
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val), torch.FloatTensor(y_val)
        )
        test_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_test), torch.FloatTensor(y_test)
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
        
        model = LSTM_OHLCV(
            input_size=input_size,
            hidden_size=Config.HIDDEN_SIZE,
            num_layers=Config.NUM_LAYERS,
            dropout=Config.DROPOUT
        ).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n🧠 Model: {total_params:,} parameters")
        
        start_time = time.time()
        train_losses, val_losses = train_model(model, train_loader, val_loader, device)
        training_time = time.time() - start_time
        
        print(f"\n⏱️ Training time: {training_time/60:.1f} min")
        
        predictions, targets, metrics = evaluate_model(
            model, test_loader, scalers['output'], device
        )
        
        plot_results(train_losses, val_losses, metrics)
        
        # Guardar modelo
        model_dir = 'EURUSD_OHLCV_MODEL'
        os.makedirs(model_dir, exist_ok=True)
        
        print("\n💾 Guardando modelo...")
        torch.save(model.state_dict(), f'{model_dir}/lstm_ohlcv.pth')
        
        metadata = {
            'model_type': 'LSTM_OHLCV',
            'version': '1.0_zero_leakage',
            'output': 'OHLCV_complete',
            'fixes': [
                'zero_data_leakage',
                'temporal_validation',
                'differentiable_direction_loss',
                'minimal_features',
                'ohlcv_prediction'
            ],
            'input_size': int(input_size),
            'feature_cols': feature_cols,
            'target_cols': target_cols,
            'config': {
                'hidden_size': int(Config.HIDDEN_SIZE),
                'num_layers': int(Config.NUM_LAYERS),
                'seq_len': int(Config.SEQ_LEN),
                'direction_weight': float(Config.DIRECTION_WEIGHT)
            },
            'metrics_test': metrics,
            'training_time_min': float(training_time / 60),
            'total_params': int(total_params)
        }
        
        with open(f'{model_dir}/config_ohlcv.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        joblib.dump(scalers['input'], f'{model_dir}/scaler_input.pkl')
        joblib.dump(scalers['output'], f'{model_dir}/scaler_output.pkl')
        
        print("   ✅ Guardado completo")
        
        print("\n" + "="*70)
        print("  ✅ TRAINING COMPLETE - OHLCV PREDICTOR")
        print("="*70)
        
        print(f"\n📊 RESUMEN FINAL:")
        print(f"   • CLOSE Direction: {metrics['close']['direction_accuracy']:.2f}%")
        print(f"   • Avg OHLC Accuracy: {np.mean([metrics[c]['direction_accuracy'] for c in ['open','high','low','close']]):.2f}%")
        print(f"   • CLOSE MAE: {metrics['close']['mae']:.6f}")
        
        if metrics['close']['direction_accuracy'] > 55:
            print("\n💡 Nota sobre resultados:")
            print("   >55% en forex intrahorario con datos públicos es razonable")
            print("   >60% sería excelente (difícil de mantener OOS)")
            print("   Para mejorar: añadir microestructura, order flow, sentiment")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
