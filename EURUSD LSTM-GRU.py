"""
LSTM-GRU OPTIMIZADO PARA DIRECCIÓN (TRADING)
✅ Enfoque: Maximizar accuracy direccional (no R²)
✅ Loss: Penaliza errores de dirección más que magnitud
✅ Arquitectura: Simplificada para reducir overfitting
✅ Métrica: Direction accuracy > 70% = profitable trading
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
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
# 🎛️ CONFIGURACIÓN OPTIMIZADA
# ================================
class Config:
    # Arquitectura SIMPLIFICADA (menos overfitting)
    SEQ_LEN = 120
    ENCODER_HIDDEN = 256
    DECODER_HIDDEN = 256
    ENCODER_LAYERS = 3
    DECODER_LAYERS = 2
    DROPOUT = 0.2
    BIDIRECTIONAL_ENCODER = False
    
    # Entrenamiento
    BATCH_SIZE = 256
    EPOCHS = 200
    LEARNING_RATE = 0.0015
    WEIGHT_DECAY = 1e-3
    PATIENCE = 45
    MIN_DELTA = 1e-5
    GRAD_CLIP = 0.5
    
    # Loss weights - DIRECCIÓN > MAGNITUD
    DIRECTION_WEIGHT = 2.0      # ✅ Prioridad a dirección
    MAGNITUDE_WEIGHT = 0.5      # ✅ Magnitud secundaria
    CONSTRAINT_WEIGHT = 0.1
    
    # Warmup
    WARMUP_EPOCHS = 5
    
    # Datos
    TRAIN_SIZE = 0.70
    VAL_SIZE = 0.15
    TEST_SIZE = 0.15

# ================================
# 📊 FEATURES SIMPLIFICADAS
# ================================
def calculate_features(df):
    """Features esenciales para dirección"""
    df = df.copy()
    
    # Returns multi-timeframe (CLAVE para dirección)
    for period in [1, 2, 3, 5, 10, 20]:
        df[f'return_{period}'] = df['close'].pct_change(period)
        df[f'return_{period}_sign'] = np.sign(df[f'return_{period}'])
    
    # Momentum
    for period in [5, 10, 20]:
        df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)
        df[f'roc_{period}'] = df['close'].pct_change(period)
    
    # SMAs y cruces
    for period in [5, 10, 20, 50]:
        df[f'sma_{period}'] = df['close'].rolling(period).mean()
        df[f'close_sma_{period}'] = df['close'] / (df[f'sma_{period}'] + 1e-10)
        df[f'sma_{period}_slope'] = df[f'sma_{period}'].diff()
    
    # EMAs
    df['ema_9'] = df['close'].ewm(span=9).mean()
    df['ema_21'] = df['close'].ewm(span=21).mean()
    df['ema_cross'] = (df['ema_9'] > df['ema_21']).astype(int)
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
    df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
    
    # MACD
    df['macd'] = df['ema_9'] - df['ema_21']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_cross'] = (df['macd'] > df['macd_signal']).astype(int)
    
    # Bollinger
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    
    # Volatilidad
    for period in [5, 10, 20]:
        df[f'volatility_{period}'] = df['close'].pct_change().rolling(period).std()
    
    # Volume
    df['log_volume'] = np.log1p(df['volume'])
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-10)
    df['volume_trend'] = df['volume'].diff().rolling(5).mean()
    
    # OHLC patterns
    df['body'] = df['close'] - df['open']
    df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
    df['body_pct'] = df['body'] / (df['close'] + 1e-10)
    
    # Limpieza
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = np.clip(df[col], -1e6, 1e6)
    
    return df

# ================================
# 🧠 ARQUITECTURA SIMPLIFICADA
# ================================
class SimpleAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, encoder_outputs):
        attn_weights = torch.softmax(self.attention(encoder_outputs), dim=1)
        context = torch.sum(attn_weights * encoder_outputs, dim=1)
        return context, attn_weights

class SimpleLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, 
                 dropout=0.4, bidirectional=True):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        if bidirectional:
            self.projection = nn.Linear(hidden_size * 2, hidden_size)
    
    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        
        if self.bidirectional:
            outputs = self.projection(outputs)
            batch_size = x.size(0)
            hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_size)
            hidden = hidden[:, 0, :, :] + hidden[:, 1, :, :]
            cell = cell.view(self.num_layers, 2, batch_size, self.hidden_size)
            cell = cell[:, 0, :, :] + cell[:, 1, :, :]
        
        return outputs, (hidden, cell)

class SimpleGRUDecoder(nn.Module):
    def __init__(self, hidden_size=128, num_layers=1, dropout=0.4):
        super().__init__()
        
        self.gru = nn.GRU(
            input_size=hidden_size + 1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = SimpleAttention(hidden_size)
        
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, prev_output, hidden, encoder_outputs):
        context, attn_weights = self.attention(encoder_outputs)
        gru_input = torch.cat([context, prev_output], dim=-1).unsqueeze(1)
        gru_out, new_hidden = self.gru(gru_input, hidden)
        combined = torch.cat([gru_out.squeeze(1), context], dim=-1)
        prediction = self.output_net(combined)
        return prediction, new_hidden, attn_weights

class DirectionOptimizedModel(nn.Module):
    def __init__(self, input_size, encoder_hidden=128, decoder_hidden=128,
                 encoder_layers=2, decoder_layers=1, dropout=0.4):
        super().__init__()
        
        self.encoder = SimpleLSTMEncoder(
            input_size=input_size,
            hidden_size=encoder_hidden,
            num_layers=encoder_layers,
            dropout=dropout,
            bidirectional=True
        )
        
        self.decoder = SimpleGRUDecoder(
            hidden_size=decoder_hidden,
            num_layers=decoder_layers,
            dropout=dropout
        )
        
        self.decoder_layers = decoder_layers
        self.output_activation = nn.Tanh()
    
    def forward(self, x, target=None, teacher_forcing_ratio=0.0):
        batch_size = x.size(0)
        
        encoder_outputs, (encoder_hidden, _) = self.encoder(x)
        decoder_hidden = encoder_hidden[-self.decoder_layers:]
        
        outputs = []
        prev_output = torch.zeros(batch_size, 1).to(x.device)
        
        for t in range(5):
            pred, decoder_hidden, _ = self.decoder(prev_output, decoder_hidden, encoder_outputs)
            outputs.append(pred)
            prev_output = pred.detach()
        
        outputs = torch.cat(outputs, dim=-1)
        outputs = self.output_activation(outputs) * 0.08
        
        return outputs

# ================================
# 🎯 LOSS OPTIMIZADO PARA DIRECCIÓN
# ================================
class DirectionOptimizedLoss(nn.Module):
    def __init__(self, direction_weight=2.0, magnitude_weight=0.5, constraint_weight=0.1):
        super().__init__()
        self.direction_weight = direction_weight
        self.magnitude_weight = magnitude_weight
        self.constraint_weight = constraint_weight
        self.huber = nn.HuberLoss(delta=0.5)
    
    def forward(self, predictions, targets):
        # Separar precio y volumen
        pred_price = predictions[:, :4]
        pred_volume = predictions[:, 4:]
        target_price = targets[:, :4]
        target_volume = targets[:, 4:]
        
        # 1. LOSS DE DIRECCIÓN (prioridad)
        pred_direction = torch.sign(pred_price)
        target_direction = torch.sign(target_price)
        
        # Penaliza MÁS cuando la dirección es incorrecta
        direction_correct = (pred_direction == target_direction).float()
        direction_loss = 1.0 - direction_correct.mean()
        
        # 2. LOSS DE MAGNITUD (secundario)
        magnitude_loss = self.huber(pred_price, target_price)
        volume_loss = self.huber(pred_volume, target_volume)
        
        # 3. CONSTRAINTS OHLC
        delta_high = predictions[:, 1]
        delta_low = predictions[:, 2]
        delta_close = predictions[:, 3]
        
        high_low_violation = torch.clamp(delta_low - delta_high, min=0)
        constraint_loss = high_low_violation.mean()
        
        # TOTAL con pesos ajustados
        total_loss = (
            self.direction_weight * direction_loss +
            self.magnitude_weight * magnitude_loss +
            self.magnitude_weight * 0.3 * volume_loss +
            self.constraint_weight * constraint_loss
        )
        
        return total_loss, {
            'direction': direction_loss.item(),
            'magnitude': magnitude_loss.item(),
            'volume': volume_loss.item(),
            'constraint': constraint_loss.item()
        }

# ================================
# 📦 PREPARACIÓN DE DATOS
# ================================
def prepare_dataset(df):
    print("\n" + "="*70)
    print("  🔧 PREPARACIÓN DE DATOS")
    print("="*70)
    
    df = calculate_features(df)
    
    # Targets
    df['delta_open'] = (df['open'].shift(-1) - df['close']) / (df['close'] + 1e-10)
    df['delta_high'] = (df['high'].shift(-1) - df['close']) / (df['close'] + 1e-10)
    df['delta_low'] = (df['low'].shift(-1) - df['close']) / (df['close'] + 1e-10)
    df['delta_close'] = (df['close'].shift(-1) - df['close']) / (df['close'] + 1e-10)
    df['delta_volume'] = np.log1p(df['volume'].shift(-1)) - np.log1p(df['volume'])
    
    df = df.dropna()
    print(f"📊 Datos limpios: {len(df):,} velas")
    
    all_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ['delta_open', 'delta_high', 'delta_low', 'delta_close', 'delta_volume']
    feature_cols = [c for c in all_cols if c not in exclude]
    target_cols = ['delta_open', 'delta_high', 'delta_low', 'delta_close', 'delta_volume']
    
    print(f"🎯 {len(feature_cols)} features, {len(target_cols)} targets")
    
    # Split
    train_end = int(len(df) * Config.TRAIN_SIZE)
    val_end = int(len(df) * (Config.TRAIN_SIZE + Config.VAL_SIZE))
    
    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()
    
    print(f"📊 Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
    
    # Scalers
    scaler_in = RobustScaler(quantile_range=(10, 90))
    scaler_out_price = RobustScaler(quantile_range=(10, 90))
    scaler_out_volume = RobustScaler(quantile_range=(10, 90))
    
    X_train = scaler_in.fit_transform(df_train[feature_cols])
    X_val = scaler_in.transform(df_val[feature_cols])
    X_test = scaler_in.transform(df_test[feature_cols])
    
    y_train_price = scaler_out_price.fit_transform(df_train[target_cols[:4]])
    y_train_volume = scaler_out_volume.fit_transform(df_train[[target_cols[4]]])
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
    
    print(f"🔄 Creando secuencias (seq_len={Config.SEQ_LEN})...")
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, Config.SEQ_LEN)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, Config.SEQ_LEN)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, Config.SEQ_LEN)
    
    print(f"✅ Train: {X_train_seq.shape}, Val: {X_val_seq.shape}, Test: {X_test_seq.shape}")
    
    scalers = {
        'input': scaler_in,
        'output_price': scaler_out_price,
        'output_volume': scaler_out_volume
    }
    
    return (X_train_seq, y_train_seq), (X_val_seq, y_val_seq), (X_test_seq, y_test_seq), \
           scalers, feature_cols, target_cols

# ================================
# 🏋️ ENTRENAMIENTO
# ================================
def train_model(model, train_loader, val_loader, device):
    print("\n" + "="*70)
    print("  🏋️ ENTRENAMIENTO - OPTIMIZADO PARA DIRECCIÓN")
    print("="*70)
    
    criterion = DirectionOptimizedLoss(
        direction_weight=Config.DIRECTION_WEIGHT,
        magnitude_weight=Config.MAGNITUDE_WEIGHT,
        constraint_weight=Config.CONSTRAINT_WEIGHT
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-6
    )
    
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=Config.WARMUP_EPOCHS
    )
    
    start_epoch = 0
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print(f"⚙️ Config: LR={Config.LEARNING_RATE}, Batch={Config.BATCH_SIZE}")
    print(f"🎯 Optimizando: DIRECCIÓN (weight={Config.DIRECTION_WEIGHT}) > Magnitud")
    print(f"⏰ Patience: {Config.PATIENCE} epochs")
    
    epoch_bar = tqdm(range(Config.EPOCHS), desc="Training")
    
    for epoch in epoch_bar:
        # TRAIN
        model.train()
        train_loss = 0
        train_components = {'direction': 0, 'magnitude': 0, 'volume': 0, 'constraint': 0}
        
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
        
        # Schedulers
        if epoch < Config.WARMUP_EPOCHS:
            warmup_scheduler.step()
        else:
            scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < (best_val_loss - Config.MIN_DELTA):
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_bar.set_postfix({
            'train': f'{train_loss:.4f}',
            'val': f'{val_loss:.4f}',
            'best': f'{best_val_loss:.4f}',
            'patience': f'{patience_counter}/{Config.PATIENCE}',
            'lr': f'{current_lr:.6f}',
            'dir_loss': f'{train_components["direction"]:.3f}'
        })
        
        if (epoch + 1) % 20 == 0:
            print(f"\n📊 Epoch {epoch+1}/{Config.EPOCHS}:")
            print(f"   Loss: Train={train_loss:.4f}, Val={val_loss:.4f}, Best={best_val_loss:.4f}")
            print(f"   Direction loss: {train_components['direction']:.4f}")
            print(f"   Patience: {patience_counter}/{Config.PATIENCE}")
        
        if patience_counter >= Config.PATIENCE:
            print(f"\n⏹️ Early stopping at epoch {epoch+1}")
            break
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return train_losses, val_losses

# ================================
# 📊 EVALUACIÓN
# ================================
def evaluate_model(model, test_loader, scalers, target_cols, device):
    print("\n" + "="*70)
    print("  📊 EVALUACIÓN - FOCO EN DIRECCIÓN")
    print("="*70)
    
    model.eval()
    predictions, targets = [], []
    
    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader, desc="Evaluating"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            predictions.extend(pred.cpu().numpy())
            targets.extend(y_batch.cpu().numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
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
        accuracy = accuracy_score(direction_true, direction_pred) * 100
        
        metrics[col] = {
            'MAE': float(mae),
            'RMSE': float(rmse),
            'R2': float(r2),
            'Direction_Accuracy': float(accuracy)
        }
        
        emoji = "🎯" if accuracy > 60 else "⚠️"
        print(f"{emoji} {col}: Acc={accuracy:.2f}% | MAE={mae:.6f}, R²={r2:.4f}")
    
    return predictions_denorm, targets_denorm, metrics

def plot_results(train_losses, val_losses, metrics):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Direction-Optimized LSTM-GRU', fontsize=14, fontweight='bold')
    
    # Loss
    axes[0].plot(train_losses, label='Train', linewidth=2, alpha=0.8)
    axes[0].plot(val_losses, label='Val', linewidth=2, alpha=0.8)
    axes[0].set_title('Training Loss (Direction-Focused)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # Direction Accuracy
    names = list(metrics.keys())
    acc_scores = [metrics[n]['Direction_Accuracy'] for n in names]
    colors = ['green' if acc > 60 else 'orange' if acc > 55 else 'red' for acc in acc_scores]
    axes[1].bar(names, acc_scores, color=colors, alpha=0.7)
    axes[1].set_title('Direction Accuracy (Trading Metric)')
    axes[1].axhline(y=50, color='red', linestyle='--', label='Random', alpha=0.5)
    axes[1].axhline(y=60, color='orange', linestyle='--', label='Good', alpha=0.5)
    axes[1].axhline(y=70, color='green', linestyle='--', label='Excellent', alpha=0.5)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].legend()
    
    # R² (informativo)
    r2_scores = [metrics[n]['R2'] for n in names]
    colors = ['green' if r2 > 0.1 else 'orange' if r2 > 0 else 'red' for r2 in r2_scores]
    axes[2].bar(names, r2_scores, color=colors, alpha=0.7)
    axes[2].set_title('R² Scores (Reference)')
    axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('direction_optimized_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("📈 Plot saved: direction_optimized_results.png")

# ================================
# 🚀 MAIN
# ================================
def main():
    try:
        print("\n" + "="*70)
        print("  🚀 LSTM-GRU - DIRECTION OPTIMIZED")
        print("="*70)
        print("\n🎯 Objetivo: Maximizar Direction Accuracy (>70% = profitable)")
        print("📊 Métrica secundaria: R² (informativo)")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n🖥️ Device: {device}")
        
        print("\n📥 Downloading EURUSD=X data...")
        ticker = yf.Ticker("EURUSD=X")
        df = ticker.history(period="2y", interval="1h")
        
        if len(df) == 0:
            raise ValueError("Failed to download data")
        
        df = df.reset_index()
        df.columns = [col.lower() for col in df.columns]
        
        rename_dict = {}
        for col in df.columns:
            if 'date' in col or 'time' in col:
                rename_dict[col] = 'time'
        df.rename(columns=rename_dict, inplace=True)
        
        required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        df = df[required_cols]
        
        print(f"✅ Downloaded: {len(df):,} candles")
        
        (X_train, y_train), (X_val, y_val), (X_test, y_test), \
        scalers, feature_cols, target_cols = prepare_dataset(df)
        
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
        
        model = DirectionOptimizedModel(
            input_size=input_size,
            encoder_hidden=Config.ENCODER_HIDDEN,
            decoder_hidden=Config.DECODER_HIDDEN,
            encoder_layers=Config.ENCODER_LAYERS,
            decoder_layers=Config.DECODER_LAYERS,
            dropout=Config.DROPOUT
        ).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n🧠 Model: {total_params:,} parameters")
        
        start_time = time.time()
        train_losses, val_losses = train_model(model, train_loader, val_loader, device)
        training_time = time.time() - start_time
        
        print(f"\n⏱️ Training time: {training_time/60:.1f} min")
        
        predictions, targets, metrics = evaluate_model(
            model, test_loader, scalers, target_cols, device
        )
        
        print("\n" + "="*70)
        print("  📈 FINAL RESULTS")
        print("="*70)
        
        price_metrics = {k: v for k, v in metrics.items() if 'volume' not in k}
        avg_acc = np.mean([m['Direction_Accuracy'] for m in price_metrics.values()])
        avg_r2 = np.mean([m['R2'] for m in price_metrics.values()])
        
        print(f"\n🎯 Price (OHLC) Direction Accuracy:")
        print(f"   Average: {avg_acc:.2f}%")
        if avg_acc > 70:
            print(f"   Status: ✅ EXCELLENT for trading!")
        elif avg_acc > 60:
            print(f"   Status: ✅ GOOD for trading")
        elif avg_acc > 55:
            print(f"   Status: ⚠️ MARGINAL - needs improvement")
        else:
            print(f"   Status: ❌ POOR - not profitable")
        
        print(f"\n📊 Average R²: {avg_r2:.4f} (informativo)")
        
        plot_results(train_losses, val_losses, metrics)
        
        # Guardar modelo
        model_dir = 'EURUSD_MODELS'
        os.makedirs(model_dir, exist_ok=True)
        
        print("\n💾 Guardando modelo...")
        torch.save(model.state_dict(), f'{model_dir}/direction_optimized.pth')
        
        from datetime import datetime
        
        def to_native(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: to_native(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [to_native(i) for i in obj]
            return obj
        
        metadata = {
            'model_type': 'DirectionOptimizedModel',
            'version': '3.0',
            'optimization_target': 'direction_accuracy',
            'input_size': int(input_size),
            'feature_cols': feature_cols,
            'target_cols': target_cols,
            'config': {
                'encoder_hidden': int(Config.ENCODER_HIDDEN),
                'decoder_hidden': int(Config.DECODER_HIDDEN),
                'encoder_layers': int(Config.ENCODER_LAYERS),
                'decoder_layers': int(Config.DECODER_LAYERS),
                'direction_weight': float(Config.DIRECTION_WEIGHT),
                'magnitude_weight': float(Config.MAGNITUDE_WEIGHT)
            },
            'metrics_test': to_native(metrics),
            'avg_direction_accuracy': float(avg_acc),
            'avg_r2': float(avg_r2),
            'training_time_min': float(training_time / 60),
            'total_params': int(total_params),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(f'{model_dir}/config_direction_optimized.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        joblib.dump(scalers['input'], f'{model_dir}/scaler_input_dir.pkl')
        joblib.dump(scalers['output_price'], f'{model_dir}/scaler_output_price_dir.pkl')
        joblib.dump(scalers['output_volume'], f'{model_dir}/scaler_output_volume_dir.pkl')
        
        print("   ✅ Todos los archivos guardados")
        
        print("\n" + "="*70)
        print("  ✅ DIRECTION-OPTIMIZED TRAINING COMPLETE")
        print("="*70)
        print(f"\n🎯 Direction Accuracy: {avg_acc:.2f}%")
        print(f"📊 R²: {avg_r2:.4f} (secundario)")
        print("\n💡 Recuerda: Para trading, direction accuracy > 60% = profitable\n")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
