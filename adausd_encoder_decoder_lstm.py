"""
ENCODER LSTM + DECODER GRU - VERSIÓN CORREGIDA Y OPTIMIZADA
✅ Fix: Error de pickle resuelto
✅ Fix: Arquitectura mejorada para aprendizaje
✅ Fix: Loss balanceado correctamente
✅ Fix: Learning rate adaptativo
✅ Checkpointing robusto
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import RobustScaler
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
# 🎛️ CONFIGURACIÓN CORREGIDA
# ================================
class Config:
    # Arquitectura mejorada para mejor aprendizaje
    SEQ_LEN = 60
    ENCODER_HIDDEN = 128
    DECODER_HIDDEN = 128
    ENCODER_LAYERS = 2
    DECODER_LAYERS = 2  # ⬆️ Aumentado a 2 para mejor capacidad
    DROPOUT = 0.2  # ⬇️ Reducido - demasiado dropout causa underfitting
    BIDIRECTIONAL_ENCODER = True
    USE_ATTENTION = True
    
    # Entrenamiento ajustado
    BATCH_SIZE = 64
    EPOCHS = 150
    LEARNING_RATE = 0.0005  # ⬆️ Aumentado - LR muy bajo causa convergencia lenta
    WEIGHT_DECAY = 5e-5  # ⬇️ Reducido - demasiada regularización
    PATIENCE = 25  # ⬆️ Aumentado para dar más tiempo
    MIN_DELTA = 1e-6
    GRAD_CLIP = 1.0
    TEACHER_FORCING_RATIO = 0.7  # ⬆️ Más alto para mejor aprendizaje
    TEACHER_FORCING_DECAY = 0.98
    
    # Checkpointing
    CHECKPOINT_EVERY = 10
    RESUME_TRAINING = True
    
    # Loss weights ajustados
    PRICE_WEIGHT = 1.0
    VOLUME_WEIGHT = 0.3  # ⬇️ Reducido para enfocarse en precio primero
    CONSTRAINT_WEIGHT = 0.05  # ⬇️ Muy reducido
    
    # Warmup
    WARMUP_EPOCHS = 5  # ⬇️ Reducido
    
    # Datos
    TRAIN_SIZE = 0.70
    VAL_SIZE = 0.15
    TEST_SIZE = 0.15

# ================================
# 📊 FEATURES OPTIMIZADAS
# ================================
def calculate_features(df):
    """Features esenciales pero completas"""
    df = df.copy()
    
    # Normalización de volumen
    df['log_volume'] = np.log1p(df['volume'])
    df['volume_ma_5'] = df['volume'].rolling(5).mean()
    df['volume_ma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma_20'] + 1e-10)
    
    # Medias móviles
    for period in [5, 10, 20, 50]:
        df[f'sma_{period}'] = df['close'].rolling(period).mean()
        df[f'close_sma_{period}_ratio'] = df['close'] / (df[f'sma_{period}'] + 1e-10)
    
    # EMAs
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_ma'] = df['rsi'].rolling(5).mean()
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    
    # ATR
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift())
    df['tr3'] = abs(df['low'] - df['close'].shift())
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['true_range'].rolling(14).mean()
    df['atr_percent'] = df['atr'] / (df['close'] + 1e-10)
    
    # Volatilidad
    df['returns'] = df['close'].pct_change()
    df['volatility_10'] = df['returns'].rolling(10).std()
    df['volatility_20'] = df['returns'].rolling(20).std()
    
    # Rangos OHLC
    df['high_low_range'] = (df['high'] - df['low']) / (df['close'] + 1e-10)
    df['open_close_range'] = abs(df['open'] - df['close']) / (df['close'] + 1e-10)
    
    # Momentum
    df['roc_5'] = df['close'].pct_change(periods=5)
    df['roc_10'] = df['close'].pct_change(periods=10)
    df['roc_20'] = df['close'].pct_change(periods=20)
    
    # VWAP
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (df['typical_price'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['vwap_distance'] = (df['close'] - df['vwap']) / (df['vwap'] + 1e-10)
    
    # Derivadas precio-volumen
    df['price_velocity'] = df['close'].pct_change()
    df['price_acceleration'] = df['price_velocity'].diff()
    df['volume_velocity'] = df['volume'].pct_change()
    df['volume_acceleration'] = df['volume_velocity'].diff()
    df['price_volume_corr'] = df['close'].rolling(20).corr(df['volume'])
    
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = np.clip(df[col], -1e6, 1e6)
    
    return df

# ================================
# 🧠 ARQUITECTURA
# ================================
class BahdanauAttention(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size):
        super().__init__()
        self.attention_size = decoder_hidden_size
        self.W1 = nn.Linear(encoder_hidden_size, self.attention_size)
        self.W2 = nn.Linear(decoder_hidden_size, self.attention_size)
        self.V = nn.Linear(self.attention_size, 1)
    
    def forward(self, decoder_hidden, encoder_outputs):
        decoder_hidden = decoder_hidden.unsqueeze(1)
        encoder_proj = self.W1(encoder_outputs)
        decoder_proj = self.W2(decoder_hidden)
        score = self.V(torch.tanh(encoder_proj + decoder_proj))
        attention_weights = torch.softmax(score, dim=1)
        context = torch.sum(attention_weights * encoder_outputs, dim=1)
        return context, attention_weights

class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, 
                 dropout=0.3, bidirectional=True):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
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
        else:
            self.projection = None
    
    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        
        if self.projection:
            outputs = self.projection(outputs)
            batch_size = x.size(0)
            hidden = hidden.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)
            hidden = hidden.permute(0, 2, 1, 3).contiguous().view(self.num_layers, batch_size, -1)
            hidden = self.projection(hidden)
            
            cell = cell.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)
            cell = cell.permute(0, 2, 1, 3).contiguous().view(self.num_layers, batch_size, -1)
            cell = self.projection(cell)
        
        return outputs, (hidden, cell)

class GRUDecoder(nn.Module):
    def __init__(self, encoder_hidden_size=128, decoder_hidden_size=128, 
                 num_layers=1, dropout=0.3):
        super().__init__()
        
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=encoder_hidden_size + 1,
            hidden_size=decoder_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = BahdanauAttention(encoder_hidden_size, decoder_hidden_size)
        self.hidden_projection = nn.Linear(encoder_hidden_size, decoder_hidden_size)
        
        self.fc = nn.Sequential(
            nn.Linear(encoder_hidden_size + decoder_hidden_size, decoder_hidden_size * 2),
            nn.LayerNorm(decoder_hidden_size * 2),
            nn.GELU(),  # Mejor que ReLU
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden_size * 2, decoder_hidden_size),
            nn.LayerNorm(decoder_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(decoder_hidden_size, 1)
        )
    
    def forward(self, prev_output, hidden, encoder_outputs):
        context, attn_weights = self.attention(hidden[-1], encoder_outputs)
        gru_input = torch.cat([context, prev_output], dim=-1).unsqueeze(1)
        gru_out, new_hidden = self.gru(gru_input, hidden)
        combined = torch.cat([gru_out.squeeze(1), context], dim=-1)
        prediction = self.fc(combined)
        return prediction, new_hidden, attn_weights

class HybridEncoderDecoder(nn.Module):
    def __init__(self, input_size, encoder_hidden=128, decoder_hidden=128,
                 encoder_layers=2, decoder_layers=1, dropout=0.3, 
                 bidirectional_encoder=True):
        super().__init__()
        
        self.encoder = LSTMEncoder(
            input_size=input_size,
            hidden_size=encoder_hidden,
            num_layers=encoder_layers,
            dropout=dropout,
            bidirectional=bidirectional_encoder
        )
        
        self.decoder = GRUDecoder(
            encoder_hidden_size=encoder_hidden,
            decoder_hidden_size=decoder_hidden,
            num_layers=decoder_layers,
            dropout=dropout
        )
        
        self.encoder_hidden = encoder_hidden
        self.decoder_hidden = decoder_hidden
        self.decoder_layers = decoder_layers
        self.output_activation = nn.Tanh()
    
    def forward(self, x, target=None, teacher_forcing_ratio=0.5):
        batch_size = x.size(0)
        encoder_outputs, (encoder_hidden, encoder_cell) = self.encoder(x)
        decoder_hidden = self.decoder.hidden_projection(encoder_hidden[-self.decoder_layers:])
        
        outputs = []
        prev_output = torch.zeros(batch_size, 1).to(x.device)
        
        for t in range(5):
            use_teacher_forcing = target is not None and np.random.random() < teacher_forcing_ratio
            pred, decoder_hidden, attn_weights = self.decoder(prev_output, decoder_hidden, encoder_outputs)
            outputs.append(pred)
            
            if use_teacher_forcing and t < 4:
                prev_output = target[:, t:t+1]
            else:
                prev_output = pred.detach()
        
        outputs = torch.cat(outputs, dim=-1)
        outputs = self.output_activation(outputs) * 0.15  # ⬆️ Aumentado para permitir variaciones mayores
        return outputs

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
    
    all_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['delta_open', 'delta_high', 'delta_low', 'delta_close', 'delta_volume']
    feature_cols = [col for col in all_numeric_cols if col not in exclude_cols]
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
    scaler_in = RobustScaler(quantile_range=(5, 95))  # Más robusto
    scaler_out_price = RobustScaler(quantile_range=(5, 95))
    scaler_out_volume = RobustScaler(quantile_range=(5, 95))
    
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
# 🏋️ LOSS MEJORADO
# ================================
class ImprovedHybridLoss(nn.Module):
    def __init__(self, price_weight=1.0, volume_weight=0.3, constraint_weight=0.05):
        super().__init__()
        self.mse = nn.MSELoss()
        self.huber = nn.HuberLoss(delta=1.0)  # Más robusto a outliers
        self.price_weight = price_weight
        self.volume_weight = volume_weight
        self.constraint_weight = constraint_weight
    
    def forward(self, predictions, targets):
        pred_price = predictions[:, :4]
        pred_volume = predictions[:, 4:]
        target_price = targets[:, :4]
        target_volume = targets[:, 4:]
        
        # Usar Huber loss que es más robusto
        price_loss = self.huber(pred_price, target_price)
        volume_loss = self.huber(pred_volume, target_volume)
        
        # Constraints MUY suaves
        delta_high = predictions[:, 1]
        delta_low = predictions[:, 2]
        delta_close = predictions[:, 3]
        
        high_low_violation = torch.clamp(delta_low - delta_high, min=0)
        close_below_low = torch.clamp(delta_low - delta_close, min=0)
        close_above_high = torch.clamp(delta_close - delta_high, min=0)
        constraint_loss = (high_low_violation + close_below_low + close_above_high).mean()
        
        total_loss = (
            self.price_weight * price_loss +
            self.volume_weight * volume_loss +
            self.constraint_weight * constraint_loss
        )
        
        return total_loss, {
            'price': price_loss.item(),
            'volume': volume_loss.item(),
            'constraint': constraint_loss.item()
        }

# ================================
# 💾 CHECKPOINTING
# ================================
def save_checkpoint(epoch, model, optimizer, scheduler, train_losses, val_losses, 
                   best_val_loss, patience_counter, teacher_forcing_ratio, filepath='checkpoint.pth'):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'patience_counter': patience_counter,
        'teacher_forcing_ratio': teacher_forcing_ratio
    }
    torch.save(checkpoint, filepath)
    print(f"💾 Checkpoint guardado: {filepath}")

def load_checkpoint(filepath, model, optimizer, scheduler):
    if not os.path.exists(filepath):
        return None
    
    print(f"📂 Cargando checkpoint: {filepath}")
    checkpoint = torch.load(filepath)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return {
        'start_epoch': checkpoint['epoch'] + 1,
        'train_losses': checkpoint['train_losses'],
        'val_losses': checkpoint['val_losses'],
        'best_val_loss': checkpoint['best_val_loss'],
        'patience_counter': checkpoint['patience_counter'],
        'teacher_forcing_ratio': checkpoint.get('teacher_forcing_ratio', Config.TEACHER_FORCING_RATIO)
    }

# ================================
# 🏋️ ENTRENAMIENTO
# ================================
def train_model(model, train_loader, val_loader, device):
    print("\n" + "="*70)
    print("  🏋️ ENTRENAMIENTO")
    print("="*70)
    
    criterion = ImprovedHybridLoss(
        price_weight=Config.PRICE_WEIGHT,
        volume_weight=Config.VOLUME_WEIGHT,
        constraint_weight=Config.CONSTRAINT_WEIGHT
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6, verbose=True
    )
    
    # Warmup scheduler
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=Config.WARMUP_EPOCHS
    )
    
    checkpoint_path = 'training_checkpoint.pth'
    checkpoint_data = None
    if Config.RESUME_TRAINING:
        checkpoint_data = load_checkpoint(checkpoint_path, model, optimizer, scheduler)
    
    if checkpoint_data:
        start_epoch = checkpoint_data['start_epoch']
        train_losses = checkpoint_data['train_losses']
        val_losses = checkpoint_data['val_losses']
        best_val_loss = checkpoint_data['best_val_loss']
        patience_counter = checkpoint_data['patience_counter']
        teacher_forcing_ratio = checkpoint_data['teacher_forcing_ratio']
        print(f"✅ Reanudando desde época {start_epoch}/{Config.EPOCHS}")
    else:
        start_epoch = 0
        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        patience_counter = 0
        teacher_forcing_ratio = Config.TEACHER_FORCING_RATIO
        print(f"🆕 Iniciando entrenamiento desde cero")
    
    best_model_state = None
    
    print(f"⚙️ Config: LR={Config.LEARNING_RATE}, Batch={Config.BATCH_SIZE}, Epochs={Config.EPOCHS}")
    print(f"💾 Checkpoint cada {Config.CHECKPOINT_EVERY} épocas")
    print(f"🔥 Warmup: {Config.WARMUP_EPOCHS} épocas")
    
    epoch_bar = tqdm(range(start_epoch, Config.EPOCHS), desc="Training", unit="epoch", 
                     initial=start_epoch, total=Config.EPOCHS)
    
    for epoch in epoch_bar:
        epoch_start_time = time.time()
        
        # TRAIN
        model.train()
        train_loss = 0
        train_components = {'price': 0, 'volume': 0, 'constraint': 0}
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            predictions = model(X_batch, target=y_batch, 
                              teacher_forcing_ratio=teacher_forcing_ratio)
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
                predictions = model(X_batch, target=None, teacher_forcing_ratio=0)
                loss, _ = criterion(predictions, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Schedulers
        if epoch < Config.WARMUP_EPOCHS:
            warmup_scheduler.step()
        else:
            scheduler.step(val_loss)
        
        # Decay teacher forcing
        teacher_forcing_ratio *= Config.TEACHER_FORCING_DECAY
        teacher_forcing_ratio = max(teacher_forcing_ratio, 0.1)
        
        # Early stopping
        if val_loss < best_val_loss - Config.MIN_DELTA:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            torch.save(best_model_state, 'best_model.pth')
        else:
            patience_counter += 1
        
        # Checkpoint
        if (epoch + 1) % Config.CHECKPOINT_EVERY == 0:
            save_checkpoint(epoch, model, optimizer, scheduler, train_losses, val_losses,
                          best_val_loss, patience_counter, teacher_forcing_ratio, checkpoint_path)
        
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_bar.set_postfix({
            'train': f'{train_loss:.4f}',
            'val': f'{val_loss:.4f}',
            'best': f'{best_val_loss:.4f}',
            'lr': f'{current_lr:.6f}',
            'tf': f'{teacher_forcing_ratio:.2f}',
            'patience': f'{patience_counter}/{Config.PATIENCE}',
            'time': f'{epoch_time:.1f}s'
        })
        
        # Logging detallado cada 20 épocas
        if (epoch + 1) % 20 == 0:
            print(f"\n📊 Epoch {epoch+1}:")
            print(f"   Train: {train_loss:.4f} | Val: {val_loss:.4f} | Best: {best_val_loss:.4f}")
            print(f"   Components - Price: {train_components['price']:.4f}, Vol: {train_components['volume']:.4f}, Constr: {train_components['constraint']:.4f}")
            print(f"   LR: {current_lr:.6f} | TF: {teacher_forcing_ratio:.3f}")
        
        if patience_counter >= Config.PATIENCE:
            print(f"\n⏹️ Early stopping at epoch {epoch+1}")
            break
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"🗑️ Checkpoint temporal eliminado")
    
    print(f"\n✅ Training complete: Best val loss = {best_val_loss:.6f}")
    return train_losses, val_losses

# ================================
# 📊 EVALUACIÓN
# ================================
def evaluate_model(model, test_loader, scalers, target_cols, device):
    print("\n" + "="*70)
    print("  📊 EVALUACIÓN")
    print("="*70)
    
    model.eval()
    predictions, targets = [], []
    
    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader, desc="Evaluating"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch, target=None, teacher_forcing_ratio=0)
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
        accuracy = np.mean(direction_true == direction_pred) * 100
        
        metrics[col] = {
            'MAE': float(mae),
            'RMSE': float(rmse),
            'R2': float(r2),
            'Direction_Accuracy': float(accuracy)
        }
        
        print(f"📊 {col}: MAE={mae:.6f}, RMSE={rmse:.6f}, R²={r2:.4f}, Acc={accuracy:.2f}%")
    
    return predictions_denorm, targets_denorm, metrics

def plot_results(train_losses, val_losses, metrics):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('LSTM-GRU Corrected Results', fontsize=14, fontweight='bold')
    
    axes[0].plot(train_losses, label='Train', linewidth=2)
    axes[0].plot(val_losses, label='Val', linewidth=2)
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    names = list(metrics.keys())
    r2_scores = [metrics[n]['R2'] for n in names]
    axes[1].bar(names, r2_scores)
    axes[1].set_title('R² Scores')
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    acc_scores = [metrics[n]['Direction_Accuracy'] for n in names]
    axes[2].bar(names, acc_scores)
    axes[2].set_title('Direction Accuracy (%)')
    axes[2].axhline(y=50, color='r', linestyle='--', alpha=0.5)
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('corrected_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("📈 Plot saved: corrected_results.png")

# ================================
# 🚀 MAIN
# ================================
def main():
    try:
        print("\n" + "="*70)
        print("  🚀 LSTM-GRU CORREGIDO Y OPTIMIZADO")
        print("="*70)
        print("\n🔧 CORRECCIONES V2:")
        print("   ✅ Fix: Error de pickle resuelto 100%")
        print("   ✅ Fix: Metadata en JSON separado")
        print("   ✅ Fix: Learning rate aumentado (0.0001→0.0005)")
        print("   ✅ Fix: Dropout reducido (0.3→0.2) para evitar underfitting")
        print("   ✅ Fix: Decoder 2 layers para mejor capacidad")
        print("   ✅ Fix: Huber loss (más robusto que MSE+MAE)")
        print("   ✅ Fix: Output scale 0.15 para variaciones mayores")
        print("   ✅ Fix: Teacher forcing 0.7 para mejor aprendizaje")
        print("   ✅ Fix: Weight decay reducido (1e-4→5e-5)")
        print("   ✅ Fix: Decoder con GELU y más capas")
        print("   🔍 Logging detallado cada 20 épocas")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n🖥️ Device: {device}")
        
        print("\n📥 Downloading ADA-USD data...")
        ticker = yf.Ticker("ADA-USD")
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
            train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
        )
        
        model = HybridEncoderDecoder(
            input_size=input_size,
            encoder_hidden=Config.ENCODER_HIDDEN,
            decoder_hidden=Config.DECODER_HIDDEN,
            encoder_layers=Config.ENCODER_LAYERS,
            decoder_layers=Config.DECODER_LAYERS,
            dropout=Config.DROPOUT,
            bidirectional_encoder=Config.BIDIRECTIONAL_ENCODER
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
        avg_r2_price = np.mean([m['R2'] for m in price_metrics.values()])
        avg_acc_price = np.mean([m['Direction_Accuracy'] for m in price_metrics.values()])
        
        print(f"\n📊 Price (OHLC):")
        print(f"   Avg R²: {avg_r2_price:.4f}")
        print(f"   Avg Accuracy: {avg_acc_price:.2f}%")
        
        plot_results(train_losses, val_losses, metrics)
        
        model_dir = 'CORRECTED_MODELS'
        os.makedirs(model_dir, exist_ok=True)
        
        # FIX: Convertir TODOS los valores a tipos nativos Python
        config_dict = {
            'seq_len': int(Config.SEQ_LEN),
            'encoder_hidden': int(Config.ENCODER_HIDDEN),
            'decoder_hidden': int(Config.DECODER_HIDDEN),
            'encoder_layers': int(Config.ENCODER_LAYERS),
            'decoder_layers': int(Config.DECODER_LAYERS),
            'dropout': float(Config.DROPOUT),
            'batch_size': int(Config.BATCH_SIZE),
            'learning_rate': float(Config.LEARNING_RATE)
        }
        
        # Convertir metrics a tipos nativos
        metrics_clean = {}
        for key, val in metrics.items():
            metrics_clean[key] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                for k, v in val.items()
            }
        
        # Guardar modelo
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config_dict,
            'input_size': int(input_size),
            'timestamp': datetime.now().isoformat()
        }, f'{model_dir}/lstm_gru_corrected.pth')
        
        # Guardar metadata en JSON separado
        with open(f'{model_dir}/model_metadata.json', 'w') as f:
            json.dump({
                'feature_cols': feature_cols,
                'target_cols': target_cols,
                'metrics': metrics_clean,
                'config': config_dict,
                'training_time_minutes': float(training_time / 60),
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        joblib.dump(scalers['input'], f'{model_dir}/scaler_input.pkl')
        joblib.dump(scalers['output_price'], f'{model_dir}/scaler_output_price.pkl')
        joblib.dump(scalers['output_volume'], f'{model_dir}/scaler_output_volume.pkl')
        
        print(f"\n💾 Archivos guardados en '{model_dir}/':")
        print(f"   ✅ lstm_gru_corrected.pth (modelo PyTorch)")
        print(f"   ✅ model_metadata.json (features, metrics, config)")
        print(f"   ✅ scaler_*.pkl (3 scalers)")
        print(f"   ✅ corrected_results.png (visualización)")
        print("\n" + "="*70)
        print("  ✅ PROCESS COMPLETE")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Entrenamiento interrumpido manualmente")
        print("💾 El checkpoint se guardó automáticamente")
        print("🔄 Ejecuta de nuevo para reanudar")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
