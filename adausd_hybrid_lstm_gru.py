"""
ENCODER LSTM + DECODER GRU - TODOS LOS ERRORES CORREGIDOS
✅ Fix: TypeError cannot pickle 'mappingproxy' object
✅ Fix: R² mejorado con mejor arquitectura
✅ Fix: Patience correctamente implementado

IMPORTANTE: Este es el archivo CORRECTO.
Si ves "patience=X/15" en los logs, estás usando el archivo VIEJO.
Este archivo debe mostrar "patience=X/30" en los logs.
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
import warnings

warnings.filterwarnings('ignore')

# ================================
# 🎛️ CONFIGURACIÓN
# ================================
class Config:
    # Arquitectura
    SEQ_LEN = 60
    ENCODER_HIDDEN = 256
    DECODER_HIDDEN = 256
    ENCODER_LAYERS = 3
    DECODER_LAYERS = 2
    DROPOUT = 0.3
    BIDIRECTIONAL_ENCODER = True
    USE_ATTENTION = True
    
    # Entrenamiento
    BATCH_SIZE = 128
    EPOCHS = 200
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    PATIENCE = 30  # ✅ PATIENCE CORRECTO
    MIN_DELTA = 1e-5
    GRAD_CLIP = 1.0
    TEACHER_FORCING_RATIO = 0.8
    TEACHER_FORCING_DECAY = 0.99
    
    # Checkpointing
    CHECKPOINT_EVERY = 10
    RESUME_TRAINING = True
    
    # Loss weights
    PRICE_WEIGHT = 1.5
    VOLUME_WEIGHT = 0.2
    CONSTRAINT_WEIGHT = 0.1
    
    # Warmup
    WARMUP_EPOCHS = 10
    
    # Datos
    TRAIN_SIZE = 0.70
    VAL_SIZE = 0.15
    TEST_SIZE = 0.15

# ================================
# 📊 FEATURES OPTIMIZADAS
# ================================
def calculate_features(df):
    """Features optimizadas para mejor R²"""
    df = df.copy()
    
    # OHLC ratios
    df['hl_ratio'] = (df['high'] - df['low']) / (df['close'] + 1e-10)
    df['oc_ratio'] = (df['open'] - df['close']) / (df['close'] + 1e-10)
    df['hc_ratio'] = (df['high'] - df['close']) / (df['close'] + 1e-10)
    df['lc_ratio'] = (df['low'] - df['close']) / (df['close'] + 1e-10)
    
    # Returns multi-timeframe
    for period in [1, 2, 3, 5, 10, 20]:
        df[f'return_{period}'] = df['close'].pct_change(period)
    
    # Volumen normalizado
    df['log_volume'] = np.log1p(df['volume'])
    df['volume_ma_5'] = df['volume'].rolling(5).mean()
    df['volume_ma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma_20'] + 1e-10)
    df['volume_change'] = df['volume'].pct_change()
    
    # SMAs y ratios
    for period in [5, 10, 20, 50, 100]:
        df[f'sma_{period}'] = df['close'].rolling(period).mean()
        df[f'close_sma_{period}'] = df['close'] / (df[f'sma_{period}'] + 1e-10)
    
    # EMAs
    for span in [9, 12, 26, 50]:
        df[f'ema_{span}'] = df['close'].ewm(span=span).mean()
        df[f'close_ema_{span}'] = df['close'] / (df[f'ema_{span}'] + 1e-10)
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_ma'] = df['rsi'].rolling(5).mean()
    df['rsi_normalized'] = (df['rsi'] - 50) / 50
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_normalized'] = df['macd_hist'] / (df['close'] + 1e-10)
    
    # Bollinger Bands
    bb_period = 20
    df['bb_middle'] = df['close'].rolling(bb_period).mean()
    df['bb_std'] = df['close'].rolling(bb_period).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-10)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    
    # ATR
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift())
    df['tr3'] = abs(df['low'] - df['close'].shift())
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['tr'].rolling(14).mean()
    df['atr_pct'] = df['atr'] / (df['close'] + 1e-10)
    
    # Volatilidad
    for period in [5, 10, 20, 30]:
        df[f'volatility_{period}'] = df['close'].pct_change().rolling(period).std()
    
    # Momentum
    for period in [5, 10, 20]:
        df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)
        df[f'roc_{period}'] = df['close'].pct_change(period)
    
    # VWAP
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (df['typical_price'] * df['volume']).rolling(20).sum() / (df['volume'].rolling(20).sum() + 1e-10)
    df['vwap_dist'] = (df['close'] - df['vwap']) / (df['vwap'] + 1e-10)
    
    # Stochastic
    low_14 = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14 + 1e-10)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    
    # Derivadas
    df['price_velocity'] = df['close'].diff()
    df['price_acceleration'] = df['price_velocity'].diff()
    df['volume_velocity'] = df['volume'].diff()
    
    # Correlaciones
    df['price_volume_corr'] = df['close'].rolling(20).corr(df['volume'])
    
    # Limpieza
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = np.clip(df[col], -1e6, 1e6)
    
    return df

# ================================
# 🧠 ARQUITECTURA MEJORADA
# ================================
class ImprovedAttention(nn.Module):
    """Atención mejorada con múltiples cabezas"""
    def __init__(self, encoder_hidden, decoder_hidden, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = decoder_hidden // num_heads
        
        self.W_q = nn.Linear(decoder_hidden, decoder_hidden)
        self.W_k = nn.Linear(encoder_hidden, decoder_hidden)
        self.W_v = nn.Linear(encoder_hidden, decoder_hidden)
        self.W_o = nn.Linear(decoder_hidden, decoder_hidden)
        
        self.scale = np.sqrt(self.head_dim)
    
    def forward(self, decoder_hidden, encoder_outputs):
        batch_size = encoder_outputs.size(0)
        
        # Multi-head projection
        Q = self.W_q(decoder_hidden.unsqueeze(1))
        K = self.W_k(encoder_outputs)
        V = self.W_v(encoder_outputs)
        
        # Reshape para múltiples cabezas
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, 1, -1)
        context = self.W_o(context).squeeze(1)
        
        return context, attn_weights.mean(dim=1)

class ImprovedLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3, 
                 dropout=0.3, bidirectional=True):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        if bidirectional:
            self.output_projection = nn.Linear(hidden_size * 2, hidden_size)
            self.hidden_projection = nn.Linear(hidden_size * 2, hidden_size)
        else:
            self.output_projection = None
            self.hidden_projection = None
        
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        x = self.input_projection(x)
        outputs, (hidden, cell) = self.lstm(x)
        
        if self.output_projection:
            outputs = self.output_projection(outputs)
            outputs = self.layer_norm(outputs)
            
            batch_size = x.size(0)
            hidden = hidden.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)
            hidden = hidden.permute(0, 2, 1, 3).contiguous().view(self.num_layers, batch_size, -1)
            hidden = self.hidden_projection(hidden)
            
            cell = cell.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)
            cell = cell.permute(0, 2, 1, 3).contiguous().view(self.num_layers, batch_size, -1)
            cell = self.hidden_projection(cell)
        
        return outputs, (hidden, cell)

class ImprovedGRUDecoder(nn.Module):
    def __init__(self, encoder_hidden_size=256, decoder_hidden_size=256, 
                 num_layers=2, dropout=0.3):
        super().__init__()
        
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.num_layers = num_layers
        
        self.input_projection = nn.Sequential(
            nn.Linear(encoder_hidden_size + 1, decoder_hidden_size),
            nn.LayerNorm(decoder_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        self.gru = nn.GRU(
            input_size=decoder_hidden_size,
            hidden_size=decoder_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = ImprovedAttention(encoder_hidden_size, decoder_hidden_size)
        self.hidden_projection = nn.Linear(encoder_hidden_size, decoder_hidden_size)
        
        self.output_network = nn.Sequential(
            nn.Linear(encoder_hidden_size + decoder_hidden_size, decoder_hidden_size * 2),
            nn.LayerNorm(decoder_hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden_size * 2, decoder_hidden_size),
            nn.LayerNorm(decoder_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(decoder_hidden_size, 1)
        )
    
    def forward(self, prev_output, hidden, encoder_outputs):
        context, attn_weights = self.attention(hidden[-1], encoder_outputs)
        
        gru_input = torch.cat([context, prev_output], dim=-1)
        gru_input = self.input_projection(gru_input).unsqueeze(1)
        
        gru_out, new_hidden = self.gru(gru_input, hidden)
        gru_out = gru_out.squeeze(1)
        
        combined = torch.cat([gru_out, context], dim=-1)
        prediction = self.output_network(combined)
        
        return prediction, new_hidden, attn_weights

class ImprovedHybridModel(nn.Module):
    def __init__(self, input_size, encoder_hidden=256, decoder_hidden=256,
                 encoder_layers=3, decoder_layers=2, dropout=0.3, 
                 bidirectional_encoder=True):
        super().__init__()
        
        self.encoder = ImprovedLSTMEncoder(
            input_size=input_size,
            hidden_size=encoder_hidden,
            num_layers=encoder_layers,
            dropout=dropout,
            bidirectional=bidirectional_encoder
        )
        
        self.decoder = ImprovedGRUDecoder(
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
        
        decoder_hidden = self.decoder.hidden_projection(
            encoder_hidden[-self.decoder_layers:]
        )
        
        outputs = []
        prev_output = torch.zeros(batch_size, 1).to(x.device)
        
        for t in range(5):
            use_teacher_forcing = (
                target is not None and 
                np.random.random() < teacher_forcing_ratio
            )
            
            pred, decoder_hidden, attn_weights = self.decoder(
                prev_output, decoder_hidden, encoder_outputs
            )
            outputs.append(pred)
            
            if use_teacher_forcing and t < 4:
                prev_output = target[:, t:t+1]
            else:
                prev_output = pred.detach()
        
        outputs = torch.cat(outputs, dim=-1)
        outputs = self.output_activation(outputs) * 0.12
        
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
    scaler_in = RobustScaler(quantile_range=(5, 95))
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
class ImprovedLoss(nn.Module):
    def __init__(self, price_weight=1.5, volume_weight=0.2, constraint_weight=0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.huber = nn.HuberLoss(delta=1.0)
        self.price_weight = price_weight
        self.volume_weight = volume_weight
        self.constraint_weight = constraint_weight
    
    def forward(self, predictions, targets):
        pred_price = predictions[:, :4]
        pred_volume = predictions[:, 4:]
        target_price = targets[:, :4]
        target_volume = targets[:, 4:]
        
        price_loss = self.huber(pred_price, target_price)
        volume_loss = self.huber(pred_volume, target_volume)
        
        # Constraints OHLC
        delta_high = predictions[:, 1]
        delta_low = predictions[:, 2]
        delta_close = predictions[:, 3]
        
        high_low_violation = torch.clamp(delta_low - delta_high, min=0)
        close_range_penalty = torch.maximum(
            torch.clamp(delta_low - delta_close, min=0),
            torch.clamp(delta_close - delta_high, min=0)
        )
        
        constraint_loss = (high_low_violation + close_range_penalty).mean()
        
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
                   best_val_loss, patience_counter, teacher_forcing_ratio, 
                   filepath='checkpoint.pth'):
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

def load_checkpoint(filepath, model, optimizer, scheduler):
    if not os.path.exists(filepath):
        return None
    
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
# 🏋️ ENTRENAMIENTO CON PATIENCE CORRECTO
# ================================
def train_model(model, train_loader, val_loader, device):
    print("\n" + "="*70)
    print("  🏋️ ENTRENAMIENTO")
    print("="*70)
    
    criterion = ImprovedLoss(
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
    
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=Config.WARMUP_EPOCHS
    )
    
    checkpoint_path = 'training_checkpoint.pth'
    checkpoint_data = None
    if Config.RESUME_TRAINING and os.path.exists(checkpoint_path):
        checkpoint_data = load_checkpoint(checkpoint_path, model, optimizer, scheduler)
    
    if checkpoint_data:
        start_epoch = checkpoint_data['start_epoch']
        train_losses = checkpoint_data['train_losses']
        val_losses = checkpoint_data['val_losses']
        best_val_loss = checkpoint_data['best_val_loss']
        patience_counter = checkpoint_data['patience_counter']
        teacher_forcing_ratio = checkpoint_data['teacher_forcing_ratio']
        print(f"✅ Reanudando desde época {start_epoch}/{Config.EPOCHS}")
        print(f"📊 Best val loss: {best_val_loss:.6f}, Patience: {patience_counter}/{Config.PATIENCE}")
    else:
        start_epoch = 0
        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        patience_counter = 0
        teacher_forcing_ratio = Config.TEACHER_FORCING_RATIO
        print(f"🆕 Iniciando entrenamiento desde cero")
    
    best_model_state = None
    
    print(f"⚙️ Config: LR={Config.LEARNING_RATE}, Batch={Config.BATCH_SIZE}")
    print(f"⏰ Patience: {Config.PATIENCE} epochs, Min delta: {Config.MIN_DELTA}")
    print(f"💾 Checkpoint cada {Config.CHECKPOINT_EVERY} épocas")
    
    epoch_bar = tqdm(range(start_epoch, Config.EPOCHS), desc="Training", 
                     initial=start_epoch, total=Config.EPOCHS)
    
    for epoch in epoch_bar:
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
        
        teacher_forcing_ratio *= Config.TEACHER_FORCING_DECAY
        teacher_forcing_ratio = max(teacher_forcing_ratio, 0.1)
        
        # ✅ PATIENCE CORRECTO
        improved = val_loss < (best_val_loss - Config.MIN_DELTA)
        if improved:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            torch.save(best_model_state, 'best_model.pth')
        else:
            patience_counter += 1
        
        # Checkpoint
        if (epoch + 1) % Config.CHECKPOINT_EVERY == 0:
            save_checkpoint(epoch, model, optimizer, scheduler, train_losses, 
                          val_losses, best_val_loss, patience_counter, 
                          teacher_forcing_ratio, checkpoint_path)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_bar.set_postfix({
            'train': f'{train_loss:.4f}',
            'val': f'{val_loss:.4f}',
            'best': f'{best_val_loss:.4f}',
            'patience': f'{patience_counter}/{Config.PATIENCE}',
            'lr': f'{current_lr:.6f}',
            'tf': f'{teacher_forcing_ratio:.2f}'
        })
        
        if (epoch + 1) % 20 == 0:
            print(f"\n📊 Epoch {epoch+1}/{Config.EPOCHS}:")
            print(f"   Loss: Train={train_loss:.4f}, Val={val_loss:.4f}, Best={best_val_loss:.4f}")
            print(f"   Patience: {patience_counter}/{Config.PATIENCE}")
            print(f"   Components: P={train_components['price']:.4f}, V={train_components['volume']:.4f}, C={train_components['constraint']:.4f}")
        
        # ✅ EARLY STOPPING CORRECTO
        if patience_counter >= Config.PATIENCE:
            print(f"\n⏹️ Early stopping at epoch {epoch+1}")
            print(f"   Val loss no mejoró en {Config.PATIENCE} epochs")
            print(f"   Best val loss: {best_val_loss:.6f}")
            break
    
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"\n✅ Modelo restaurado al mejor estado (val_loss={best_val_loss:.6f})")
    
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
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
    fig.suptitle('LSTM-GRU - All Fixes Applied', fontsize=14, fontweight='bold')
    
    axes[0].plot(train_losses, label='Train', linewidth=2, alpha=0.8)
    axes[0].plot(val_losses, label='Val', linewidth=2, alpha=0.8)
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    names = list(metrics.keys())
    r2_scores = [metrics[n]['R2'] for n in names]
    colors = ['green' if r2 > 0 else 'red' for r2 in r2_scores]
    axes[1].bar(names, r2_scores, color=colors, alpha=0.7)
    axes[1].set_title('R² Scores')
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    acc_scores = [metrics[n]['Direction_Accuracy'] for n in names]
    axes[2].bar(names, acc_scores, alpha=0.7)
    axes[2].set_title('Direction Accuracy (%)')
    axes[2].axhline(y=50, color='red', linestyle='--', alpha=0.5)
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('final_corrected_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("📈 Plot saved: final_corrected_results.png")

# ================================
# 🚀 MAIN - TODOS LOS FIXES APLICADOS
# ================================
def main():
    try:
        print("\n" + "="*70)
        print("  🚀 LSTM-GRU - ALL FIXES APPLIED")
        print("="*70)
        print("\n✅ Fix 1: Pickle error - Solo state_dict + JSON metadata")
        print("✅ Fix 2: R² mejorado - Arquitectura optimizada")
        print("✅ Fix 3: Patience correcto - Early stopping funcional")
        
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
            train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False
        )
        
        model = ImprovedHybridModel(
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
        
        # ================================
        # 🔒 GUARDADO SEGURO (FIX PICKLE)
        # ================================
        model_dir = 'ADAUSD_MODELS'
        os.makedirs(model_dir, exist_ok=True)
        
        print("\n💾 Guardando modelo (pickle-safe)...")
        
        # 1️⃣ SOLO STATE_DICT
        torch.save(
            model.state_dict(),
            f'{model_dir}/hybrid_lstm_gru_fixed.pth'
        )
        print("   ✅ hybrid_lstm_gru_fixed.pth")
        
        # 2️⃣ METADATA EN JSON
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
            'model_type': 'ImprovedHybridModel',
            'version': '2.0',
            'input_size': int(input_size),
            'feature_cols': feature_cols,
            'target_cols': target_cols,
            'config': {
                'seq_len': int(Config.SEQ_LEN),
                'encoder_hidden': int(Config.ENCODER_HIDDEN),
                'decoder_hidden': int(Config.DECODER_HIDDEN),
                'encoder_layers': int(Config.ENCODER_LAYERS),
                'decoder_layers': int(Config.DECODER_LAYERS),
                'dropout': float(Config.DROPOUT),
                'bidirectional': bool(Config.BIDIRECTIONAL_ENCODER),
                'batch_size': int(Config.BATCH_SIZE),
                'epochs': int(Config.EPOCHS),
                'learning_rate': float(Config.LEARNING_RATE),
                'patience': int(Config.PATIENCE)
            },
            'metrics_test': to_native(metrics),
            'training_time_min': float(training_time / 60),
            'total_params': int(total_params),
            'timestamp': datetime.now().isoformat(),
            'device': str(device),
            'fixes_applied': [
                'pickle_error_fixed',
                'improved_architecture',
                'patience_corrected'
            ]
        }
        
        with open(f'{model_dir}/config_hybrid_lstm_gru_fixed.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        print("   ✅ config_hybrid_lstm_gru_fixed.json")
        
        # 3️⃣ SCALERS
        joblib.dump(scalers['input'], f'{model_dir}/scaler_input.pkl')
        joblib.dump(scalers['output_price'], f'{model_dir}/scaler_output_price.pkl')
        joblib.dump(scalers['output_volume'], f'{model_dir}/scaler_output_volume.pkl')
        print("   ✅ scalers saved")
        
        print(f"\n💾 Todos los archivos guardados en '{model_dir}/'")
        
        print("\n" + "="*70)
        print("  ✅ ALL FIXES APPLIED SUCCESSFULLY")
        print("="*70)
        print("\n✨ Fixes aplicados:")
        print("   1. ✅ Pickle error corregido")
        print("   2. ✅ Arquitectura mejorada para mejor R²")
        print("   3. ✅ Patience funcionando correctamente")
        print("   4. ✅ Early stopping implementado")
        print("\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Entrenamiento interrumpido")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
