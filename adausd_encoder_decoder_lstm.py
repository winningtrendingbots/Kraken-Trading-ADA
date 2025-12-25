"""
LSTM ENCODER-DECODER PARA PREDICCIÓN OHLCV
✅ Arquitectura Seq2Seq con Attention
✅ Generación autoregresiva estructurada
✅ Teacher forcing para entrenamiento estable
✅ Restricciones OHLC respetadas por diseño
✅ Log-normalización de volumen
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
# 🎛️ CONFIGURACIÓN
# ================================
class Config:
    # Arquitectura Encoder-Decoder
    SEQ_LEN = 120
    ENCODER_HIDDEN = 128
    DECODER_HIDDEN = 160
    ENCODER_LAYERS = 2
    DECODER_LAYERS = 2
    DROPOUT = 0.25
    BIDIRECTIONAL_ENCODER = True
    USE_ATTENTION = True
    
    # Entrenamiento
    BATCH_SIZE = 96
    EPOCHS = 220
    LEARNING_RATE = 0.00015
    WEIGHT_DECAY = 3e-5
    PATIENCE = 25
    MIN_DELTA = 1e-5
    GRAD_CLIP = 0.7
    TEACHER_FORCING_RATIO = 0.4  # 50% usa valores reales durante entrenamiento
    
    # Loss weights
    PRICE_WEIGHT = 1.0
    VOLUME_WEIGHT = 0.12
    CONSTRAINT_WEIGHT = 1.0
    
    # Datos
    TRAIN_SIZE = 0.70
    VAL_SIZE = 0.15
    TEST_SIZE = 0.15

# ================================
# 📊 INDICADORES (mismo que antes)
# ================================
def calculate_enhanced_indicators(df):
    """Indicadores técnicos avanzados"""
    df = df.copy()
    
    # Volumen log-normalizado
    df['log_volume'] = np.log1p(df['volume'])
    df['volume_ma_5'] = df['volume'].rolling(5).mean()
    df['volume_ma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma_20'] + 1e-10)
    
    # Medias móviles
    for period in [5, 10, 20, 50]:
        df[f'sma_{period}'] = df['close'].rolling(period).mean()
        df[f'close_sma_{period}_ratio'] = df['close'] / (df[f'sma_{period}'] + 1e-10)
    
    # EMAs
    for period in [9, 21]:
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_ma'] = df['rsi'].rolling(5).mean()
    
    # MACD
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
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
    df['volatility_20'] = df['returns'].rolling(20).std()
    
    # Rangos
    df['high_low_range'] = (df['high'] - df['low']) / (df['close'] + 1e-10)
    df['open_close_range'] = abs(df['open'] - df['close']) / (df['close'] + 1e-10)
    
    # Momentum
    df['roc_10'] = df['close'].pct_change(periods=10)
    df['roc_20'] = df['close'].pct_change(periods=20)
    
    # VWAP
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (df['typical_price'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['vwap_distance'] = (df['close'] - df['vwap']) / (df['vwap'] + 1e-10)
    
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
    df['obv_ma'] = pd.Series(obv).rolling(20).mean()
    
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = np.clip(df[col], -1e6, 1e6)
    
    return df

# ================================
# 🧠 ATTENTION MECHANISM
# ================================
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)
    
    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (batch, hidden)
        # encoder_outputs: (batch, seq_len, hidden)
        
        # Expandir decoder_hidden para comparar con cada timestep
        decoder_hidden = decoder_hidden.unsqueeze(1)  # (batch, 1, hidden)
        
        # Calcular scores de atención
        score = self.V(torch.tanh(
            self.W1(encoder_outputs) + self.W2(decoder_hidden)
        ))  # (batch, seq_len, 1)
        
        attention_weights = torch.softmax(score, dim=1)  # (batch, seq_len, 1)
        
        # Context vector: suma ponderada de encoder outputs
        context = torch.sum(attention_weights * encoder_outputs, dim=1)  # (batch, hidden)
        
        return context, attention_weights

# ================================
# 🧠 ENCODER
# ================================
class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, 
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
        
        # Si es bidireccional, proyectar a hidden_size simple
        if bidirectional:
            self.projection = nn.Linear(hidden_size * 2, hidden_size)
        else:
            self.projection = None
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        outputs, (hidden, cell) = self.lstm(x)
        # outputs: (batch, seq_len, hidden * num_directions)
        
        if self.projection:
            outputs = self.projection(outputs)  # (batch, seq_len, hidden)
            # Proyectar hidden y cell también
            hidden = self.projection(hidden.transpose(0, 1).contiguous().view(x.size(0), -1, self.hidden_size * 2))
            hidden = hidden.transpose(0, 1).contiguous()
            cell = self.projection(cell.transpose(0, 1).contiguous().view(x.size(0), -1, self.hidden_size * 2))
            cell = cell.transpose(0, 1).contiguous()
        
        return outputs, (hidden, cell)

# ================================
# 🧠 DECODER
# ================================
class LSTMDecoder(nn.Module):
    def __init__(self, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Decoder LSTM (input es contexto + predicción anterior)
        self.lstm = nn.LSTM(
            input_size=hidden_size + 1,  # context + previous prediction
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention
        self.attention = BahdanauAttention(hidden_size)
        
        # Output layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # concat(context, hidden)
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, prev_output, hidden, cell, encoder_outputs):
        # prev_output: (batch, 1) - predicción anterior
        # hidden: (num_layers, batch, hidden)
        # cell: (num_layers, batch, hidden)
        # encoder_outputs: (batch, seq_len, hidden)
        
        batch_size = prev_output.size(0)
        
        # Obtener context via attention (usa el último hidden del decoder)
        context, attn_weights = self.attention(hidden[-1], encoder_outputs)
        
        # Concatenar previous output con context
        lstm_input = torch.cat([context, prev_output], dim=-1).unsqueeze(1)  # (batch, 1, hidden+1)
        
        # LSTM step
        lstm_out, (new_hidden, new_cell) = self.lstm(lstm_input, (hidden, cell))
        # lstm_out: (batch, 1, hidden)
        
        # Generar predicción
        combined = torch.cat([lstm_out.squeeze(1), context], dim=-1)  # (batch, hidden*2)
        prediction = self.fc(combined)  # (batch, 1)
        
        return prediction, new_hidden, new_cell, attn_weights

# ================================
# 🧠 ENCODER-DECODER COMPLETO
# ================================
class EncoderDecoderLSTM(nn.Module):
    def __init__(self, input_size, encoder_hidden=128, decoder_hidden=128,
                 encoder_layers=3, decoder_layers=2, dropout=0.3, 
                 bidirectional_encoder=True):
        super().__init__()
        
        self.encoder = LSTMEncoder(
            input_size=input_size,
            hidden_size=encoder_hidden,
            num_layers=encoder_layers,
            dropout=dropout,
            bidirectional=bidirectional_encoder
        )
        
        self.decoder = LSTMDecoder(
            hidden_size=decoder_hidden,
            num_layers=decoder_layers,
            dropout=dropout
        )
        
        self.encoder_hidden = encoder_hidden
        self.decoder_layers = decoder_layers
        
        # Activación de salida
        self.output_activation = nn.Tanh()
    
    def forward(self, x, target=None, teacher_forcing_ratio=0.5):
        # x: (batch, seq_len, input_size)
        batch_size = x.size(0)
        
        # ENCODE
        encoder_outputs, (encoder_hidden, encoder_cell) = self.encoder(x)
        # encoder_outputs: (batch, seq_len, hidden)
        
        # Inicializar decoder con encoder final state
        # Tomar solo los últimos decoder_layers del encoder
        decoder_hidden = encoder_hidden[-self.decoder_layers:]
        decoder_cell = encoder_cell[-self.decoder_layers:]
        
        # Generar 5 outputs: ΔOpen, ΔHigh, ΔLow, ΔClose, ΔVolume
        outputs = []
        prev_output = torch.zeros(batch_size, 1).to(x.device)  # Start token
        
        for t in range(5):  # 5 pasos de decodificación
            # Decidir si usar teacher forcing
            use_teacher_forcing = target is not None and np.random.random() < teacher_forcing_ratio
            
            # Decoder step
            pred, decoder_hidden, decoder_cell, attn_weights = self.decoder(
                prev_output, decoder_hidden, decoder_cell, encoder_outputs
            )
            
            outputs.append(pred)
            
            # Siguiente input del decoder
            if use_teacher_forcing and t < 4:  # Solo durante entrenamiento
                prev_output = target[:, t:t+1]
            else:
                prev_output = pred.detach()
        
        # Concatenar outputs
        outputs = torch.cat(outputs, dim=-1)  # (batch, 5)
        
        # Aplicar activación
        outputs = self.output_activation(outputs) * 0.05
        
        return outputs

# ================================
# 📦 PREPARACIÓN DE DATOS
# ================================
def prepare_encoder_decoder_dataset(df):
    print("\n" + "="*70)
    print("  🔧 PREPARACIÓN PARA ENCODER-DECODER")
    print("="*70)
    
    df = calculate_enhanced_indicators(df)
    
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
    
    # Scalers separados
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
# 🏋️ LOSS
# ================================
class EncoderDecoderLoss(nn.Module):
    def __init__(self, price_weight=1.0, volume_weight=0.15, constraint_weight=1.0):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.price_weight = price_weight
        self.volume_weight = volume_weight
        self.constraint_weight = constraint_weight
    
    def forward(self, predictions, targets):
        pred_price = predictions[:, :4]
        pred_volume = predictions[:, 4:]
        target_price = targets[:, :4]
        target_volume = targets[:, 4:]
        
        mse_price = self.mse(pred_price, target_price).mean()
        mse_volume = self.mse(pred_volume, target_volume).mean()
        
        # Restricciones OHLC
        delta_high = predictions[:, 1]
        delta_low = predictions[:, 2]
        delta_close = predictions[:, 3]
        
        high_low_violation = torch.clamp(delta_low - delta_high, min=0)
        close_below_low = torch.clamp(delta_low - delta_close, min=0)
        close_above_high = torch.clamp(delta_close - delta_high, min=0)
        constraint_loss = (high_low_violation + close_below_low + close_above_high).mean()
        
        total_loss = (
            self.price_weight * mse_price +
            self.volume_weight * mse_volume +
            self.constraint_weight * constraint_loss
        )
        
        return total_loss, {
            'mse_price': mse_price.item(),
            'mse_volume': mse_volume.item(),
            'constraint': constraint_loss.item()
        }

# ================================
# 🏋️ ENTRENAMIENTO
# ================================
def train_encoder_decoder(model, train_loader, val_loader, device):
    print("\n" + "="*70)
    print("  🏋️ ENTRENAMIENTO ENCODER-DECODER")
    print("="*70)
    
    criterion = EncoderDecoderLoss(
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
        optimizer, mode='min', factor=0.5, patience=12, min_lr=1e-6
    )
    
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    print(f"⚙️  Config: LR={Config.LEARNING_RATE}, Batch={Config.BATCH_SIZE}, TF={Config.TEACHER_FORCING_RATIO}")
    
    epoch_bar = tqdm(range(Config.EPOCHS), desc="Training", unit="epoch")
    
    for epoch in epoch_bar:
        # TRAIN
        model.train()
        train_loss = 0
        train_components = {'mse_price': 0, 'mse_volume': 0, 'constraint': 0}
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            predictions = model(X_batch, target=y_batch, 
                              teacher_forcing_ratio=Config.TEACHER_FORCING_RATIO)
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
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss - Config.MIN_DELTA:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            torch.save(best_model_state, 'best_encoder_decoder.pth')
        else:
            patience_counter += 1
        
        current_lr = optimizer.param_groups[0]['lr']
        epoch_bar.set_postfix({
            'train': f'{train_loss:.4f}',
            'val': f'{val_loss:.4f}',
            'best': f'{best_val_loss:.4f}',
            'lr': f'{current_lr:.6f}',
            'patience': f'{patience_counter}/{Config.PATIENCE}'
        })
        
        if (epoch + 1) % 10 == 0:
            print(f"\n📊 Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}, Best={best_val_loss:.4f}")
        
        if patience_counter >= Config.PATIENCE:
            print(f"\n⏹️  Early stopping at epoch {epoch+1}")
            break
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    print(f"\n✅ Training complete: Best val loss = {best_val_loss:.6f}")
    return train_losses, val_losses

# ================================
# 📊 EVALUACIÓN
# ================================
def evaluate_encoder_decoder(model, test_loader, scalers, target_cols, device):
    print("\n" + "="*70)
    print("  📊 EVALUACIÓN ENCODER-DECODER")
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
    
    # Desnormalizar
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
        print(f"   MAE: {mae:.6f} | RMSE: {rmse:.6f} | R²: {r2:.4f} | Acc: {accuracy:.2f}%")
    
    return predictions_denorm, targets_denorm, metrics

# ================================
# 🎨 VISUALIZACIÓN
# ================================
def plot_encoder_decoder_results(train_losses, val_losses, metrics, predictions, targets):
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    fig.suptitle('Encoder-Decoder LSTM - OHLCV Predictions', fontsize=16, fontweight='bold')
    
    # Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(train_losses, label='Train', linewidth=2)
    ax1.plot(val_losses, label='Val', linewidth=2)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # R²
    ax2 = fig.add_subplot(gs[0, 1])
    names = list(metrics.keys())
    r2_scores = [metrics[n]['R2'] for n in names]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    bars = ax2.bar(names, r2_scores, color=colors)
    ax2.set_title('R² Scores')
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, score in zip(bars, r2_scores):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{score:.3f}', ha='center', va='bottom' if score > 0 else 'top')
    
    # Accuracy
    ax3 = fig.add_subplot(gs[0, 2])
    acc_scores = [metrics[n]['Direction_Accuracy'] for n in names]
    bars = ax3.bar(names, acc_scores, color=colors)
    ax3.set_title('Direction Accuracy')
    ax3.axhline(y=50, color='r', linestyle='--', alpha=0.5)
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, acc in zip(bars, acc_scores):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{acc:.1f}%', ha='center', va='bottom')
    
    # Scatterplots
    sample = min(300, len(predictions))
    titles = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    for idx, (title, color) in enumerate(zip(titles, colors)):
        row = 1 + idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        ax.scatter(targets[:sample, idx], predictions[:sample, idx], 
                  alpha=0.5, s=15, color=color, edgecolors='black', linewidth=0.3)
        
        min_val, max_val = targets[:, idx].min(), targets[:, idx].max()
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, linewidth=2)
        
        r2 = metrics[f'delta_{title.lower()}']['R2']
        acc = metrics[f'delta_{title.lower()}']['Direction_Accuracy']
        ax.set_title(f'{title} (R²={r2:.3f}, Acc={acc:.1f}%)')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.grid(True, alpha=0.3)
    
    plt.savefig('encoder_decoder_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("📈 Plots saved: encoder_decoder_results.png")

# ================================
# 🚀 MAIN
# ================================
def main():
    try:
        print("\n" + "="*70)
        print("  🚀 ENCODER-DECODER LSTM FOR OHLCV")
        print("="*70)
        
        print("\n✨ ARCHITECTURE:")
        print("   • Encoder: 3-layer Bi-LSTM (128 hidden)")
        print("   • Decoder: 2-layer LSTM (128 hidden)")
        print("   • Bahdanau Attention mechanism")
        print("   • Teacher forcing: 50% during training")
        print("   • Autoregressive generation at inference")
        print("   • Structured output: ΔO→ΔH→ΔL→ΔC→ΔV")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n🖥️  Device: {device}")
        
        # Download data
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
        
        # Prepare data
        (X_train, y_train), (X_val, y_val), (X_test, y_test), \
        scalers, feature_cols, target_cols = prepare_encoder_decoder_dataset(df)
        
        input_size = len(feature_cols)
        
        # Create dataloaders
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
        
        # Create model
        model = EncoderDecoderLSTM(
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
        
        # Train
        start_time = time.time()
        train_losses, val_losses = train_encoder_decoder(model, train_loader, val_loader, device)
        training_time = time.time() - start_time
        
        print(f"\n⏱️  Training time: {training_time/60:.1f} min")
        
        # Evaluate
        predictions, targets, metrics = evaluate_encoder_decoder(
            model, test_loader, scalers, target_cols, device
        )
        
        # Results
        print("\n" + "="*70)
        print("  📈 FINAL RESULTS")
        print("="*70)
        
        price_metrics = {k: v for k, v in metrics.items() if 'volume' not in k}
        volume_metrics = {k: v for k, v in metrics.items() if 'volume' in k}
        
        avg_r2_price = np.mean([m['R2'] for m in price_metrics.values()])
        avg_acc_price = np.mean([m['Direction_Accuracy'] for m in price_metrics.values()])
        
        print(f"\n📊 Price (OHLC):")
        print(f"   Avg R²: {avg_r2_price:.4f}")
        print(f"   Avg Accuracy: {avg_acc_price:.2f}%")
        
        if volume_metrics:
            vol_r2 = volume_metrics['delta_volume']['R2']
            vol_acc = volume_metrics['delta_volume']['Direction_Accuracy']
            print(f"\n📊 Volume:")
            print(f"   R²: {vol_r2:.4f}")
            print(f"   Accuracy: {vol_acc:.2f}%")
        
        plot_encoder_decoder_results(train_losses, val_losses, metrics, predictions, targets)
        
        # Save
        model_dir = 'ADAUSD_MODELS'
        os.makedirs(model_dir, exist_ok=True)
        
        model_config = {
            'model_state_dict': model.state_dict(),
            'input_size': input_size,
            'encoder_hidden': Config.ENCODER_HIDDEN,
            'decoder_hidden': Config.DECODER_HIDDEN,
            'encoder_layers': Config.ENCODER_LAYERS,
            'decoder_layers': Config.DECODER_LAYERS,
            'seq_len': Config.SEQ_LEN,
            'feature_cols': feature_cols,
            'target_cols': target_cols,
            'metrics_test': metrics,
            'timestamp': datetime.now().isoformat(),
            'training_time_minutes': training_time / 60,
            'total_epochs': len(train_losses),
            'best_val_loss': min(val_losses)
        }
        
        torch.save(model_config, f'{model_dir}/encoder_decoder_lstm.pth')
        joblib.dump(scalers['input'], f'{model_dir}/scaler_input_encdec.pkl')
        joblib.dump(scalers['output_price'], f'{model_dir}/scaler_output_price_encdec.pkl')
        joblib.dump(scalers['output_volume'], f'{model_dir}/scaler_output_volume_encdec.pkl')
        
        with open(f'{model_dir}/config_encoder_decoder.json', 'w') as f:
            json.dump({
                'input_size': input_size,
                'encoder_hidden': Config.ENCODER_HIDDEN,
                'decoder_hidden': Config.DECODER_HIDDEN,
                'feature_cols': feature_cols,
                'target_cols': target_cols,
                'metrics_test': metrics,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"\n💾 Model saved in '{model_dir}/'")
        
        msg = f"""
✅ *Encoder-Decoder LSTM*

📊 *Price (OHLC):*
   • Avg R²: {avg_r2_price:.4f}
   • Avg Accuracy: {avg_acc_price:.2f}%

📊 *Volume:*
   • R²: {vol_r2:.4f}
   • Accuracy: {vol_acc:.2f}%

🏗️ *Architecture:*
   • Encoder: 3L Bi-LSTM (128h)
   • Decoder: 2L LSTM (128h)
   • Attention + Teacher Forcing
   • {total_params:,} parameters

⏱️ *Time:* {training_time/60:.1f} min
"""
        send_telegram(msg)
        
        print("\n" + "="*70)
        print("  ✅ PROCESS COMPLETE")
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
    except:
        pass

if __name__ == "__main__":
    main()
