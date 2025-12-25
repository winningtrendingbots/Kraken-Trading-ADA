"""
ENCODER LSTM + DECODER GRU HÍBRIDO CON ATTENTION + ANÁLISIS AVANZADO
✅ Encoder: LSTM bidireccional (captura contexto completo)
✅ Decoder: GRU (más rápido, menos overfitting)
✅ Bahdanau Attention (enfoque dinámico en historia)
✅ Teacher forcing adaptativo
✅ Restricciones OHLC por diseño
✅ Log-normalización de volumen
✅ Derivadas primeras y segundas de precio y volumen
✅ Correlaciones precio-volumen
✅ Exportación de métricas avanzadas a CSV
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

'''
# ================================
# 🎛️ CONFIGURACIÓN OPTIMIZADA
# ================================
class Config:
    # Arquitectura Híbrida Encoder-Decoder
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
    TEACHER_FORCING_RATIO = 0.4
    
    # Loss weights
    PRICE_WEIGHT = 1.0
    VOLUME_WEIGHT = 0.12
    CONSTRAINT_WEIGHT = 1.0
    
    # Datos
    TRAIN_SIZE = 0.70
    VAL_SIZE = 0.15
    TEST_SIZE = 0.15

'''
class Config:
    # Arquitectura (REDUCIDA para velocidad)
    SEQ_LEN = 60  # ⬇️ Reducido de 120 a 60
    ENCODER_HIDDEN = 96  # ⬇️ Reducido de 128 a 96
    DECODER_HIDDEN = 128  # ⬇️ Reducido de 160 a 128
    ENCODER_LAYERS = 1  # ⬇️ Reducido de 2 a 1 (gran impacto en velocidad)
    DECODER_LAYERS = 1  # ⬇️ Reducido de 2 a 1
    DROPOUT = 0.2
    BIDIRECTIONAL_ENCODER = True
    USE_ATTENTION = True
    
    # Entrenamiento (OPTIMIZADO)
    BATCH_SIZE = 128  # ⬆️ Aumentado para procesamiento más rápido
    EPOCHS = 100  # ⬇️ Reducido de 220 a 100
    LEARNING_RATE = 0.0003  # ⬆️ Aumentado para convergencia más rápida
    WEIGHT_DECAY = 1e-5
    PATIENCE = 15  # ⬇️ Reducido de 25 a 15
    MIN_DELTA = 1e-5
    GRAD_CLIP = 1.0
    TEACHER_FORCING_RATIO = 0.3  # ⬇️ Reducido de 0.4 a 0.3
    
    # Checkpointing
    CHECKPOINT_EVERY = 5  # Guardar cada 5 épocas
    RESUME_TRAINING = True  # Reanudar si existe checkpoint
    
    # Loss weights
    PRICE_WEIGHT = 1.0
    VOLUME_WEIGHT = 0.12
    CONSTRAINT_WEIGHT = 0.5  # ⬇️ Reducido para acelerar convergencia
    
    # Datos
    TRAIN_SIZE = 0.70
    VAL_SIZE = 0.15
    TEST_SIZE = 0.15


# ================================
# 📊 ANÁLISIS SIMPLIFICADO (FEATURES ESENCIALES)
# ================================
def calculate_essential_features(df):
    """Calcula solo las features más importantes para velocidad"""
    df = df.copy()
    
    # Volumen log-normalizado
    df['log_volume'] = np.log1p(df['volume'])
    df['volume_ma_5'] = df['volume'].rolling(5).mean()
    df['volume_ma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma_20'] + 1e-10)
    
    # Medias móviles esenciales
    for period in [5, 10, 20]:
        df[f'sma_{period}'] = df['close'].rolling(period).mean()
        df[f'close_sma_{period}_ratio'] = df['close'] / (df[f'sma_{period}'] + 1e-10)
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # Volatilidad
    df['returns'] = df['close'].pct_change()
    df['volatility_20'] = df['returns'].rolling(20).std()
    
    # Derivadas esenciales
    df['price_velocity'] = df['close'].pct_change()
    df['volume_velocity'] = df['volume'].pct_change()
    df['price_volume_corr_20'] = df['close'].rolling(20).corr(df['volume'])
    
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = np.clip(df[col], -1e6, 1e6)
    
    return df

# ================================
# 📊 ANÁLISIS AVANZADO DE DERIVADAS Y CORRELACIONES
# ================================
def calculate_advanced_derivatives(df):
    """
    Calcula derivadas primeras y segundas para precio y volumen,
    así como correlaciones entre ambos
    """
    df = df.copy()
    
    # ========== DERIVADAS DE PRECIO ==========
    # Primera derivada (velocidad de cambio del precio)
    df['price_first_deriv'] = df['close'].diff()
    df['price_velocity'] = df['close'].pct_change()
    
    # Segunda derivada (aceleración del precio)
    df['price_second_deriv'] = df['price_first_deriv'].diff()
    df['price_acceleration'] = df['price_velocity'].diff()
    
    # Derivadas suavizadas con media móvil
    df['price_first_deriv_ma5'] = df['price_first_deriv'].rolling(5).mean()
    df['price_second_deriv_ma5'] = df['price_second_deriv'].rolling(5).mean()
    
    # ========== DERIVADAS DE VOLUMEN ==========
    # Primera derivada (velocidad de cambio del volumen)
    df['volume_first_deriv'] = df['volume'].diff()
    df['volume_velocity'] = df['volume'].pct_change()
    
    # Segunda derivada (aceleración del volumen)
    df['volume_second_deriv'] = df['volume_first_deriv'].diff()
    df['volume_acceleration'] = df['volume_velocity'].diff()
    
    # Derivadas suavizadas con media móvil
    df['volume_first_deriv_ma5'] = df['volume_first_deriv'].rolling(5).mean()
    df['volume_second_deriv_ma5'] = df['volume_second_deriv'].rolling(5).mean()
    
    # ========== CORRELACIONES PRECIO-VOLUMEN ==========
    # Correlación rolling entre precio y volumen
    df['price_volume_corr_10'] = df['close'].rolling(10).corr(df['volume'])
    df['price_volume_corr_20'] = df['close'].rolling(20).corr(df['volume'])
    df['price_volume_corr_50'] = df['close'].rolling(50).corr(df['volume'])
    
    # Correlación entre velocidades
    df['velocity_corr_10'] = df['price_velocity'].rolling(10).corr(df['volume_velocity'])
    df['velocity_corr_20'] = df['price_velocity'].rolling(20).corr(df['volume_velocity'])
    
    # Correlación entre aceleraciones
    df['accel_corr_10'] = df['price_acceleration'].rolling(10).corr(df['volume_acceleration'])
    df['accel_corr_20'] = df['price_acceleration'].rolling(20).corr(df['volume_acceleration'])
    
    # ========== MÉTRICAS ADICIONALES ==========
    # Ratio de cambio precio/volumen
    df['price_vol_ratio'] = df['price_velocity'] / (df['volume_velocity'].abs() + 1e-10)
    
    # Divergencia precio-volumen (cuando se mueven en direcciones opuestas)
    df['price_vol_divergence'] = (np.sign(df['price_velocity']) != np.sign(df['volume_velocity'])).astype(int)
    
    # Momentum conjunto
    df['joint_momentum'] = df['price_velocity'] * df['volume_velocity']
    
    # Limpieza de valores
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = np.clip(df[col], -1e6, 1e6)
    
    return df

# ================================
# 💾 CHECKPOINTING
# ================================
def save_checkpoint(epoch, model, optimizer, scheduler, train_losses, val_losses, 
                   best_val_loss, patience_counter, filepath='checkpoint.pth'):
    """Guarda checkpoint completo del entrenamiento"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'patience_counter': patience_counter,
        'config': {
            'seq_len': Config.SEQ_LEN,
            'encoder_hidden': Config.ENCODER_HIDDEN,
            'decoder_hidden': Config.DECODER_HIDDEN,
            'encoder_layers': Config.ENCODER_LAYERS,
            'decoder_layers': Config.DECODER_LAYERS,
        }
    }
    torch.save(checkpoint, filepath)
    print(f"💾 Checkpoint guardado: {filepath}")


def load_checkpoint(filepath, model, optimizer, scheduler):
    """Carga checkpoint para reanudar entrenamiento"""
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
        'patience_counter': checkpoint['patience_counter']
    }

# ================================
# 📊 EXPORTAR MÉTRICAS A CSV
# ================================
def export_derivatives_to_csv(df, filename='price_volume_derivatives.csv'):
    """
    Exporta las derivadas y correlaciones calculadas a un CSV
    """
    derivative_cols = [
        'time', 'close', 'volume',
        # Derivadas de precio
        'price_first_deriv', 'price_velocity', 
        'price_second_deriv', 'price_acceleration',
        'price_first_deriv_ma5', 'price_second_deriv_ma5',
        # Derivadas de volumen
        'volume_first_deriv', 'volume_velocity',
        'volume_second_deriv', 'volume_acceleration',
        'volume_first_deriv_ma5', 'volume_second_deriv_ma5',
        # Correlaciones
        'price_volume_corr_10', 'price_volume_corr_20', 'price_volume_corr_50',
        'velocity_corr_10', 'velocity_corr_20',
        'accel_corr_10', 'accel_corr_20',
        # Métricas adicionales
        'price_vol_ratio', 'price_vol_divergence', 'joint_momentum'
    ]
    
    # Filtrar solo las columnas que existen
    export_cols = [col for col in derivative_cols if col in df.columns]
    
    df_export = df[export_cols].copy()
    df_export.to_csv(filename, index=False)
    print(f"\n✅ Derivadas y correlaciones exportadas a: {filename}")
    
    return filename

# ================================
# 📊 ESTADÍSTICAS DE DERIVADAS
# ================================
def analyze_derivatives_statistics(df):
    """
    Genera estadísticas descriptivas de las derivadas y correlaciones
    """
    print("\n" + "="*70)
    print("  📊 ESTADÍSTICAS DE DERIVADAS Y CORRELACIONES")
    print("="*70)
    
    stats_dict = {}
    
    # Grupos de análisis
    price_derivs = ['price_first_deriv', 'price_second_deriv', 'price_velocity', 'price_acceleration']
    volume_derivs = ['volume_first_deriv', 'volume_second_deriv', 'volume_velocity', 'volume_acceleration']
    correlations = ['price_volume_corr_10', 'price_volume_corr_20', 'price_volume_corr_50',
                   'velocity_corr_10', 'velocity_corr_20', 'accel_corr_10', 'accel_corr_20']
    
    # Analizar derivadas de precio
    print("\n📈 DERIVADAS DE PRECIO:")
    for col in price_derivs:
        if col in df.columns:
            data = df[col].dropna()
            stats_dict[col] = {
                'mean': float(data.mean()),
                'std': float(data.std()),
                'min': float(data.min()),
                'max': float(data.max()),
                'median': float(data.median())
            }
            print(f"  {col:25} | Mean: {data.mean():10.6f} | Std: {data.std():10.6f}")
    
    # Analizar derivadas de volumen
    print("\n📊 DERIVADAS DE VOLUMEN:")
    for col in volume_derivs:
        if col in df.columns:
            data = df[col].dropna()
            stats_dict[col] = {
                'mean': float(data.mean()),
                'std': float(data.std()),
                'min': float(data.min()),
                'max': float(data.max()),
                'median': float(data.median())
            }
            print(f"  {col:25} | Mean: {data.mean():10.6f} | Std: {data.std():10.6f}")
    
    # Analizar correlaciones
    print("\n🔗 CORRELACIONES PRECIO-VOLUMEN:")
    for col in correlations:
        if col in df.columns:
            data = df[col].dropna()
            stats_dict[col] = {
                'mean': float(data.mean()),
                'std': float(data.std()),
                'min': float(data.min()),
                'max': float(data.max()),
                'median': float(data.median())
            }
            print(f"  {col:25} | Mean: {data.mean():10.6f} | Std: {data.std():10.6f}")
    
    # Exportar estadísticas a JSON
    with open('derivatives_statistics.json', 'w') as f:
        json.dump(stats_dict, f, indent=2)
    print(f"\n✅ Estadísticas exportadas a: derivatives_statistics.json")
    
    return stats_dict

# ================================
# 📊 INDICADORES TÉCNICOS (MANTENIDO DEL ORIGINAL)
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
# 🧠 BAHDANAU ATTENTION (HÍBRIDO)
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

# ================================
# 🧠 ENCODER (LSTM)
# ================================
class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size=96, num_layers=1, 
                 dropout=0.2, bidirectional=True):
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

# ================================
# 🧠 DECODER (GRU) ✨
# ================================
class GRUDecoder(nn.Module):
    def __init__(self, encoder_hidden_size=96, decoder_hidden_size=128, 
                 num_layers=1, dropout=0.2):
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
            nn.Linear(encoder_hidden_size + decoder_hidden_size, decoder_hidden_size),
            nn.LayerNorm(decoder_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden_size, 1)
        )
    
    def forward(self, prev_output, hidden, encoder_outputs):
        context, attn_weights = self.attention(hidden[-1], encoder_outputs)
        gru_input = torch.cat([context, prev_output], dim=-1).unsqueeze(1)
        gru_out, new_hidden = self.gru(gru_input, hidden)
        combined = torch.cat([gru_out.squeeze(1), context], dim=-1)
        prediction = self.fc(combined)
        return prediction, new_hidden, attn_weights

# ================================
# 🧠 ENCODER-DECODER HÍBRIDO
# ================================
class HybridEncoderDecoder(nn.Module):
    def __init__(self, input_size, encoder_hidden=96, decoder_hidden=128,
                 encoder_layers=1, decoder_layers=1, dropout=0.2, 
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
    
    def forward(self, x, target=None, teacher_forcing_ratio=0.3):
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
        outputs = self.output_activation(outputs) * 0.05
        return outputs


# ================================
# 📦 PREPARACIÓN DE DATOS
# ================================
def prepare_dataset(df):
    print("\n" + "="*70)
    print("  🔧 PREPARACIÓN DE DATOS OPTIMIZADA")
    print("="*70)
    
    df = calculate_essential_features(df)
    
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
# 📦 PREPARACIÓN DE DATOS CON DERIVADAS
# ================================
def prepare_hybrid_dataset(df):
    print("\n" + "="*70)
    print("  🔧 PREPARACIÓN PARA MODELO HÍBRIDO + DERIVADAS")
    print("="*70)
    
    # Calcular indicadores técnicos
    df = calculate_enhanced_indicators(df)
    
    # Calcular derivadas y correlaciones avanzadas
    df = calculate_advanced_derivatives(df)
    
    # Exportar derivadas a CSV
    export_derivatives_to_csv(df, 'ADAUSD_derivatives_correlations.csv')
    
    # Analizar estadísticas de derivadas
    analyze_derivatives_statistics(df)
    
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
    
    print(f"🎯 {len(feature_cols)} features (incluyendo derivadas), {len(target_cols)} targets")
    
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
class HybridLoss(nn.Module):
    def __init__(self, price_weight=1.0, volume_weight=0.12, constraint_weight=0.5):
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
def train_hybrid_model(model, train_loader, val_loader, device):
    print("\n" + "="*70)
    print("  🏋️ ENTRENAMIENTO MODELO HÍBRIDO")
    print("="*70)
    
    criterion = HybridLoss(
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
    
    print(f"⚙️ Config: LR={Config.LEARNING_RATE}, Batch={Config.BATCH_SIZE}, TF={Config.TEACHER_FORCING_RATIO}")
    
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
            torch.save(best_model_state, 'best_hybrid_model.pth')
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
            print(f"\nℹ️ Early stopping at epoch {epoch+1}")
            break
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    print(f"\n✅ Training complete: Best val loss = {best_val_loss:.6f}")
    return train_losses, val_losses

def train_model(model, train_loader, val_loader, device):
    print("\n" + "="*70)
    print("  🏋️ ENTRENAMIENTO OPTIMIZADO CON CHECKPOINTING")
    print("="*70)
    
    criterion = HybridLoss(
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
        optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-6
    )
    
    # Intentar cargar checkpoint
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
        print(f"✅ Reanudando desde época {start_epoch}/{Config.EPOCHS}")
    else:
        start_epoch = 0
        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        patience_counter = 0
        print(f"🆕 Iniciando entrenamiento desde cero")
    
    best_model_state = None
    
    print(f"⚙️ Config: LR={Config.LEARNING_RATE}, Batch={Config.BATCH_SIZE}, Epochs={Config.EPOCHS}")
    print(f"💾 Checkpoint automático cada {Config.CHECKPOINT_EVERY} épocas")
    
    epoch_bar = tqdm(range(start_epoch, Config.EPOCHS), desc="Training", unit="epoch", initial=start_epoch, total=Config.EPOCHS)
    
    for epoch in epoch_bar:
        epoch_start_time = time.time()
        
        # TRAIN
        model.train()
        train_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            predictions = model(X_batch, target=y_batch, 
                              teacher_forcing_ratio=Config.TEACHER_FORCING_RATIO)
            loss, _ = criterion(predictions, y_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
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
            torch.save(best_model_state, 'best_model_optimized.pth')
        else:
            patience_counter += 1
        
        # Checkpoint periódico
        if (epoch + 1) % Config.CHECKPOINT_EVERY == 0:
            save_checkpoint(epoch, model, optimizer, scheduler, train_losses, val_losses,
                          best_val_loss, patience_counter, checkpoint_path)
        
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_bar.set_postfix({
            'train': f'{train_loss:.4f}',
            'val': f'{val_loss:.4f}',
            'best': f'{best_val_loss:.4f}',
            'lr': f'{current_lr:.6f}',
            'patience': f'{patience_counter}/{Config.PATIENCE}',
            'time': f'{epoch_time:.1f}s'
        })
        
        if patience_counter >= Config.PATIENCE:
            print(f"\n⏹️ Early stopping at epoch {epoch+1}")
            break
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Limpiar checkpoint al finalizar con éxito
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"🗑️ Checkpoint temporal eliminado")
    
    print(f"\n✅ Training complete: Best val loss = {best_val_loss:.6f}")
    return train_losses, val_losses

# ================================
# 📊 EVALUACIÓN
# ================================
def evaluate_hybrid_model(model, test_loader, scalers, target_cols, device):
    print("\n" + "="*70)
    print("  📊 EVALUACIÓN MODELO HÍBRIDO")
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
        
        print(f"📊 {col}: MAE={mae:.6f}, RMSE={rmse:.6f}, R²={r2:.4f}, Acc={accuracy:.2f}%")
    
    return predictions_denorm, targets_denorm, metrics

# ================================
# 🎨 VISUALIZACIÓN
# ================================
def plot_hybrid_results(train_losses, val_losses, metrics, predictions, targets):
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    fig.suptitle('Hybrid LSTM-GRU Encoder-Decoder - OHLCV Predictions with Derivatives', 
                 fontsize=16, fontweight='bold')
    
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
    
    plt.savefig('hybrid_lstm_gru_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("📈 Plots saved: hybrid_lstm_gru_results.png")

# ================================
# 📊 VISUALIZACIÓN DE DERIVADAS
# ================================
def plot_derivatives_analysis(df, save_path='derivatives_analysis.png'):
    """
    Crea visualizaciones de las derivadas y correlaciones
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('Price-Volume Derivatives and Correlations Analysis', fontsize=16, fontweight='bold')
    
    sample = min(500, len(df))
    df_plot = df.iloc[-sample:].copy()
    
    # 1. Derivadas de precio
    ax1 = axes[0, 0]
    ax1.plot(df_plot.index, df_plot['price_first_deriv'], label='1st Derivative', alpha=0.7)
    ax1.plot(df_plot.index, df_plot['price_first_deriv_ma5'], label='1st Deriv (MA5)', linewidth=2)
    ax1.set_title('Price First Derivative (Velocity)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Aceleración de precio
    ax2 = axes[0, 1]
    ax2.plot(df_plot.index, df_plot['price_second_deriv'], label='2nd Derivative', alpha=0.7)
    ax2.plot(df_plot.index, df_plot['price_second_deriv_ma5'], label='2nd Deriv (MA5)', linewidth=2)
    ax2.set_title('Price Second Derivative (Acceleration)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Derivadas de volumen
    ax3 = axes[1, 0]
    ax3.plot(df_plot.index, df_plot['volume_first_deriv'], label='1st Derivative', alpha=0.7)
    ax3.plot(df_plot.index, df_plot['volume_first_deriv_ma5'], label='1st Deriv (MA5)', linewidth=2)
    ax3.set_title('Volume First Derivative (Velocity)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Aceleración de volumen
    ax4 = axes[1, 1]
    ax4.plot(df_plot.index, df_plot['volume_second_deriv'], label='2nd Derivative', alpha=0.7)
    ax4.plot(df_plot.index, df_plot['volume_second_deriv_ma5'], label='2nd Deriv (MA5)', linewidth=2)
    ax4.set_title('Volume Second Derivative (Acceleration)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Correlaciones precio-volumen
    ax5 = axes[2, 0]
    ax5.plot(df_plot.index, df_plot['price_volume_corr_10'], label='Corr 10', alpha=0.7)
    ax5.plot(df_plot.index, df_plot['price_volume_corr_20'], label='Corr 20', linewidth=2)
    ax5.plot(df_plot.index, df_plot['price_volume_corr_50'], label='Corr 50', linewidth=2)
    ax5.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax5.set_title('Price-Volume Correlation (Rolling)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Momentum conjunto
    ax6 = axes[2, 1]
    ax6.plot(df_plot.index, df_plot['joint_momentum'], label='Joint Momentum', alpha=0.7)
    ax6.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax6.fill_between(df_plot.index, 0, df_plot['joint_momentum'], 
                      where=(df_plot['joint_momentum'] > 0), alpha=0.3, color='green', label='Positive')
    ax6.fill_between(df_plot.index, 0, df_plot['joint_momentum'], 
                      where=(df_plot['joint_momentum'] < 0), alpha=0.3, color='red', label='Negative')
    ax6.set_title('Joint Price-Volume Momentum')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📈 Derivatives analysis plot saved: {save_path}")

def plot_results(train_losses, val_losses, metrics):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('LSTM-GRU Optimized Results', fontsize=14, fontweight='bold')
    
    # Loss
    axes[0].plot(train_losses, label='Train', linewidth=2)
    axes[0].plot(val_losses, label='Val', linewidth=2)
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # R²
    names = list(metrics.keys())
    r2_scores = [metrics[n]['R2'] for n in names]
    axes[1].bar(names, r2_scores)
    axes[1].set_title('R² Scores')
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Accuracy
    acc_scores = [metrics[n]['Direction_Accuracy'] for n in names]
    axes[2].bar(names, acc_scores)
    axes[2].set_title('Direction Accuracy (%)')
    axes[2].axhline(y=50, color='r', linestyle='--', alpha=0.5)
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('optimized_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("📈 Plot saved: optimized_results.png")

# ================================
# 🚀 MAIN
# ================================
def main():
    try:
        print("\n" + "="*70)
        print("  🚀 LSTM-GRU OPTIMIZADO CON CHECKPOINTING")
        print("="*70)
        print("\n⚡ OPTIMIZACIONES:")
        print("   • Arquitectura reducida: 1 layer LSTM + 1 layer GRU")
        print("   • Secuencias más cortas: 60 timesteps")
        print("   • Batch size aumentado: 128")
        print("   • Menos épocas: 100")
        print("   • Features esenciales solamente")
        print("   • Checkpointing automático cada 5 épocas")
        print("   • Recuperación de entrenamiento interrumpido")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n🖥️ Device: {device}")
        
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
        scalers, feature_cols, target_cols = prepare_dataset(df)
        
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
            train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
        )
        
        # Create model
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
        
        # Train
        start_time = time.time()
        train_losses, val_losses = train_model(model, train_loader, val_loader, device)
        training_time = time.time() - start_time
        
        print(f"\n⏱️ Training time: {training_time/60:.1f} min")
        
        # Evaluate
        predictions, targets, metrics = evaluate_model(
            model, test_loader, scalers, target_cols, device
        )
        
        # Results
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
        
        # Save
        model_dir = 'OPTIMIZED_MODELS'
        os.makedirs(model_dir, exist_ok=True)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': vars(Config),
            'metrics': metrics,
            'feature_cols': feature_cols,
            'target_cols': target_cols
        }, f'{model_dir}/optimized_lstm_gru.pth')
        
        joblib.dump(scalers['input'], f'{model_dir}/scaler_input.pkl')
        joblib.dump(scalers['output_price'], f'{model_dir}/scaler_output_price.pkl')
        joblib.dump(scalers['output_volume'], f'{model_dir}/scaler_output_volume.pkl')
        
        print(f"\n💾 Model saved in '{model_dir}/'")
        print("\n" + "="*70)
        print("  ✅ PROCESS COMPLETE")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Entrenamiento interrumpido manualmente")
        print("💾 El checkpoint se guardó automáticamente")
        print("🔄 Ejecuta de nuevo para reanudar desde el último checkpoint")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
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
