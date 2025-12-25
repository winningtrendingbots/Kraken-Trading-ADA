"""
LSTM AGRESIVO - CONFIGURACIÓN PARA CONVERGENCIA
✅ StandardScaler en lugar de RobustScaler
✅ Feature selection (solo las más importantes)
✅ SEQ_LEN reducido para captar patrones locales
✅ Pesos ajustados en la pérdida
✅ Sin restricciones tan agresivas en deltas
✅ Diagnóstico detallado de predicciones
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
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
# 🎛️ CONFIGURACIÓN AGRESIVA
# ================================
class Config:
    # Arquitectura MÁS SIMPLE
    SEQ_LEN = 30                   # ⬇️ Más corto para patrones locales
    HIDDEN_SIZE = 32               # ⬇️ Mucho más simple
    NUM_LAYERS = 1                 # ⬇️ Solo 1 capa
    DROPOUT = 0.2                  # ⬇️ Menos dropout
    BIDIRECTIONAL = False          # ⬇️ Solo forward
    
    # Entrenamiento AGRESIVO
    BATCH_SIZE = 64                # ⬇️ Batches más pequeños
    EPOCHS = 200
    LEARNING_RATE = 0.001          # ⬆️ LR más alto
    WEIGHT_DECAY = 1e-5            # ⬇️ Menos regularización
    PATIENCE = 30
    MIN_DELTA = 1e-6
    GRAD_CLIP = 1.0
    
    # Normalización
    USE_STANDARD_SCALER = True     # ✅ StandardScaler
    
    # Features REDUCIDAS (solo las más importantes)
    SELECTED_FEATURES = [
        'close', 'volume', 'high', 'low',
        'rsi', 'volume_rsi',
        'obv_roc', 'vwap_deviation',
        'price_1st_deriv', 'volume_1st_deriv',
        'volatility', 'volume_adjusted_volatility',
        'bullish_divergence', 'bearish_divergence'
    ]
    
    # Datos
    TRAIN_SIZE = 0.70
    VAL_SIZE = 0.15
    TEST_SIZE = 0.15

def calculate_core_indicators(df):
    """Solo indicadores esenciales"""
    df = df.copy()
    
    # RSI
    def calc_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    df['rsi'] = calc_rsi(df['close'], 14)
    df['volume_rsi'] = calc_rsi(df['volume'], 14)
    
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
    df['obv_roc'] = df['obv'].pct_change(periods=14)
    
    # VWAP
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    df['vwap_deviation'] = (df['close'] - vwap) / vwap * 100
    
    # Derivadas
    df['price_1st_deriv'] = df['close'].diff()
    df['volume_1st_deriv'] = df['volume'].diff()
    
    # Volatilidad
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=20).std()
    df['volume_adjusted_volatility'] = df['volatility'] * (df['volume'] / df['volume'].rolling(window=20).mean())
    
    # Divergencias simples
    price_slope = df['close'].rolling(window=10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else 0
    )
    obv_slope = df['obv'].rolling(window=10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else 0
    )
    
    df['bullish_divergence'] = ((price_slope < 0) & (obv_slope > 0)).astype(int)
    df['bearish_divergence'] = ((price_slope > 0) & (obv_slope < 0)).astype(int)
    
    # Limpieza
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return df

# ================================
# 🧠 MODELO SIMPLIFICADO
# ================================
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=32, output_size=3, dropout=0.2):
        super().__init__()
        
        # LSTM simple (no bidireccional)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        
        # FC muy simple
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        lstm_out, (hidden, _) = self.lstm(x)
        # Usar solo el último hidden state
        out = self.fc(hidden[-1])
        return out

# ================================
# 📦 PREPARACIÓN
# ================================
def prepare_dataset(df):
    print("\n" + "="*70)
    print("  🔧 PREPARACIÓN SIMPLIFICADA")
    print("="*70)
    
    df = calculate_core_indicators(df)
    
    # Deltas
    df['delta_high'] = (df['high'].shift(-1) - df['close']) / df['close']
    df['delta_low'] = (df['low'].shift(-1) - df['close']) / df['close']
    df['delta_close'] = (df['close'].shift(-1) - df['close']) / df['close']
    
    df = df.dropna()
    print(f"📊 Datos: {len(df):,} velas")
    
    # Solo features seleccionadas
    feature_cols = [col for col in Config.SELECTED_FEATURES if col in df.columns]
    target_cols = ['delta_high', 'delta_low', 'delta_close']
    
    print(f"\n🎯 {len(feature_cols)} Características SELECCIONADAS")
    print(f"   {feature_cols}")
    
    # División
    train_end = int(len(df) * Config.TRAIN_SIZE)
    val_end = int(len(df) * (Config.TRAIN_SIZE + Config.VAL_SIZE))
    
    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()
    
    print(f"\n📊 División:")
    print(f"   Train: {len(df_train):,}")
    print(f"   Val:   {len(df_val):,}")
    print(f"   Test:  {len(df_test):,}")
    
    # StandardScaler
    scaler_in = StandardScaler()
    scaler_out = StandardScaler()
    
    print(f"\n🔧 Usando StandardScaler...")
    
    X_train = scaler_in.fit_transform(df_train[feature_cols])
    X_val = scaler_in.transform(df_val[feature_cols])
    X_test = scaler_in.transform(df_test[feature_cols])
    
    y_train = scaler_out.fit_transform(df_train[target_cols])
    y_val = scaler_out.transform(df_val[target_cols])
    y_test = scaler_out.transform(df_test[target_cols])
    
    # Diagnóstico de targets
    print(f"\n📊 Estadísticas de Targets (raw):")
    for col in target_cols:
        vals = df_train[col]
        print(f"   {col:12} mean={vals.mean():.6f}, std={vals.std():.6f}, "
              f"min={vals.min():.6f}, max={vals.max():.6f}")
    
    # Secuencias
    def create_sequences(X, y, seq_len):
        X_seq, y_seq = [], []
        for i in range(seq_len, len(X)):
            X_seq.append(X[i-seq_len:i])
            y_seq.append(y[i-1])
        return np.array(X_seq), np.array(y_seq)
    
    print(f"\n🔄 Secuencias (seq_len={Config.SEQ_LEN})...")
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, Config.SEQ_LEN)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, Config.SEQ_LEN)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, Config.SEQ_LEN)
    
    print(f"✅ Train: X{X_train_seq.shape}, y{y_train_seq.shape}")
    print(f"✅ Val:   X{X_val_seq.shape}, y{y_val_seq.shape}")
    print(f"✅ Test:  X{X_test_seq.shape}, y{y_test_seq.shape}")
    
    return (X_train_seq, y_train_seq), (X_val_seq, y_val_seq), (X_test_seq, y_test_seq), \
           scaler_in, scaler_out, feature_cols, target_cols

# ================================
# 🏋️ ENTRENAMIENTO SIMPLIFICADO
# ================================
class SimpleLoss(nn.Module):
    """Pérdida MSE pura sin restricciones complejas"""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, predictions, targets):
        return self.mse(predictions, targets)

def train_model(model, train_loader, val_loader, device):
    print("\n" + "="*70)
    print("  🏋️ ENTRENAMIENTO SIMPLIFICADO")
    print("="*70)
    
    criterion = SimpleLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    print(f"\n⚙️  Config: LR={Config.LEARNING_RATE}, Batch={Config.BATCH_SIZE}, Patience={Config.PATIENCE}")
    print()
    
    epoch_bar = tqdm(range(Config.EPOCHS), desc="Entrenando", unit="epoch")
    
    for epoch in epoch_bar:
        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Val
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss - Config.MIN_DELTA:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            torch.save(best_model_state, 'best_model.pth')
        else:
            patience_counter += 1
        
        # Update bar
        current_lr = optimizer.param_groups[0]['lr']
        epoch_bar.set_postfix({
            'train': f'{train_loss:.4f}',
            'val': f'{val_loss:.4f}',
            'best': f'{best_val_loss:.4f}',
            'lr': f'{current_lr:.6f}',
            'p': f'{patience_counter}/{Config.PATIENCE}'
        })
        
        # Log detallado
        if (epoch + 1) % 10 == 0:
            print(f"\n📊 Época {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}, "
                  f"Best={best_val_loss:.4f}, LR={current_lr:.6f}, Patience={patience_counter}/{Config.PATIENCE}")
        
        if patience_counter >= Config.PATIENCE:
            print(f"\n⏹️  Early stopping en época {epoch+1}")
            break
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    print(f"\n✅ Completado: Best val loss={best_val_loss:.4f}, Épocas={len(train_losses)}")
    return train_losses, val_losses

# ================================
# 📊 EVALUACIÓN CON DIAGNÓSTICO
# ================================
def evaluate_model(model, test_loader, scaler_out, target_cols, device):
    print("\n" + "="*70)
    print("  📊 EVALUACIÓN Y DIAGNÓSTICO")
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
    
    # DIAGNÓSTICO DETALLADO
    print(f"\n🔍 DIAGNÓSTICO DE PREDICCIONES:")
    for i, col in enumerate(target_cols):
        pred_vals = predictions_denorm[:, i]
        true_vals = targets_denorm[:, i]
        
        print(f"\n{col}:")
        print(f"  Predicciones: mean={pred_vals.mean():.6f}, std={pred_vals.std():.6f}, "
              f"min={pred_vals.min():.6f}, max={pred_vals.max():.6f}")
        print(f"  Real:         mean={true_vals.mean():.6f}, std={true_vals.std():.6f}, "
              f"min={true_vals.min():.6f}, max={true_vals.max():.6f}")
        
        # ¿Está prediciendo solo ~0?
        near_zero = np.abs(pred_vals) < 0.001
        print(f"  Predicciones ~0: {near_zero.sum()}/{len(pred_vals)} ({near_zero.sum()/len(pred_vals)*100:.1f}%)")
    
    # Métricas
    metrics = {}
    print(f"\n📊 MÉTRICAS:")
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
        
        print(f"\n{col}:")
        print(f"  MAE:  {mae:.6f} ({mae*100:.4f}%)")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  R²:   {r2:.4f}")
        print(f"  Acc:  {accuracy:.2f}%")
    
    return predictions_denorm, targets_denorm, metrics

# ================================
# 🎨 VISUALIZACIÓN
# ================================
def plot_results(train_losses, val_losses, metrics, predictions, targets):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('LSTM Simplificado - Diagnóstico', fontsize=16, fontweight='bold')
    
    # Losses
    ax = axes[0, 0]
    ax.plot(train_losses, label='Train', linewidth=2)
    ax.plot(val_losses, label='Val', linewidth=2)
    ax.set_title('Pérdidas')
    ax.set_xlabel('Época')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # R²
    ax = axes[0, 1]
    targets_names = list(metrics.keys())
    r2_scores = [metrics[t]['R2'] for t in targets_names]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax.bar(targets_names, r2_scores, color=colors)
    ax.set_title('R² Score')
    ax.set_ylabel('R²')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, score in zip(bars, r2_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}', ha='center', va='bottom' if score > 0 else 'top')
    
    # Accuracy
    ax = axes[0, 2]
    acc_scores = [metrics[t]['Direction_Accuracy'] for t in targets_names]
    bars = ax.bar(targets_names, acc_scores, color=colors)
    ax.set_title('Accuracy Direccional')
    ax.set_ylabel('Accuracy (%)')
    ax.axhline(y=50, color='r', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, acc in zip(bars, acc_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%', ha='center', va='bottom')
    
    # Scatterplots
    sample_size = min(200, len(predictions))
    for idx, (ax, target_name, color) in enumerate(zip(axes[1], targets_names, colors)):
        ax.scatter(targets[:sample_size, idx], predictions[:sample_size, idx], 
                  alpha=0.6, s=20, color=color)
        ax.plot([targets[:, idx].min(), targets[:, idx].max()],
               [targets[:, idx].min(), targets[:, idx].max()], 'r--', alpha=0.7)
        ax.set_title(f'{target_name}')
        ax.set_xlabel('Real')
        ax.set_ylabel('Predicción')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('adausd_hybrid_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n📈 Gráficas guardadas")

# ================================
# 🚀 MAIN
# ================================
def main():
    try:
        print("\n" + "="*70)
        print("  🚀 LSTM SIMPLIFICADO AGRESIVO")
        print("="*70)
        
        print("\n⚙️  CONFIGURACIÓN AGRESIVA:")
        print(f"   SEQ_LEN: {Config.SEQ_LEN} (corto)")
        print(f"   HIDDEN: {Config.HIDDEN_SIZE} (simple)")
        print(f"   LAYERS: {Config.NUM_LAYERS}")
        print(f"   BIDIRECTIONAL: {Config.BIDIRECTIONAL}")
        print(f"   LR: {Config.LEARNING_RATE} (alto)")
        print(f"   BATCH: {Config.BATCH_SIZE}")
        print(f"   SCALER: StandardScaler")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n🖥️  Device: {device}")
        
        # Datos
        print("\n" + "="*70)
        print("  📥 DESCARGANDO DATOS")
        print("="*70)
        
        ticker = yf.Ticker("ADA-USD")
        df = ticker.history(period="2y", interval="1h")
        
        df = df.reset_index()
        df.columns = [col.lower() for col in df.columns]
        
        for col in df.columns:
            if 'date' in col or 'time' in col:
                df.rename(columns={col: 'time'}, inplace=True)
                break
        
        df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
        print(f"✅ {len(df):,} velas")
        
        # Preparar
        (X_train, y_train), (X_val, y_val), (X_test, y_test), \
        scaler_in, scaler_out, feature_cols, target_cols = prepare_dataset(df)
        
        input_size = len(feature_cols)
        
        # DataLoaders
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
        
        # Modelo
        model = SimpleLSTM(
            input_size=input_size,
            hidden_size=Config.HIDDEN_SIZE,
            output_size=3,
            dropout=Config.DROPOUT
        ).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n🧠 Modelo: {total_params:,} parámetros")
        
        # Entrenar
        start_time = time.time()
        train_losses, val_losses = train_model(model, train_loader, val_loader, device)
        training_time = time.time() - start_time
        
        print(f"\n⏱️  Tiempo: {training_time/60:.1f} min")
        
        # Evaluar
        predictions, targets, metrics = evaluate_model(
            model, test_loader, scaler_out, target_cols, device
        )
        
        # Resultados
        print("\n" + "="*70)
        print("  📈 RESULTADOS")
        print("="*70)
        
        avg_r2 = np.mean([metrics[t]['R2'] for t in metrics.keys()])
        avg_acc = np.mean([metrics[t]['Direction_Accuracy'] for t in metrics.keys()])
        
        print(f"\n📊 Promedios:")
        print(f"   R²: {avg_r2:.4f}")
        print(f"   Accuracy: {avg_acc:.2f}%")
        
        if avg_r2 < 0:
            print(f"\n⚠️  R² SIGUE NEGATIVO")
            print(f"   Esto indica que el problema puede ser:")
            print(f"   1. Los deltas son demasiado pequeños y ruidosos")
            print(f"   2. No hay patrones predecibles en los datos")
            print(f"   3. Se necesita un enfoque completamente diferente")
        
        plot_results(train_losses, val_losses, metrics, predictions, targets)
        
        # Guardar
        model_dir = 'ADAUSD_MODELS'
        os.makedirs(model_dir, exist_ok=True)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_size': input_size,
            'hidden_size': Config.HIDDEN_SIZE,
            'seq_len': Config.SEQ_LEN,
            'feature_cols': feature_cols,
            'target_cols': target_cols,
            'metrics_test': metrics,
            'timestamp': datetime.now().isoformat()
        }, f'{model_dir}/adausd_hybrid_lstm.pth')
        
        joblib.dump(scaler_in, f'{model_dir}/scaler_input_hybrid.pkl')
        joblib.dump(scaler_out, f'{model_dir}/scaler_output_hybrid.pkl')
        
        with open(f'{model_dir}/config_hybrid.json', 'w') as f:
            json.dump({
                'input_size': input_size,
                'hidden_size': Config.HIDDEN_SIZE,
                'seq_len': Config.SEQ_LEN,
                'feature_cols': feature_cols,
                'target_cols': target_cols,
                'metrics_test': metrics,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"\n💾 Guardado en '{model_dir}/'")
        
        msg = f"""
✅ *LSTM Simplificado*

📊 *Resultados:*
   • R²: {avg_r2:.4f}
   • Accuracy: {avg_acc:.2f}%
   • Tiempo: {training_time/60:.1f} min
   • Épocas: {len(train_losses)}

🔧 *Config Agresiva:*
   • SEQ_LEN: {Config.SEQ_LEN}
   • Hidden: {Config.HIDDEN_SIZE}
   • LR: {Config.LEARNING_RATE}
   • StandardScaler usado
"""
        send_telegram(msg)
        
        print("\n" + "="*70)
        print("  ✅ COMPLETADO")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        send_telegram(f"❌ Error: {str(e)}")
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
