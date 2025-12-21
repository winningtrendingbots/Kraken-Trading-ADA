import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torch.utils.data import Dataset, DataLoader
import os
import time
import json
import joblib
import yfinance as yf
from tqdm.auto import tqdm
import requests
from datetime import datetime

# Configuración de Telegram
TELEGRAM_API = os.environ.get('TELEGRAM_API', '')
CHAT_ID = os.environ.get('CHAT_ID', '')

def send_telegram_message(message):
    if not TELEGRAM_API or not CHAT_ID:
        print("⚠️ Telegram no configurado")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_API}/sendMessage"
        data = {
            'chat_id': CHAT_ID,
            'text': message,
            'parse_mode': 'Markdown'
        }
        response = requests.post(url, data=data, timeout=10)
        print(f"📱 Telegram: {response.status_code}")
    except Exception as e:
        print(f"⚠️ Error Telegram: {e}")

def download_adausd(interval='1h', path='ADAUSD_1h_data.csv'):
    print("="*70)
    print(f"  DESCARGA ADAUSD - {interval.upper()}")
    print("="*70 + "\n")

    if os.path.exists(path):
        df_exist = pd.read_csv(path)
        df_exist['time'] = pd.to_datetime(df_exist['time'])
        
        if df_exist['time'].dt.tz is not None:
            now = pd.Timestamp.now(tz='UTC')
        else:
            now = pd.Timestamp.now()
        
        print(f"📂 Archivo existente: {len(df_exist):,} velas")
        print(f"📅 Rango: {df_exist['time'].min()} → {df_exist['time'].max()}")

        diff_h = (now - df_exist['time'].max()).total_seconds() / 3600
        
        if (interval == '1d' and diff_h < 24) or (interval == '1h' and diff_h < 1):
            print(f"✅ Datos actualizados (hace {diff_h:.1f}h)\n")
            return df_exist
        print(f"🔄 Actualizando (hace {diff_h:.1f}h)...\n")
    
    ticker = yf.Ticker("ADA-USD")
    df_new = ticker.history(period="2y", interval=interval)
    df_new = df_new.reset_index()
    df_new.columns = [str(c).lower() for c in df_new.columns]
    df_new.rename(columns={'date': 'time', 'datetime': 'time'}, inplace=True)
    
    if 'time' in df_new.columns:
        df_new['time'] = pd.to_datetime(df_new['time']).dt.tz_localize(None)
    
    df_new = df_new[['time', 'open', 'high', 'low', 'close']]

    if 'df_exist' in locals():
        df_exist['time'] = df_exist['time'].dt.tz_localize(None) if df_exist['time'].dt.tz is not None else df_exist['time']
        
        df = pd.concat([df_exist, df_new], ignore_index=True)
        df.sort_values('time', inplace=True)
        df.drop_duplicates('time', keep='last', inplace=True)
        df.reset_index(drop=True, inplace=True)
        print(f"✅ {len(df) - len(df_exist)} velas nuevas")
    else:
        df = df_new

    df.to_csv(path, index=False)
    print(f"\n✅ Guardado: {len(df):,} velas")
    return df

class ForexDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MultiOutputLSTM(nn.Module):
    """Versión con BatchNorm - cambio mínimo respecto a tu código"""
    def __init__(self, input_size=4, hidden_size=192, num_layers=2,
                 output_size=3, dropout=0.35):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, 
                           dropout=dropout if num_layers > 1 else 0)
        
        # ✅ NUEVO: BatchNorm después de LSTM
        self.bn1 = nn.BatchNorm1d(hidden_size)
        
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        
        # ✅ NUEVO: BatchNorm intermedia
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        
        x = self.bn1(x)  # ✅ Normalizar
        x = self.fc1(x)
        x = self.bn2(x)  # ✅ Normalizar
        x = self.relu(x)
        x = self.dropout(x)
        
        return self.fc2(x)

class EarlyStopping:
    def __init__(self, patience=10):
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

def prepare_data_CORRECTED(df, seq_len=60, train_size=0.75, val_size=0.15):
    """
    Preparación de datos SIN data leakage
    - Fit scaler SOLO en train
    - Predicción correcta (i->i+1, no i->i+2)
    """
    print(f"\n{'='*70}")
    print("  PREPARANDO DATOS (SIN LEAKAGE)")
    print("="*70)
    print(f"Input: OHLC (4) → Output: HLC (3) | Secuencia: {seq_len} velas\n")

    # 1. Dividir PRIMERO en train/val/test (temporalmente)
    total = len(df)
    train_end = int(total * train_size)
    val_end = int(total * (train_size + val_size))
    
    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()
    
    print(f"📊 División temporal:")
    print(f"   Train: {len(df_train):,} velas ({df_train['time'].min()} → {df_train['time'].max()})")
    print(f"   Val:   {len(df_val):,} velas ({df_val['time'].min()} → {df_val['time'].max()})")
    print(f"   Test:  {len(df_test):,} velas ({df_test['time'].min()} → {df_test['time'].max()})\n")

    # 2. Extraer features y targets
    inp_train = df_train[['open', 'high', 'low', 'close']].values
    out_train = df_train[['high', 'low', 'close']].values
    
    inp_val = df_val[['open', 'high', 'low', 'close']].values
    out_val = df_val[['high', 'low', 'close']].values
    
    inp_test = df_test[['open', 'high', 'low', 'close']].values
    out_test = df_test[['high', 'low', 'close']].values

    # 3. FIT scaler SOLO en train
    scaler_in = MinMaxScaler()
    scaler_out = MinMaxScaler()
    
    inp_train_scaled = scaler_in.fit_transform(inp_train)        # ✅ FIT solo train
    out_train_scaled = scaler_out.fit_transform(out_train)       # ✅ FIT solo train
    
    inp_val_scaled = scaler_in.transform(inp_val)                # ✅ TRANSFORM val
    out_val_scaled = scaler_out.transform(out_val)               # ✅ TRANSFORM val
    
    inp_test_scaled = scaler_in.transform(inp_test)              # ✅ TRANSFORM test
    out_test_scaled = scaler_out.transform(out_test)             # ✅ TRANSFORM test

    # 4. Crear secuencias
    def create_sequences(inp_scaled, out_scaled, seq_len):
        X, y = [], []
        # ✅ CORRECCIÓN: Predecir i (no i+1)
        for i in range(seq_len, len(inp_scaled)):
            X.append(inp_scaled[i-seq_len:i])    # Velas [i-seq_len, ..., i-1]
            y.append(out_scaled[i, :])            # Predice vela i (la SIGUIENTE)
        return np.array(X), np.array(y)
    
    print("🔄 Creando secuencias...")
    X_train, y_train = create_sequences(inp_train_scaled, out_train_scaled, seq_len)
    X_val, y_val = create_sequences(inp_val_scaled, out_val_scaled, seq_len)
    X_test, y_test = create_sequences(inp_test_scaled, out_test_scaled, seq_len)
    
    print(f"✅ Train: X={X_train.shape}, y={y_train.shape}")
    print(f"✅ Val:   X={X_val.shape}, y={y_val.shape}")
    print(f"✅ Test:  X={X_test.shape}, y={y_test.shape}\n")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler_in, scaler_out

def calc_metrics(y_true, y_pred):
    metrics = {}
    for i, label in enumerate(['High', 'Low', 'Close']):
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        mape = np.mean(np.abs((y_true[:, i] - y_pred[:, i]) / (y_true[:, i] + 1e-8))) * 100
        metrics[label] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}
    return metrics

def train_model(model, train_loader, val_loader, epochs, lr, device, patience):
    print(f"\n{'='*70}")
    print("  ENTRENANDO MODELO")
    print("="*70)
    print(f"Epochs: {epochs} | LR: {lr} | Device: {device} | Patience: {patience}\n")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)  
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    early_stop = EarlyStopping(patience)

    train_losses, val_losses, lrs = [], [], []
    best_state = None
    best_val = float('inf')

    model.to(device)
    for epoch in tqdm(range(epochs), desc="Progreso"):
        lrs.append(optimizer.param_groups[0]['lr'])

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

def evaluate(model, test_loader, scaler_out, device):
    print(f"\n{'='*70}")
    print("  EVALUANDO MODELO")
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

    metrics = calc_metrics(act_denorm, pred_denorm)

    print("📊 MÉTRICAS:\n")
    for label in ['High', 'Low', 'Close']:
        m = metrics[label]
        print(f"   {label}:")
        print(f"      MAE: ${m['MAE']:.2f} | RMSE: ${m['RMSE']:.2f}")
        print(f"      R²: {m['R2']:.4f} | MAPE: {m['MAPE']:.2f}%\n")

    return preds, acts, metrics, pred_denorm, act_denorm

def plot_results(train_l, val_l, lrs, pred_d, act_d, metrics, path):
    """Gráficas mejoradas con métricas visibles"""
    def smooth(data, w=0.85):
        s, last = [], data[0]
        for p in data:
            val = last * w + (1 - w) * p
            s.append(val)
            last = val
        return s

    fig = plt.figure(figsize=(24, 16))
    fig.suptitle('ADAUSD Multi-Output LSTM - Resultados Completos', 
                 fontsize=18, fontweight='bold', y=0.995)

    labels = ['High', 'Low', 'Close']
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']

    # FILA 1: Training, LR, Loss
    ax1 = plt.subplot(4, 3, 1)
    ax1.plot(smooth(train_l), 'b-', linewidth=2, label='Train', alpha=0.8)
    ax1.plot(smooth(val_l), 'r-', linewidth=2, label='Val', alpha=0.8)
    ax1.set_title('Training History', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')
    ax1.grid(alpha=0.3)
    ax1.text(0.02, 0.98, f'Best Val Loss: {min(val_l):.6f}', 
             transform=ax1.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax2 = plt.subplot(4, 3, 2)
    ax2.plot(lrs, color='purple', linewidth=2)
    ax2.set_yscale('log')
    ax2.set_title('Learning Rate Schedule', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.grid(alpha=0.3)

    ax3 = plt.subplot(4, 3, 3)
    ax3.plot(smooth(train_l), 'b-', linewidth=2, label='Train', alpha=0.8)
    ax3.plot(smooth(val_l), 'r-', linewidth=2, label='Val', alpha=0.8)
    ax3.set_yscale('log')
    ax3.set_title('Loss (Log Scale)', fontweight='bold', fontsize=12)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss (log)')
    ax3.legend(loc='upper right')
    ax3.grid(alpha=0.3)

    n = min(500, len(pred_d))

    # FILA 2: Predicciones con métricas
    for i, (lbl, col) in enumerate(zip(labels, colors)):
        ax = plt.subplot(4, 3, 4 + i)
        ax.plot(act_d[:n, i], 'k-', linewidth=1.5, alpha=0.7, label='Real')
        ax.plot(pred_d[:n, i], color=col, linewidth=1.5, label='Predicción', alpha=0.8)
        ax.fill_between(range(n), act_d[:n, i], pred_d[:n, i], alpha=0.2, color=col)
        ax.set_title(f'{lbl} - Predicciones vs Real', fontweight='bold', fontsize=12)
        ax.set_xlabel('Samples')
        ax.set_ylabel('Price ($)')
        ax.legend(loc='upper left')
        ax.grid(alpha=0.3)
        
        # Añadir métricas como texto
        m = metrics[lbl]
        metrics_text = f"MAE: ${m['MAE']:.2f}\nRMSE: ${m['RMSE']:.2f}\nR²: {m['R2']:.4f}\nMAPE: {m['MAPE']:.2f}%"
        ax.text(0.98, 0.02, metrics_text, transform=ax.transAxes, 
                fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor=col, alpha=0.3))

    # FILA 3: Scatter plots con línea de regresión
    for i, (lbl, col) in enumerate(zip(labels, colors)):
        ax = plt.subplot(4, 3, 7 + i)
        ax.scatter(act_d[:, i], pred_d[:, i], alpha=0.5, s=10, c=col, edgecolors='none')
        mn, mx = act_d[:, i].min(), act_d[:, i].max()
        ax.plot([mn, mx], [mn, mx], 'r--', linewidth=2, alpha=0.7, label='Ideal')
        
        # Añadir línea de regresión
        z = np.polyfit(act_d[:, i], pred_d[:, i], 1)
        p = np.poly1d(z)
        ax.plot([mn, mx], p([mn, mx]), 'b-', linewidth=2, alpha=0.7, 
                label=f'Fit: y={z[0]:.3f}x+{z[1]:.1f}')
        
        ax.set_title(f'{lbl} - Scatter Plot', fontweight='bold', fontsize=12)
        ax.set_xlabel('Precio Real ($)')
        ax.set_ylabel('Precio Predicho ($)')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(alpha=0.3)
        
        # R² en el gráfico
        m = metrics[lbl]
        ax.text(0.98, 0.02, f"R² = {m['R2']:.4f}", transform=ax.transAxes,
                fontsize=11, fontweight='bold', verticalalignment='bottom', 
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    # FILA 4: Distribución de errores con estadísticas
    for i, (lbl, col) in enumerate(zip(labels, colors)):
        ax = plt.subplot(4, 3, 10 + i)
        err = pred_d[:, i] - act_d[:, i]
        
        n_bins = 60
        counts, bins, patches = ax.hist(err, bins=n_bins, alpha=0.7, color=col, 
                                        edgecolor='black', linewidth=0.5)
        
        # Líneas de referencia
        ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Cero')
        ax.axvline(err.mean(), color='orange', linestyle='--', linewidth=2, 
                   alpha=0.7, label=f'Media: ${err.mean():.2f}')
        ax.axvline(np.median(err), color='green', linestyle='--', linewidth=2,
                   alpha=0.7, label=f'Mediana: ${np.median(err):.2f}')
        
        ax.set_title(f'{lbl} - Distribución de Errores', fontweight='bold', fontsize=12)
        ax.set_xlabel('Error ($)')
        ax.set_ylabel('Frecuencia')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(alpha=0.3, axis='y')
        
        # Estadísticas
        stats_text = f"σ: ${err.std():.2f}\nMin: ${err.min():.2f}\nMax: ${err.max():.2f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"📈 Gráficas mejoradas: {path}\n")

if __name__ == "__main__":
    try:
        print("\n" + "="*70)
        print("  PIPELINE COMPLETO - VERSIÓN MEJORADA")
        print("="*70 + "\n")

        # CONFIGURACIÓN
        """ había posibles problemas de sobreajusto con los errores
        INTERVAL = '1h'
        SEQ_LEN = 60
        HIDDEN = 128
        LAYERS = 2
        DROPOUT = 0.4
        BATCH = 128
        EPOCHS = 150
        LR = 0.001
        PATIENCE = 15
        INTERVAL = '1h'
        """
        
        INTERVAL = '1h'
        SEQ_LEN = 72       # 🔼 3 días exactos
        HIDDEN = 160       # 🔼 Más capacidad
        LAYERS = 3         # 🔼 Más profundidad
        DROPOUT = 0.35     # 🔽 Menos dropout
        BATCH = 96         # 🔽 Batches más pequeños
        EPOCHS = 180       # ✅ Similar
        LR = 0.0012        # 🔼 Learning rate mayor
        PATIENCE = 15      # ✅ Mantener
        """
        SEQ_LEN = 90       # 🔼 Más contexto histórico (3.75 días)
        HIDDEN = 96        # 🔽 Reducir complejidad
        LAYERS = 2         # ✅ Mantener
        DROPOUT = 0.45     # 🔼 Más regularización
        BATCH = 128        # ✅ Mantener
        EPOCHS = 200       # 🔼 Más tiempo con patience
        LR = 0.0008        # 🔽 Learning rate más bajo
        PATIENCE = 20      # 🔼 Más paciencia
        """

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Device: {device}\n")

        # 1. Descargar
        df = download_adausd(interval='1h', path='ADAUSD_1h_data.csv')
    
        # 2. Preparar datos
        (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler_in, scaler_out = \
            prepare_data_CORRECTED(df, seq_len=60, train_size=0.75, val_size=0.15)
        
        # 3. Loaders
        train_loader = DataLoader(ForexDataset(X_train, y_train), BATCH, shuffle=True)
        val_loader = DataLoader(ForexDataset(X_val, y_val), BATCH, shuffle=False)
        test_loader = DataLoader(ForexDataset(X_test, y_test), BATCH, shuffle=False)

        # 4. Modelo
        model = MultiOutputLSTM(4, HIDDEN, LAYERS, 3, DROPOUT)
        params = sum(p.numel() for p in model.parameters())
        print(f"🧠 Modelo: {params:,} parámetros\n")

        # 5. Entrenar
        start = time.time()
        train_l, val_l, lrs = train_model(model, train_loader, val_loader, EPOCHS, LR, device, PATIENCE)

        # ============================================================================
        # 🔍 PEGA AQUÍ EL CÓDIGO DE DIAGNÓSTICO (REEMPLAZA LA LÍNEA 6 ORIGINAL)
        # ============================================================================
        
        # ❌ ELIMINA ESTA LÍNEA ORIGINAL:
        # preds, acts, metrics, pred_d, act_d = evaluate(model, test_loader, scaler_out, device)
        
        # ✅ REEMPLÁZALA CON ESTO:
        
        print("\n" + "="*70)
        print("  📊 EVALUACIÓN COMPLETA (Train/Val/Test)")
        print("="*70 + "\n")
        
        # 6a. Evaluar en TRAIN
        print("🔄 Evaluando en Train...")
        train_loader_eval = DataLoader(ForexDataset(X_train, y_train), 
                                       BATCH, shuffle=False)
        _, _, metrics_train, pred_train, act_train = evaluate(
            model, train_loader_eval, scaler_out, device
        )
        
        # 6b. Evaluar en VAL
        print("🔄 Evaluando en Val...")
        _, _, metrics_val, pred_val, act_val = evaluate(
            model, val_loader, scaler_out, device
        )
        
        # 6c. Evaluar en TEST
        print("🔄 Evaluando en Test...")
        preds, acts, metrics_test, pred_d, act_d = evaluate(
            model, test_loader, scaler_out, device
        )
        
        # ============================================================================
        # 🔍 DIAGNÓSTICO DE OVERFITTING
        # ============================================================================
        
        print("\n" + "="*70)
        print("  🔬 DIAGNÓSTICO DE OVERFITTING")
        print("="*70 + "\n")
        
        # Extraer R² de cada conjunto
        r2_train_h = metrics_train['High']['R2']
        r2_train_l = metrics_train['Low']['R2']
        r2_train_c = metrics_train['Close']['R2']
        
        r2_val_h = metrics_val['High']['R2']
        r2_val_l = metrics_val['Low']['R2']
        r2_val_c = metrics_val['Close']['R2']
        
        r2_test_h = metrics_test['High']['R2']
        r2_test_l = metrics_test['Low']['R2']
        r2_test_c = metrics_test['Close']['R2']
        
        # Calcular gaps
        gap_train_val_c = abs(r2_train_c - r2_val_c)
        gap_train_test_c = abs(r2_train_c - r2_test_c)
        gap_val_test_c = abs(r2_val_c - r2_test_c)
        
        # MOSTRAR TABLA COMPARATIVA
        print("📊 R² SCORES COMPLETOS:\n")
        print("┌─────────┬────────┬────────┬────────┐")
        print("│         │  High  │  Low   │ Close  │")
        print("├─────────┼────────┼────────┼────────┤")
        print(f"│ Train   │ {r2_train_h:.4f} │ {r2_train_l:.4f} │ {r2_train_c:.4f} │")
        print(f"│ Val     │ {r2_val_h:.4f} │ {r2_val_l:.4f} │ {r2_val_c:.4f} │")
        print(f"│ Test    │ {r2_test_h:.4f} │ {r2_test_l:.4f} │ {r2_test_c:.4f} │")
        print("└─────────┴────────┴────────┴────────┘\n")
        
        # ANÁLISIS DE GAPS (enfocado en Close)
        print("🎯 ANÁLISIS DE GAPS (Close):\n")
        print(f"   Train vs Val:  {gap_train_val_c:.4f}")
        print(f"   Train vs Test: {gap_train_test_c:.4f}")
        print(f"   Val vs Test:   {gap_val_test_c:.4f}\n")
        
        # VEREDICTO
        print("🔬 VEREDICTO:\n")
        
        if gap_train_test_c < 0.03:
            status = "✅ EXCELENTE"
            mensaje = "Sin overfitting significativo. Modelo bien regularizado."
            color = "verde"
        elif gap_train_test_c < 0.05:
            status = "✅ BUENO"
            mensaje = "Overfitting mínimo aceptable para trading."
            color = "verde"
        elif gap_train_test_c < 0.08:
            status = "⚠️  MODERADO"
            mensaje = "Overfitting presente. Considera usar NIVEL 2 de regularización."
            color = "amarillo"
        elif gap_train_test_c < 0.12:
            status = "⚠️  ALTO"
            mensaje = "Overfitting significativo. Usa NIVEL 3 de regularización."
            color = "naranja"
        else:
            status = "🚨 SEVERO"
            mensaje = "Overfitting crítico. El modelo NO servirá en producción."
            color = "rojo"
        
        print(f"   Status: {status}")
        print(f"   {mensaje}\n")
        
        # RECOMENDACIÓN ESPECÍFICA
        print("💡 RECOMENDACIÓN:\n")
        
        if gap_train_test_c >= 0.12:
            print("   🔴 Cambia a configuración NIVEL 3:")
            print("      HIDDEN=64, LAYERS=1, DROPOUT=0.5, SEQ_LEN=120")
        elif gap_train_test_c >= 0.08:
            print("   🟡 Cambia a configuración NIVEL 2:")
            print("      HIDDEN=128, LAYERS=2, DROPOUT=0.45, SEQ_LEN=120")
        elif gap_train_test_c >= 0.05:
            print("   🟢 Prueba configuración NIVEL 1:")
            print("      HIDDEN=192, LAYERS=2, DROPOUT=0.35, SEQ_LEN=90")
        else:
            print("   🎉 ¡Configuración actual es buena! No cambies nada.")
        
        print("\n" + "="*70 + "\n")
        
        # ============================================================================
        # FIN DEL CÓDIGO DE DIAGNÓSTICO
        # ============================================================================

        # 7. Graficar (AHORA con métricas correctas)
        # ⚠️ IMPORTANTE: plot_results necesita las métricas de TEST, no de train
        plot_results(train_l, val_l, lrs, pred_d, act_d, metrics_test, 'adausd_results.png')

        # 8. Guardar
        model_dir = 'ADAUSD_MODELS'
        os.makedirs(model_dir, exist_ok=True)

        torch.save({
            'model_state_dict': model.state_dict(),
            'metrics_train': metrics_train,  # ✅ Ahora guardamos TODAS las métricas
            'metrics_val': metrics_val,
            'metrics_test': metrics_test,
            'config': {'seq_len': SEQ_LEN, 'hidden': HIDDEN, 'layers': LAYERS}
        }, f'{model_dir}/adausd_lstm_{INTERVAL}.pth')

        joblib.dump(scaler_in, f'{model_dir}/scaler_input_{INTERVAL}.pkl')
        joblib.dump(scaler_out, f'{model_dir}/scaler_output_{INTERVAL}.pkl')

        with open(f'{model_dir}/config_{INTERVAL}.json', 'w') as f:
            json.dump({
                'interval': INTERVAL,
                'seq_len': SEQ_LEN,
                'metrics_train': {k: {mk: float(mv) for mk, mv in v.items()}
                                 for k, v in metrics_train.items()},
                'metrics_val': {k: {mk: float(mv) for mk, mv in v.items()}
                               for k, v in metrics_val.items()},
                'metrics_test': {k: {mk: float(mv) for mk, mv in v.items()}
                                for k, v in metrics_test.items()}
            }, f, indent=2)

        total_time = time.time() - start
        
        # Mensaje mejorado para Telegram con diagnóstico
        msg = f"""✅ *Entrenamiento Completado*

⏱️ Tiempo: {total_time/60:.1f} min
🧠 Parámetros: {params:,}

🔬 *Diagnóstico Overfitting:*
  • Gap Train-Test: {gap_train_test_c:.4f}
  • Status: {status}

📊 *R² Scores (Close):*
  • Train: {r2_train_c:.4f}
  • Val:   {r2_val_c:.4f}
  • Test:  {r2_test_c:.4f}

📈 *Métricas Test:*
High:
  • MAE: ${metrics_test['High']['MAE']:.2f}
  • R²: {metrics_test['High']['R2']:.4f}
  • MAPE: {metrics_test['High']['MAPE']:.2f}%

Low:
  • MAE: ${metrics_test['Low']['MAE']:.2f}
  • R²: {metrics_test['Low']['R2']:.4f}
  • MAPE: {metrics_test['Low']['MAPE']:.2f}%

Close:
  • MAE: ${metrics_test['Close']['MAE']:.2f}
  • R²: {metrics_test['Close']['R2']:.4f}
  • MAPE: {metrics_test['Close']['MAPE']:.2f}%
"""

        # Guardar summary con diagnóstico
        training_summary = {
            'timestamp': datetime.now().isoformat(),
            'training_completed': True,
            'total_time_minutes': round(total_time / 60, 2),
            'epochs_completed': len(train_l),
            'best_val_loss': float(min(val_l)),
            'overfitting_diagnosis': {
                'gap_train_test': float(gap_train_test_c),
                'status': status,
                'r2_train': float(r2_train_c),
                'r2_val': float(r2_val_c),
                'r2_test': float(r2_test_c)
            },
            'final_metrics': {
                'train': {
                    'high': {k: float(v) for k, v in metrics_train['High'].items()},
                    'low': {k: float(v) for k, v in metrics_train['Low'].items()},
                    'close': {k: float(v) for k, v in metrics_train['Close'].items()}
                },
                'val': {
                    'high': {k: float(v) for k, v in metrics_val['High'].items()},
                    'low': {k: float(v) for k, v in metrics_val['Low'].items()},
                    'close': {k: float(v) for k, v in metrics_val['Close'].items()}
                },
                'test': {
                    'high': {k: float(v) for k, v in metrics_test['High'].items()},
                    'low': {k: float(v) for k, v in metrics_test['Low'].items()},
                    'close': {k: float(v) for k, v in metrics_test['Close'].items()}
                }
            },
            'model_config': {
                'hidden_size': HIDDEN,
                'num_layers': LAYERS,
                'dropout': DROPOUT,
                'seq_len': SEQ_LEN,
                'total_params': params
            }
        }
        
        with open(f'{model_dir}/training_summary.json', 'w') as f:
            json.dump(training_summary, f, indent=2)
        
        with open('LAST_TRAINING.txt', 'w') as f:
            f.write(f"Last training: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
            f.write(f"Best Val Loss: {min(val_l):.8f}\n")
            f.write(f"Gap Train-Test: {gap_train_test_c:.4f} ({status})\n")
            f.write(f"Total time: {total_time/60:.1f} minutes\n")
        
        print(f"✅ Training summary guardado: {model_dir}/training_summary.json")
        print(f"✅ Timestamp guardado: LAST_TRAINING.txt")
        
        print("\n" + "="*70)
        print("✅✅✅  COMPLETADO  ✅✅✅")
        print("="*70)
        print(msg)
        
        send_telegram_message(msg)
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        print(error_msg)
        send_telegram_message(error_msg)
        raise
