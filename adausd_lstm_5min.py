"""
ENTRENAMIENTO LSTM - ADAPTADO PARA 5 MINUTOS
✅ Usa velas de 1 hora como contexto
✅ Predice High, Low, Close de la siguiente vela
✅ Optimizado para ejecución cada 5 minutos
"""

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
import os
import time
import json
import joblib
import yfinance as yf
from tqdm.auto import tqdm
import requests
from datetime import datetime

# Configuración Telegram
TELEGRAM_API = os.environ.get('TELEGRAM_API', '')
CHAT_ID = os.environ.get('CHAT_ID', '')

def send_telegram_message(message):
    if not TELEGRAM_API or not CHAT_ID:
        print("⚠️ Telegram no configurado")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_API}/sendMessage"
        data = {'chat_id': CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
        response = requests.post(url, data=data, timeout=10)
        print(f"📱 Telegram: {response.status_code}")
    except Exception as e:
        print(f"⚠️ Error Telegram: {e}")

def download_adausd(interval='1h', path='ADAUSD_1h_data.csv'):
    """
    🔥 CAMBIO: Descarga velas de 1 HORA (no 1 día)
    """
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
        
        # 🔥 CAMBIO: Actualizar si han pasado más de 1 hora
        if diff_h < 1:
            print(f"✅ Datos actualizados (hace {diff_h:.1f}h)\n")
            return df_exist
        print(f"🔄 Actualizando (hace {diff_h:.1f}h)...\n")
    
    ticker = yf.Ticker("ADA-USD")
    # 🔥 CAMBIO: Pedir más datos para tener suficiente historial
    df_new = ticker.history(period="max", interval=interval)
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

class ForexDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MultiOutputLSTM(nn.Module):
    """
    🔥 MODELO ADAPTADO PARA 5 MINUTOS
    - Input: OHLC de 72 velas (3 días de contexto en velas de 1h)
    - Output: High, Low, Close de la SIGUIENTE vela
    """
    def __init__(self, input_size=4, hidden_size=160, num_layers=3,
                 output_size=3, dropout=0.35):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, 
                           dropout=dropout if num_layers > 1 else 0)
        
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.bn1(x)
        x = self.fc1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)

class EarlyStopping:
    def __init__(self, patience=15):
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

def prepare_data_for_5min(df, seq_len=72, train_size=0.75, val_size=0.15):
    """
    🔥 PREPARACIÓN DE DATOS PARA 5 MINUTOS
    
    - seq_len=72: 3 días de contexto (72 horas)
    - Predicción: High, Low, Close de la SIGUIENTE vela
    """
    print(f"\n{'='*70}")
    print("  PREPARANDO DATOS PARA PREDICCIÓN 5MIN")
    print("="*70)
    print(f"Input: OHLC (4) → Output: HLC (3) | Secuencia: {seq_len} velas (1h)\n")

    # 1. División temporal
    total = len(df)
    train_end = int(total * train_size)
    val_end = int(total * (train_size + val_size))
    
    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()
    
    print(f"📊 División temporal:")
    print(f"   Train: {len(df_train):,} velas")
    print(f"   Val:   {len(df_val):,} velas")
    print(f"   Test:  {len(df_test):,} velas\n")

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
    
    inp_train_scaled = scaler_in.fit_transform(inp_train)
    out_train_scaled = scaler_out.fit_transform(out_train)
    
    inp_val_scaled = scaler_in.transform(inp_val)
    out_val_scaled = scaler_out.transform(out_val)
    
    inp_test_scaled = scaler_in.transform(inp_test)
    out_test_scaled = scaler_out.transform(out_test)

    # 4. Crear secuencias
    def create_sequences(inp_scaled, out_scaled, seq_len):
        X, y = [], []
        for i in range(seq_len, len(inp_scaled)):
            X.append(inp_scaled[i-seq_len:i])
            y.append(out_scaled[i, :])
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
        print(f"      MAE: ${m['MAE']:.4f} | RMSE: ${m['RMSE']:.4f}")
        print(f"      R²: {m['R2']:.4f} | MAPE: {m['MAPE']:.2f}%\n")

    return preds, acts, metrics, pred_denorm, act_denorm

def plot_results(train_l, val_l, lrs, pred_d, act_d, metrics, path):
    """Gráficas de resultados"""
    def smooth(data, w=0.85):
        s, last = [], data[0]
        for p in data:
            val = last * w + (1 - w) * p
            s.append(val)
            last = val
        return s

    fig = plt.figure(figsize=(24, 16))
    fig.suptitle('ADAUSD LSTM - 5min Trading (1h candles)', 
                 fontsize=18, fontweight='bold', y=0.995)

    labels = ['High', 'Low', 'Close']
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']

    # Training History
    ax1 = plt.subplot(4, 3, 1)
    ax1.plot(smooth(train_l), 'b-', linewidth=2, label='Train', alpha=0.8)
    ax1.plot(smooth(val_l), 'r-', linewidth=2, label='Val', alpha=0.8)
    ax1.set_title('Training History', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')
    ax1.grid(alpha=0.3)

    # Learning Rate
    ax2 = plt.subplot(4, 3, 2)
    ax2.plot(lrs, color='purple', linewidth=2)
    ax2.set_yscale('log')
    ax2.set_title('Learning Rate', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('LR')
    ax2.grid(alpha=0.3)

    # Loss Log Scale
    ax3 = plt.subplot(4, 3, 3)
    ax3.plot(smooth(train_l), 'b-', linewidth=2, label='Train', alpha=0.8)
    ax3.plot(smooth(val_l), 'r-', linewidth=2, label='Val', alpha=0.8)
    ax3.set_yscale('log')
    ax3.set_title('Loss (Log)', fontweight='bold', fontsize=12)
    ax3.set_xlabel('Epoch')
    ax3.legend(loc='upper right')
    ax3.grid(alpha=0.3)

    n = min(500, len(pred_d))

    # Predicciones
    for i, (lbl, col) in enumerate(zip(labels, colors)):
        ax = plt.subplot(4, 3, 4 + i)
        ax.plot(act_d[:n, i], 'k-', linewidth=1.5, alpha=0.7, label='Real')
        ax.plot(pred_d[:n, i], color=col, linewidth=1.5, label='Pred', alpha=0.8)
        ax.fill_between(range(n), act_d[:n, i], pred_d[:n, i], alpha=0.2, color=col)
        ax.set_title(f'{lbl} - Predictions', fontweight='bold', fontsize=12)
        ax.set_xlabel('Samples')
        ax.set_ylabel('Price ($)')
        ax.legend(loc='upper left')
        ax.grid(alpha=0.3)
        
        m = metrics[lbl]
        metrics_text = f"MAE: ${m['MAE']:.4f}\nR²: {m['R2']:.4f}\nMAPE: {m['MAPE']:.2f}%"
        ax.text(0.98, 0.02, metrics_text, transform=ax.transAxes, 
                fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor=col, alpha=0.3))

    # Scatter plots
    for i, (lbl, col) in enumerate(zip(labels, colors)):
        ax = plt.subplot(4, 3, 7 + i)
        ax.scatter(act_d[:, i], pred_d[:, i], alpha=0.5, s=10, c=col, edgecolors='none')
        mn, mx = act_d[:, i].min(), act_d[:, i].max()
        ax.plot([mn, mx], [mn, mx], 'r--', linewidth=2, alpha=0.7)
        ax.set_title(f'{lbl} - Scatter', fontweight='bold', fontsize=12)
        ax.set_xlabel('Real ($)')
        ax.set_ylabel('Predicted ($)')
        ax.grid(alpha=0.3)

    # Error distribution
    for i, (lbl, col) in enumerate(zip(labels, colors)):
        ax = plt.subplot(4, 3, 10 + i)
        err = pred_d[:, i] - act_d[:, i]
        ax.hist(err, bins=50, alpha=0.7, color=col, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2)
        ax.set_title(f'{lbl} - Error Distribution', fontweight='bold', fontsize=12)
        ax.set_xlabel('Error ($)')
        ax.set_ylabel('Frequency')
        ax.grid(alpha=0.3, axis='y')

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"📈 Gráficas: {path}\n")

if __name__ == "__main__":
    try:
        print("\n" + "="*70)
        print("  ENTRENAMIENTO LSTM PARA TRADING 5MIN")
        print("="*70 + "\n")

        # 🔥 CONFIGURACIÓN OPTIMIZADA PARA 5 MINUTOS
        INTERVAL = '1h'      # Velas de 1 hora como contexto
        SEQ_LEN = 72         # 3 días de contexto (72 horas)
        HIDDEN = 160         # Capacidad del modelo
        LAYERS = 3           # Profundidad
        DROPOUT = 0.35       # Regularización
        BATCH = 96           # Tamaño de batch
        EPOCHS = 180         # Épocas de entrenamiento
        LR = 0.0012          # Learning rate
        PATIENCE = 15        # Paciencia para early stopping

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Device: {device}\n")

        # 1. Descargar datos
        df = download_adausd(interval=INTERVAL, path='ADAUSD_1h_data.csv')
    
        # 2. Preparar datos
        (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler_in, scaler_out = \
            prepare_data_for_5min(df, seq_len=SEQ_LEN, train_size=0.75, val_size=0.15)
        
        # 3. Loaders
        train_loader = torch.utils.data.DataLoader(ForexDataset(X_train, y_train), BATCH, shuffle=True)
        val_loader = torch.utils.data.DataLoader(ForexDataset(X_val, y_val), BATCH, shuffle=False)
        test_loader = torch.utils.data.DataLoader(ForexDataset(X_test, y_test), BATCH, shuffle=False)

        # 4. Modelo
        model = MultiOutputLSTM(4, HIDDEN, LAYERS, 3, DROPOUT)
        params = sum(p.numel() for p in model.parameters())
        print(f"🧠 Modelo: {params:,} parámetros\n")

        # 5. Entrenar
        start = time.time()
        train_l, val_l, lrs = train_model(model, train_loader, val_loader, EPOCHS, LR, device, PATIENCE)

        # 6. Evaluar
        print("\n" + "="*70)
        print("  📊 EVALUACIÓN COMPLETA")
        print("="*70 + "\n")
        
        preds, acts, metrics_test, pred_d, act_d = evaluate(
            model, test_loader, scaler_out, device
        )

        # 7. Graficar
        plot_results(train_l, val_l, lrs, pred_d, act_d, metrics_test, 'adausd_5min_results.png')

        # 8. Guardar modelo
        model_dir = 'ADAUSD_MODELS'
        os.makedirs(model_dir, exist_ok=True)

        torch.save({
            'model_state_dict': model.state_dict(),
            'metrics_test': metrics_test,
            'config': {
                'seq_len': SEQ_LEN,
                'hidden': HIDDEN,
                'layers': LAYERS,
                'interval': INTERVAL,
                'trading_frequency': '5min'
            }
        }, f'{model_dir}/adausd_lstm_5min.pth')

        joblib.dump(scaler_in, f'{model_dir}/scaler_input_5min.pkl')
        joblib.dump(scaler_out, f'{model_dir}/scaler_output_5min.pkl')

        with open(f'{model_dir}/config_5min.json', 'w') as f:
            json.dump({
                'interval': INTERVAL,
                'seq_len': SEQ_LEN,
                'hidden': HIDDEN,
                'layers': LAYERS,
                'trading_frequency': '5min',
                'metrics_test': {k: {mk: float(mv) for mk, mv in v.items()}
                                for k, v in metrics_test.items()}
            }, f, indent=2)

        total_time = time.time() - start
        
        # Mensaje Telegram
        msg = f"""✅ *Modelo 5min Entrenado*

⏱️ Tiempo: {total_time/60:.1f} min
🧠 Parámetros: {params:,}

📊 *Métricas Test (Close):*
  • MAE: ${metrics_test['Close']['MAE']:.4f}
  • R²: {metrics_test['Close']['R2']:.4f}
  • MAPE: {metrics_test['Close']['MAPE']:.2f}%

⚡ Trading: cada 5 minutos
📈 Contexto: velas 1h ({SEQ_LEN} períodos)
"""
        
        print("\n" + "="*70)
        print("✅✅✅  ENTRENAMIENTO COMPLETADO  ✅✅✅")
        print("="*70)
        print(msg)
        
        send_telegram_message(msg)
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        print(error_msg)
        send_telegram_message(error_msg)
        raise
