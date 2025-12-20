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

# Configuración de Telegram
TELEGRAM_API = os.environ.get('TELEGRAM_API', '')
CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

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

def download_ethusd(interval='1h', path='ETHUSD_1h_data.csv'):
    print("="*70)
    print(f"  DESCARGA ETHUSD - {interval.upper()}")
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
    
    ticker = yf.Ticker("ETH-USD")
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
    def __init__(self, input_size=4, hidden_size=256, num_layers=3,
                 output_size=3, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = self.fc1(lstm_out[:, -1, :])
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

def prepare_data(df, seq_len=60):
    print(f"\n{'='*70}")
    print("  PREPARANDO DATOS")
    print("="*70)
    print(f"Input: OHLC (4) → Output: HLC (3) | Secuencia: {seq_len} velas\n")

    inp = df[['open', 'high', 'low', 'close']].values
    out = df[['high', 'low', 'close']].values

    scaler_in = MinMaxScaler()
    scaler_out = MinMaxScaler()
    inp_scaled = scaler_in.fit_transform(inp)
    out_scaled = scaler_out.fit_transform(out)

    X, y = [], []
    for i in tqdm(range(seq_len, len(inp_scaled) - 1), desc="Secuencias"):
        X.append(inp_scaled[i-seq_len:i])
        y.append(out_scaled[i+1, :])

    X, y = np.array(X), np.array(y)
    print(f"\n✅ X: {X.shape} | y: {y.shape}\n")
    return X, y, scaler_in, scaler_out

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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
    fig.suptitle('ETHUSD Multi-Output LSTM - Resultados Completos', 
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
        INTERVAL = '1h'
        SEQ_LEN = 60
        HIDDEN = 256
        LAYERS = 3
        DROPOUT = 0.2
        BATCH = 128
        EPOCHS = 150
        LR = 0.001
        PATIENCE = 20

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Device: {device}\n")

        # 1. Descargar
        df = download_ethusd(interval=INTERVAL, path='ETHUSD_1h_data.csv')

        # 2. Preparar
        X, y, scaler_in, scaler_out = prepare_data(df, SEQ_LEN)

        # 3. Split
        print("="*70)
        print("  DIVISIÓN DE DATOS")
        print("="*70)
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15, shuffle=False)
        print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}\n")

        # 4. Loaders
        train_loader = DataLoader(ForexDataset(X_train, y_train), BATCH, shuffle=True)
        val_loader = DataLoader(ForexDataset(X_val, y_val), BATCH, shuffle=False)
        test_loader = DataLoader(ForexDataset(X_test, y_test), BATCH, shuffle=False)

        # 5. Modelo
        model = MultiOutputLSTM(4, HIDDEN, LAYERS, 3, DROPOUT)
        params = sum(p.numel() for p in model.parameters())
        print(f"🧠 Modelo: {params:,} parámetros\n")

        # 6. Entrenar
        start = time.time()
        train_l, val_l, lrs = train_model(model, train_loader, val_loader, EPOCHS, LR, device, PATIENCE)

        # 7. Evaluar
        preds, acts, metrics, pred_d, act_d = evaluate(model, test_loader, scaler_out, device)

        # 8. Graficar CON MÉTRICAS
        plot_results(train_l, val_l, lrs, pred_d, act_d, metrics, 'ethusd_results.png')

        # 9. Guardar
        model_dir = 'ETHUSD_MODELS'
        os.makedirs(model_dir, exist_ok=True)

        torch.save({
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'config': {'seq_len': SEQ_LEN, 'hidden': HIDDEN, 'layers': LAYERS}
        }, f'{model_dir}/ethusd_lstm_{INTERVAL}.pth')

        joblib.dump(scaler_in, f'{model_dir}/scaler_input_{INTERVAL}.pkl')
        joblib.dump(scaler_out, f'{model_dir}/scaler_output_{INTERVAL}.pkl')

        with open(f'{model_dir}/config_{INTERVAL}.json', 'w') as f:
            json.dump({
                'interval': INTERVAL,
                'seq_len': SEQ_LEN,
                'metrics': {k: {mk: float(mv) for mk, mv in v.items()}
                            for k, v in metrics.items()}
            }, f, indent=2)

        total_time = time.time() - start
        
        # Mensaje mejorado para Telegram
        msg = f"""✅ *Entrenamiento Completado*

⏱️ Tiempo: {total_time/60:.1f} min
🧠 Parámetros: {params:,}

📊 *Métricas Test:*
High:
  • MAE: ${metrics['High']['MAE']:.2f}
  • R²: {metrics['High']['R2']:.4f}
  • MAPE: {metrics['High']['MAPE']:.2f}%

Low:
  • MAE: ${metrics['Low']['MAE']:.2f}
  • R²: {metrics['Low']['R2']:.4f}
  • MAPE: {metrics['Low']['MAPE']:.2f}%

Close:
  • MAE: ${metrics['Close']['MAE']:.2f}
  • R²: {metrics['Close']['R2']:.4f}
  • MAPE: {metrics['Close']['MAPE']:.2f}%
"""
        
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
