"""
PRUEBA DE CORDURA: ¿Tu modelo supera un baseline naive?

Este script compara tu modelo LSTM con 3 baselines simples:
1. Naive: predicción = precio_actual
2. Moving Average: predicción = promedio últimas 5 velas
3. Last Change: predicción = precio_actual + (cambio_promedio_últimas_5)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import joblib
import json

def calculate_metrics(y_true, y_pred, label):
    """Calcula métricas para comparación"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    # 🆕 MÉTRICA DIRECCIONAL (la más importante para trading)
    direction_correct = np.sum(np.sign(y_pred - y_true[:-1]) == np.sign(y_true[1:] - y_true[:-1]))
    direction_accuracy = (direction_correct / (len(y_true) - 1)) * 100
    
    print(f"\n📊 {label}:")
    print(f"   MAE:  ${mae:.4f}")
    print(f"   RMSE: ${rmse:.4f}")
    print(f"   R²:   {r2:.4f}")
    print(f"   MAPE: {mape:.2f}%")
    print(f"   🎯 Dirección correcta: {direction_accuracy:.1f}%")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'direction_accuracy': direction_accuracy
    }

def main():
    print("="*70)
    print("  🔬 PRUEBA DE CORDURA: LSTM vs BASELINES")
    print("="*70)
    
    # 1. Cargar datos
    print("\n📂 Cargando datos...")
    df = pd.read_csv('ADAUSD_1h_data.csv')
    df['time'] = pd.to_datetime(df['time'])
    
    # Usar últimos 20% como test (igual que en entrenamiento)
    test_size = int(len(df) * 0.15)
    df_test = df.tail(test_size).reset_index(drop=True)
    
    print(f"   Test set: {len(df_test)} velas")
    print(f"   Rango: {df_test['time'].min()} → {df_test['time'].max()}")
    
    # 2. Obtener predicciones del modelo LSTM
    print("\n🧠 Cargando predicciones del modelo LSTM...")
    
    # Cargar modelo y hacer predicciones
    from adausd_lstm import MultiOutputLSTM
    
    with open('ADAUSD_MODELS/config_1h.json', 'r') as f:
        config = json.load(f)
    
    seq_len = config['seq_len']
    
    scaler_in = joblib.load('ADAUSD_MODELS/scaler_input_1h.pkl')
    scaler_out = joblib.load('ADAUSD_MODELS/scaler_output_1h.pkl')
    
    checkpoint = torch.load('ADAUSD_MODELS/adausd_lstm_1h.pth', 
                           map_location=torch.device('cpu'),
                           weights_only=False)
    
    model_config = checkpoint.get('config', {})
    model = MultiOutputLSTM(
        input_size=4,
        hidden_size=model_config.get('hidden', 192),
        num_layers=model_config.get('layers', 2),
        output_size=3,
        dropout=0.35
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Generar predicciones LSTM
    lstm_preds = []
    actual_values = []
    
    for i in range(seq_len, len(df_test)):
        # Secuencia de entrada
        inp = df_test.iloc[i-seq_len:i][['open', 'high', 'low', 'close']].values
        inp_scaled = scaler_in.transform(inp)
        X = torch.FloatTensor(inp_scaled).unsqueeze(0)
        
        # Predicción
        with torch.no_grad():
            pred = model(X)
        
        pred_denorm = scaler_out.inverse_transform(pred.numpy())[0]
        lstm_preds.append(pred_denorm[2])  # Close
        
        # Valor real
        actual_values.append(df_test.iloc[i]['close'])
    
    lstm_preds = np.array(lstm_preds)
    actual_values = np.array(actual_values)
    
    print(f"   ✅ {len(lstm_preds)} predicciones generadas")
    
    # 3. BASELINE 1: Naive (predicción = precio_actual)
    print("\n🎯 Generando BASELINE 1: Naive...")
    naive_preds = df_test.iloc[seq_len-1:-1]['close'].values
    
    # 4. BASELINE 2: Moving Average
    print("🎯 Generando BASELINE 2: Moving Average...")
    ma_window = 5
    ma_preds = df_test['close'].rolling(window=ma_window).mean().iloc[seq_len-1:-1].values
    
    # 5. BASELINE 3: Last Change Extrapolation
    print("🎯 Generando BASELINE 3: Last Change...")
    last_change_preds = []
    for i in range(seq_len, len(df_test)):
        # Cambio promedio de últimas 5 velas
        recent = df_test.iloc[i-5:i]['close'].values
        avg_change = np.mean(np.diff(recent))
        pred = df_test.iloc[i-1]['close'] + avg_change
        last_change_preds.append(pred)
    last_change_preds = np.array(last_change_preds)
    
    # 6. COMPARAR TODOS
    print("\n" + "="*70)
    print("  📊 COMPARACIÓN DE MODELOS")
    print("="*70)
    
    metrics_lstm = calculate_metrics(actual_values, lstm_preds, "🧠 LSTM (tu modelo)")
    metrics_naive = calculate_metrics(actual_values, naive_preds, "📌 BASELINE 1: Naive")
    metrics_ma = calculate_metrics(actual_values, ma_preds, "📊 BASELINE 2: Moving Avg")
    metrics_lc = calculate_metrics(actual_values, last_change_preds, "📈 BASELINE 3: Last Change")
    
    # 7. VEREDICTO
    print("\n" + "="*70)
    print("  🏆 VEREDICTO FINAL")
    print("="*70)
    
    # Comparar R²
    r2_improvement = metrics_lstm['r2'] - metrics_naive['r2']
    direction_improvement = metrics_lstm['direction_accuracy'] - metrics_naive['direction_accuracy']
    
    print(f"\n🔬 Mejora sobre Naive:")
    print(f"   R² improvement: {r2_improvement:+.4f}")
    print(f"   Dirección improvement: {direction_improvement:+.1f}%")
    
    if r2_improvement < 0.01 and direction_improvement < 5:
        print("\n🚨 **VEREDICTO: Modelo NO es útil**")
        print("   ❌ Tu LSTM apenas supera una predicción naive")
        print("   ❌ No hay alpha real para trading")
        print("\n💡 RECOMENDACIONES:")
        print("   1. Cambia a predicción de 4-12h adelante (no 1h)")
        print("   2. Predice el CAMBIO porcentual, no el precio absoluto")
        print("   3. Añade features externos (volumen, orden book, sentimiento)")
        print("   4. Usa clasificación (UP/DOWN) en vez de regresión")
    
    elif r2_improvement < 0.03 and direction_improvement < 10:
        print("\n⚠️ **VEREDICTO: Modelo marginalmente útil**")
        print("   🟡 Supera ligeramente el baseline")
        print("   🟡 Podría ser útil pero necesita mejoras")
        print("\n💡 RECOMENDACIONES:")
        print("   1. Optimiza para dirección, no para MAE/RMSE")
        print("   2. Añade stop-loss dinámico basado en volatilidad")
        print("   3. Filtra señales con baja confianza")
    
    else:
        print("\n✅ **VEREDICTO: Modelo útil**")
        print("   ✅ Supera significativamente el baseline")
        print("   ✅ Tiene potencial para trading real")
        print("\n💡 PRÓXIMOS PASOS:")
        print("   1. Backtest con costos de transacción")
        print("   2. Walk-forward analysis")
        print("   3. Paper trading por 1 semana antes de live")
    
    # 8. Guardar comparación
    comparison = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'lstm': {k: float(v) for k, v in metrics_lstm.items()},
        'naive': {k: float(v) for k, v in metrics_naive.items()},
        'moving_avg': {k: float(v) for k, v in metrics_ma.items()},
        'last_change': {k: float(v) for k, v in metrics_lc.items()},
        'improvement_over_naive': {
            'r2': float(r2_improvement),
            'direction': float(direction_improvement)
        }
    }
    
    with open('baseline_comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n✅ Comparación guardada en: baseline_comparison.json")
    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
