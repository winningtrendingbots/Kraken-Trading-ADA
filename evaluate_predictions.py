"""
EVALUADOR DE PREDICCIONES
✅ Compara predicciones con valores reales
✅ Calcula accuracy de High, Low, Close
✅ Actualiza prediction_tracker.csv
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

PREDICTION_TRACKER_FILE = 'prediction_tracker.csv'

def get_actual_prices(timestamp, interval='1h'):
    """
    Obtiene los valores reales de High, Low, Close
    para la vela SIGUIENTE a timestamp
    """
    try:
        # Redondear timestamp a la hora exacta
        timestamp = pd.to_datetime(timestamp)
        timestamp = timestamp.replace(minute=0, second=0, microsecond=0)
        
        # Descargar datos desde timestamp
        ticker = yf.Ticker("ADA-USD")
        
        # Pedir 3 velas para asegurar que tenemos la siguiente
        end_time = timestamp + timedelta(hours=3)
        df = ticker.history(start=timestamp, end=end_time, interval=interval)
        
        if len(df) < 2:
            print(f"   ⚠️  No hay datos suficientes para {timestamp}")
            return None, None, None
        
        # La siguiente vela es el índice 1 (índice 0 es la actual)
        next_candle = df.iloc[1]
        
        actual_high = float(next_candle['High'])
        actual_low = float(next_candle['Low'])
        actual_close = float(next_candle['Close'])
        
        return actual_high, actual_low, actual_close
        
    except Exception as e:
        print(f"   ❌ Error obteniendo datos reales: {e}")
        return None, None, None


def calculate_accuracy(predicted, actual):
    """
    Calcula accuracy como: 100 - MAPE
    MAPE = |predicted - actual| / actual * 100
    """
    if actual == 0:
        return 0.0
    
    mape = abs(predicted - actual) / abs(actual) * 100
    accuracy = max(0, 100 - mape)
    
    return accuracy


def evaluate_predictions():
    """
    Evalúa todas las predicciones que no tienen actual_close
    """
    print("="*70)
    print("  📊 EVALUANDO PREDICCIONES")
    print("="*70 + "\n")
    
    if not pd.io.common.file_exists(PREDICTION_TRACKER_FILE):
        print(f"❌ No existe {PREDICTION_TRACKER_FILE}")
        return
    
    df = pd.read_csv(PREDICTION_TRACKER_FILE)
    
    print(f"📋 Total predicciones: {len(df)}")
    
    # Filtrar predicciones sin evaluar
    df_to_evaluate = df[df['actual_close'].isna()].copy()
    
    if len(df_to_evaluate) == 0:
        print("✅ Todas las predicciones ya evaluadas")
        return
    
    print(f"🔍 Predicciones por evaluar: {len(df_to_evaluate)}")
    
    updated_count = 0
    
    for idx, row in df_to_evaluate.iterrows():
        timestamp = pd.to_datetime(row['timestamp'])
        
        # Solo evaluar predicciones de hace más de 1 hora
        time_diff = datetime.now() - timestamp
        if time_diff.total_seconds() < 3600:
            print(f"   ⏳ Predicción muy reciente: {timestamp}")
            continue
        
        print(f"\n📍 Evaluando predicción de {timestamp}")
        
        # Obtener valores reales
        actual_high, actual_low, actual_close = get_actual_prices(timestamp)
        
        if actual_high is None:
            print(f"   ⚠️  No se pudieron obtener datos reales")
            continue
        
        # Calcular accuracy para cada predicción
        pred_high = row['pred_high']
        pred_low = row['pred_low']
        pred_close = row['pred_close']
        
        acc_high = calculate_accuracy(pred_high, actual_high)
        acc_low = calculate_accuracy(pred_low, actual_low)
        acc_close = calculate_accuracy(pred_close, actual_close)
        
        # Accuracy promedio
        avg_accuracy = (acc_high + acc_low + acc_close) / 3
        
        print(f"   Pred High: ${pred_high:.4f} | Real: ${actual_high:.4f} | Acc: {acc_high:.1f}%")
        print(f"   Pred Low:  ${pred_low:.4f} | Real: ${actual_low:.4f} | Acc: {acc_low:.1f}%")
        print(f"   Pred Close: ${pred_close:.4f} | Real: ${actual_close:.4f} | Acc: {acc_close:.1f}%")
        print(f"   📊 Accuracy promedio: {avg_accuracy:.1f}%")
        
        # Actualizar dataframe
        df.loc[idx, 'actual_high'] = actual_high
        df.loc[idx, 'actual_low'] = actual_low
        df.loc[idx, 'actual_close'] = actual_close
        df.loc[idx, 'pred_accuracy_%'] = round(avg_accuracy, 2)
        
        updated_count += 1
    
    if updated_count > 0:
        # Guardar CSV actualizado
        df.to_csv(PREDICTION_TRACKER_FILE, index=False)
        print(f"\n✅ {updated_count} predicciones evaluadas y guardadas")
        
        # Calcular estadísticas generales
        evaluated_df = df[df['actual_close'].notna()]
        
        if len(evaluated_df) > 0:
            avg_accuracy = evaluated_df['pred_accuracy_%'].mean()
            
            print(f"\n📊 ESTADÍSTICAS GENERALES:")
            print(f"   Total evaluadas: {len(evaluated_df)}")
            print(f"   Accuracy promedio: {avg_accuracy:.2f}%")
            print(f"   Mejor accuracy: {evaluated_df['pred_accuracy_%'].max():.2f}%")
            print(f"   Peor accuracy: {evaluated_df['pred_accuracy_%'].min():.2f}%")
    else:
        print("\n⏳ No hay predicciones listas para evaluar")
    
    print("\n" + "="*70)
    print("  ✅ EVALUACIÓN COMPLETADA")
    print("="*70)


if __name__ == "__main__":
    try:
        evaluate_predictions()
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise
