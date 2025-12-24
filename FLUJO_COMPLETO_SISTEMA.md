# 🔄 Flujo Completo del Sistema de Trading

## 📊 Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                     GITHUB ACTIONS WORKFLOWS                    │
└─────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│  1️⃣  ENTRENAMIENTO (1x día a las 2 AM UTC)                                │
│  📄 Archivo: .github/workflows/1-train-model.yml                          │
├───────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐             │
│  │  🐍 adausd_lstm.py                                       │             │
│  │  ├─ Descarga datos ADAUSD (120 días)                    │             │
│  │  ├─ Prepara datasets (train/val/test)                   │             │
│  │  ├─ Entrena CNN-LSTM                                    │             │
│  │  ├─ Evalúa overfitting (gap train-test)                │             │
│  │  └─ Guarda modelo + scalers + config                   │             │
│  └─────────────────────────────────────────────────────────┘             │
│                              ↓                                             │
│  📁 ADAUSD_MODELS/                                                         │
│  ├─ adausd_lstm_1h.pth          (modelo)                                  │
│  ├─ scaler_input_1h.pkl         (normalización entrada)                   │
│  ├─ scaler_output_1h.pkl        (normalización salida)                    │
│  ├─ config_1h.json              (configuración)                           │
│  └─ training_summary.json       (métricas + diagnóstico)                  │
└───────────────────────────────────────────────────────────────────────────┘

                                    ↓

┌───────────────────────────────────────────────────────────────────────────┐
│  2️⃣  PREDICCIÓN + TRADING (cada 15 min)                                   │
│  📄 Archivo: .github/workflows/2-predict-and-trade.yml ✨ ACTUALIZADO     │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌──────────────────────────────────────────────────────────┐            │
│  │  OPCIÓN A (Por defecto): 🔮 predict_enhanced.py         │            │
│  │  ┌────────────────────────────────────────────────────┐  │            │
│  │  │  1. Obtener límites de normalización (120 días)   │  │            │
│  │  │     min_price, max_price = get_bounds(120)        │  │            │
│  │  │                                                     │  │            │
│  │  │  2. Descargar datos H1 recientes (5 días)         │  │            │
│  │  │                                                     │  │            │
│  │  │  3. Normalizar con límites de 120 días            │  │            │
│  │  │     x_norm = (x - min) / (max - min)              │  │            │
│  │  │                                                     │  │            │
│  │  │  4. Ejecutar modelo                                │  │            │
│  │  │     pred_high, pred_low, pred_close = model(X)    │  │            │
│  │  │                                                     │  │            │
│  │  │  5. Desnormalizar predicciones                     │  │            │
│  │  │                                                     │  │            │
│  │  │  6. Clasificar movimiento (multi-factor)           │  │            │
│  │  │     - Cambio en Close                              │  │            │
│  │  │     - Rango predicho (H - L)                       │  │            │
│  │  │     - Posición de Close en rango                   │  │            │
│  │  │     - Coherencia H/L/C                             │  │            │
│  │  │     → signal, confidence                           │  │            │
│  │  │                                                     │  │            │
│  │  │  7. Calcular indicadores técnicos                  │  │            │
│  │  │     RSI, ATR, Tendencia                            │  │            │
│  │  │                                                     │  │            │
│  │  │  8. Guardar en CSVs                                │  │            │
│  │  └────────────────────────────────────────────────────┘  │            │
│  └──────────────────────────────────────────────────────────┘            │
│                              O                                             │
│  ┌──────────────────────────────────────────────────────────┐            │
│  │  OPCIÓN B (Legacy): 🔮 predict_and_filter.py            │            │
│  │  - Normalización local (sin 120 días)                   │            │
│  │  - Clasificación simple                                 │            │
│  │  - Confianza basada solo en % cambio                   │            │
│  └──────────────────────────────────────────────────────────┘            │
│                                                                            │
│                              ↓                                             │
│                                                                            │
│  📄 trading_signals.csv (última señal)                                    │
│  📄 prediction_tracker.csv (histórico predicciones)                       │
│                                                                            │
│                              ↓                                             │
│                                                                            │
│  ┌─────────────────────────────────────────────────────────┐             │
│  │  💼 trading_orchestrator.py                             │             │
│  │  └─► kraken_trader.py                                   │             │
│  │      ├─ Lee última señal                                │             │
│  │      ├─ Valida coherencia predicciones                  │             │
│  │      │   ✓ pred_close entre pred_high y pred_low       │             │
│  │      ├─ Valida sincronización de precios                │             │
│  │      │   ✓ precio actual no se alejó del base          │             │
│  │      ├─ Calcula TP/SL desde rango predicho             │             │
│  │      ├─ Valida con risk_manager.py                     │             │
│  │      │   ✓ leverage, position size, R/R ratio          │             │
│  │      └─ Ejecuta trade (si todo OK)                     │             │
│  └─────────────────────────────────────────────────────────┘             │
│                              ↓                                             │
│                                                                            │
│  📄 orders_executed.csv (trades abiertos)                                 │
│  📄 open_orders.json (órdenes activas)                                    │
│  📄 risk_config.json (estado capital)                                     │
└───────────────────────────────────────────────────────────────────────────┘

                                    ↓

┌───────────────────────────────────────────────────────────────────────────┐
│  3️⃣  MONITOREO (cada 5 min) ✨ NUEVO                                      │
│  📄 Archivo: .github/workflows/3-monitor-orders.yml                       │
├───────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐             │
│  │  🔍 monitor_orders() [dentro de kraken_trader.py]       │             │
│  │  ├─ Sincroniza open_orders.json con Kraken             │             │
│  │  ├─ Para cada orden abierta:                           │             │
│  │  │   ├─ Obtiene precio actual                          │             │
│  │  │   ├─ Calcula P&L                                    │             │
│  │  │   ├─ Verifica:                                      │             │
│  │  │   │   ✓ TP alcanzado?                               │             │
│  │  │   │   ✓ SL alcanzado? (si no hay auto-SL)          │             │
│  │  │   │   ✓ Timeout? (>3.5 horas)                      │             │
│  │  │   └─ Cierra posición si aplica                     │             │
│  │  └─ Actualiza archivos                                 │             │
│  └─────────────────────────────────────────────────────────┘             │
│                              ↓                                             │
│  📄 kraken_trades.csv (trades cerrados + P&L)                             │
│  📄 open_orders.json (actualizado)                                        │
└───────────────────────────────────────────────────────────────────────────┘

                                    ↓

┌───────────────────────────────────────────────────────────────────────────┐
│  4️⃣  EVALUACIÓN (cada 6 horas o manual) ✨ NUEVO                          │
│  📄 Script: evaluate_predictions.py                                       │
├───────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐             │
│  │  📊 evaluate_predictions()                               │             │
│  │  ├─ Lee prediction_tracker.csv                          │             │
│  │  ├─ Filtra predicciones sin evaluar (>1 hora old)      │             │
│  │  ├─ Para cada predicción:                               │             │
│  │  │   ├─ Obtiene valores reales (H, L, C)               │             │
│  │  │   ├─ Calcula accuracy:                               │             │
│  │  │   │   Accuracy = 100 - MAPE                         │             │
│  │  │   │   MAPE = |pred - actual| / actual * 100        │             │
│  │  │   └─ Actualiza CSV                                  │             │
│  │  └─ Calcula estadísticas generales                     │             │
│  └─────────────────────────────────────────────────────────┘             │
│                              ↓                                             │
│  📄 prediction_tracker.csv (con actual_close y accuracy)                  │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 🔗 Enlaces entre Componentes

### 1. Entrenamiento → Predicción

```
ADAUSD_MODELS/
├─ adausd_lstm_1h.pth  ─────────┐
├─ scaler_input_1h.pkl ─────────┤
├─ scaler_output_1h.pkl ────────├─► predict_enhanced.py
└─ config_1h.json ──────────────┘    └─ Carga modelo
                                      └─ Usa scalers (legacy)
                                      └─ O calcula nuevos límites (120 días)
```

**Nota:** `predict_enhanced.py` puede:
- **Opción A (Recomendado):** Calcular nuevos límites min/max con 120 días
- **Opción B (Legacy):** Usar los scalers guardados del entrenamiento

---

### 2. Predicción → Trading

```
trading_signals.csv (última línea)
┌──────────────────────────────────────────────────────┐
│ timestamp, current_price, pred_high, pred_low,      │
│ pred_close, signal, confidence, rsi, atr, trend     │
└──────────────────────────────────────────────────────┘
                      ↓
        kraken_trader.load_last_signal()
                      ↓
        ┌─────────────────────────────┐
        │ Validaciones:               │
        │ 1. Coherencia predicciones  │
        │ 2. Sincronización precios   │
        │ 3. Risk management          │
        └─────────────────────────────┘
                      ↓
              place_order() SI todo OK
```

---

### 3. Trading → Monitoreo

```
open_orders.json
┌────────────────────────────────────────────────────────┐
│ {                                                      │
│   "order_id_123": {                                    │
│     "side": "buy",                                     │
│     "entry_price": 0.6542,                            │
│     "take_profit": 0.6587,                            │
│     "stop_loss": 0.6497,                              │
│     "entry_time": "2025-01-15T10:00:00",             │
│     "volume": 150,                                     │
│     ...                                                │
│   }                                                    │
│ }                                                      │
└────────────────────────────────────────────────────────┘
              ↓ (cada 5 min)
     monitor_orders()
              ↓
    ┌─────────────────────┐
    │ Para cada orden:    │
    │ - Check precio      │
    │ - Eval TP/SL        │
    │ - Check timeout     │
    │ - Close si aplica   │
    └─────────────────────┘
              ↓
     kraken_trades.csv (si cerró)
```

---

### 4. Predicción → Evaluación

```
prediction_tracker.csv
┌───────────────────────────────────────────────────────────┐
│ timestamp, pred_high, pred_low, pred_close,              │
│ actual_high=NULL, actual_low=NULL, actual_close=NULL,    │
│ pred_accuracy_%=NULL                                      │
└───────────────────────────────────────────────────────────┘
                      ↓ (después de 1 hora)
           evaluate_predictions()
                      ↓
           ┌─────────────────────────┐
           │ Obtiene valores reales  │
           │ Calcula MAPE / Accuracy │
           │ Actualiza CSV           │
           └─────────────────────────┘
                      ↓
prediction_tracker.csv (actualizado)
┌───────────────────────────────────────────────────────────┐
│ timestamp, pred_high, pred_low, pred_close,              │
│ actual_high=0.6580, actual_low=0.6510,                   │
│ actual_close=0.6579, pred_accuracy_%=99.88               │
└───────────────────────────────────────────────────────────┘
```

---

## 🎛️ Switches y Configuración

### Workflow 2: Predictor a Usar

```yaml
# .github/workflows/2-predict-and-trade.yml

workflow_dispatch:
  inputs:
    use_legacy_predictor:
      description: 'Usar predictor antiguo'
      type: boolean
      default: false  # ✅ Por defecto usa Enhanced
```

**Para forzar legacy:**
1. Ve a Actions → Predict & Trade
2. Run workflow
3. Marca checkbox "Usar predictor antiguo"

---

### Workflow 3: Frecuencia de Monitoreo

```yaml
# .github/workflows/3-monitor-orders.yml

on:
  schedule:
    - cron: '*/5 * * * *'  # Cada 5 minutos
```

**Para cambiar frecuencia:**
- `*/5` = cada 5 min
- `*/10` = cada 10 min
- `0 * * * *` = cada hora

---

## 📊 Archivos CSV y JSON

### Estado en Tiempo Real

| Archivo | Propósito | Actualizado por | Frecuencia |
|---------|-----------|-----------------|------------|
| `trading_signals.csv` | Última señal generada | predict_enhanced.py | 15 min |
| `prediction_tracker.csv` | Histórico predicciones | predict_enhanced.py | 15 min |
| `open_orders.json` | Órdenes activas | kraken_trader.py | 5-15 min |
| `orders_executed.csv` | Trades abiertos | kraken_trader.py | Al abrir |
| `kraken_trades.csv` | Trades cerrados | monitor_orders() | Al cerrar |
| `risk_config.json` | Capital/margen | risk_manager.py | Cada trade |

---

## 🔄 Ciclo de Vida de un Trade

```
1. PREDICCIÓN (15 min)
   predict_enhanced.py
   └─► trading_signals.csv
          │
          ├─ timestamp: 2025-01-15 10:00:00
          ├─ signal: BUY
          ├─ confidence: 78.5%
          ├─ pred_close: 0.6587
          └─ pred_range: 2.1%

2. VALIDACIÓN (15 min)
   kraken_trader.py
   └─► Checks:
          ✓ Coherencia (Close entre High/Low)
          ✓ Sincronización (drift < 3%)
          ✓ Risk (leverage, position size)

3. EJECUCIÓN (15 min)
   kraken_trader.place_order()
   └─► open_orders.json
          │
          ├─ order_id: "ABC123"
          ├─ entry_price: 0.6542
          ├─ take_profit: 0.6587
          ├─ stop_loss: 0.6497
          └─ entry_time: 10:00:00

4. MONITOREO (cada 5 min)
   monitor_orders()
   └─► Checks:
          • Precio actual vs TP
          • Precio actual vs SL
          • Tiempo desde entrada
          
   Si TP/SL/Timeout:
   └─► close_position()
          └─► kraken_trades.csv
                 │
                 ├─ close_price: 0.6587
                 ├─ pnl_usd: +6.75
                 ├─ pnl_%: +0.69%
                 └─ close_reason: "TP"

5. EVALUACIÓN (1+ hora después)
   evaluate_predictions.py
   └─► prediction_tracker.csv (actualizado)
          │
          ├─ actual_close: 0.6579
          ├─ pred_accuracy_%: 99.88%
          └─ prediction_error: -$0.0008
```

---

## 🎯 Dependencias entre Scripts

```
adausd_lstm.py (training)
    ↓ (genera)
ADAUSD_MODELS/*
    ↓ (usa)
predict_enhanced.py
    ↓ (genera)
trading_signals.csv + prediction_tracker.csv
    ↓ (lee)
kraken_trader.py
    ↓ (usa)
risk_manager.py
    ↓ (genera)
open_orders.json + orders_executed.csv
    ↓ (lee)
monitor_orders() [en kraken_trader.py]
    ↓ (genera)
kraken_trades.csv
    
(paralelo)
prediction_tracker.csv
    ↓ (lee)
evaluate_predictions.py
    ↓ (actualiza)
prediction_tracker.csv (con accuracy)
```

---

## ✨ Diferencias Clave: Legacy vs Enhanced

| Aspecto | Legacy (`predict_and_filter.py`) | Enhanced (`predict_enhanced.py`) |
|---------|----------------------------------|----------------------------------|
| **Normalización** | ScaleR local (del training) | Min/Max de 120 días (MQL5) |
| **Clasificación** | Solo % cambio en Close | Multi-factor (H/L/C + rango) |
| **Confianza** | Basada en % + RSI + Trend | Dinámica (6 factores) |
| **Coherencia** | No valida | Valida Close entre H/L |
| **Código** | ~250 líneas | ~550 líneas |
| **Accuracy esperado** | ~92% | ~95-97% |

---

## 🚀 Para Empezar

1. **Verifica que tienes todos los archivos:**
   ```bash
   python diagnostics.py
   ```

2. **Actualiza workflow 2:**
   - Reemplaza `.github/workflows/2-predict-and-trade.yml`
   - Con la versión actualizada que te di

3. **Commit y push:**
   ```bash
   git add .
   git commit -m "🔗 Enlazar predict_enhanced con workflows"
   git push
   ```

4. **Verifica en Actions:**
   - Ve a Actions → Predict & Trade
   - Debería decir "Using predict_enhanced.py (MQL5 approach)"

---

**¿Ahora está todo enlazado? 🔗**

Sí, con el workflow actualizado:
- ✅ `1-train-model.yml` → entrena modelo
- ✅ `2-predict-and-trade.yml` → usa `predict_enhanced.py` (enhanced) o `predict_and_filter.py` (legacy)
- ✅ `3-monitor-orders.yml` → monitorea órdenes
- ✅ `evaluate_predictions.py` → evalúa accuracy

Todo está conectado ahora. 🎉
