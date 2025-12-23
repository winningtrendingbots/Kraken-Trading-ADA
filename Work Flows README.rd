# 🤖 Trading Bot - Arquitectura de Workflows

## 📊 Estructura de Workflows Independientes

Los workflows están diseñados para ejecutarse de forma **independiente** y en diferentes frecuencias optimizadas para cada tarea.

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRADING BOT WORKFLOWS                        │
└─────────────────────────────────────────────────────────────────┘

┌───────────────────────┐
│  1-train-model.yml    │  ⏰ 1 vez al día (2 AM UTC)
│  🧠 Train Model       │  ⏱️  ~20-30 min
└───────────┬───────────┘
            │
            │ Crea/Actualiza: ADAUSD_MODELS/
            │                 ADAUSD_1h_data.csv
            ↓
     [Modelo LSTM]
            ↓
┌───────────────────────┐
│  2-predict-trade.yml  │  ⏰ Cada 10 minutos
│  🔮 Predict & Trade   │  ⏱️  ~5-8 min
└───────────┬───────────┘
            │
            │ Lee: ADAUSD_MODELS/
            │ Crea: trading_signals.csv
            │       orders_executed.csv
            ↓
     [Señales y Órdenes]
            ↓
┌───────────────────────┐
│  3-monitor-orders.yml │  ⏰ Cada 4 minutos
│  👀 Monitor Orders    │  ⏱️  ~3-5 min
└───────────┬───────────┘
            │
            │ Lee: orders_executed.csv
            │ Actualiza: kraken_trades.csv
            ↓
     [Trades Completados]
            ↓
┌───────────────────────┐
│  4-sync-dashboard.yml │  ⏰ Cada 5 minutos
│  🔄 Sync Dashboard    │  ⏱️  ~1-2 min
└───────────┬───────────┘
            │
            │ Copia: *.csv, *.json, *.png
            │ Destino: github.io repo
            ↓
     [Dashboard Web]
```

## 🎯 Flujo de Ejecución

### Primera Ejecución (Sistema Nuevo)

```
HORA    WORKFLOW              ACCIÓN
─────────────────────────────────────────────────
02:00   1-train-model        ✅ Entrena modelo inicial
        
02:30   2-predict-trade      ⏳ Espera (no hay modelo aún)
        3-monitor-orders     ⏳ Espera (no hay órdenes)
        4-sync-dashboard     ✅ Crea placeholders

02:35   1-train-model        ✅ COMPLETO - Modelo guardado
        
02:40   2-predict-trade      ✅ Genera primera señal
        3-monitor-orders     ⏳ Espera (aún no hay órdenes)
        4-sync-dashboard     ✅ Sincroniza señales
        
02:50   2-predict-trade      ✅ Genera señal + Ejecuta trade
        3-monitor-orders     ✅ Monitorea orden abierta
        4-sync-dashboard     ✅ Sincroniza todo
```

### Operación Normal

```
HORA    WORKFLOW              ESTADO
─────────────────────────────────────────────────
00:00   2-predict-trade      ✅ Predicción + Trading
        3-monitor-orders     ✅ Monitoreando
        4-sync-dashboard     ✅ Sincronizando

00:04   3-monitor-orders     ✅ Monitoreando

00:05   4-sync-dashboard     ✅ Sincronizando

00:08   3-monitor-orders     ✅ Monitoreando

00:10   2-predict-trade      ✅ Predicción + Trading
        4-sync-dashboard     ✅ Sincronizando

00:12   3-monitor-orders     ✅ Monitoreando

... y así sucesivamente
```

## 📁 Archivos Generados

| Archivo | Generado Por | Actualizado Por | Frecuencia |
|---------|--------------|-----------------|------------|
| `ADAUSD_MODELS/*.keras` | train-model | train-model | 1x día (o cada 7 días) |
| `ADAUSD_1h_data.csv` | train-model | train-model | 1x día |
| `trading_signals.csv` | predict-trade | predict-trade | Cada 10 min |
| `orders_executed.csv` | predict-trade | predict-trade | Cada 10 min |
| `kraken_trades.csv` | monitor-orders | monitor-orders | Cada 4 min |
| `*.png` | train-model | analytics | Variable |

## ⚙️ Configuración de Secrets

Todos los workflows necesitan estos secrets configurados en GitHub:

```
KRAKEN_API_KEY      → API de Kraken para trading
KRAKEN_API_SECRET   → Secret de Kraken
TELEGRAM_API        → Bot token de Telegram
CHAT_ID             → ID del chat de Telegram
DASHBOARD_TOKEN     → Personal Access Token para sync
```

## 🔧 Características de Cada Workflow

### 1️⃣ Train Model
- ✅ Verifica si necesita reentrenar (cada 7 días)
- ✅ Puede forzarse manualmente
- ✅ Se ejecuta 1 vez al día para ahorrar recursos
- ✅ Envía notificación a Telegram cuando completa

### 2️⃣ Predict & Trade
- ✅ Verifica que exista modelo antes de ejecutar
- ✅ Genera predicciones cada 10 minutos
- ✅ Ejecuta trades basados en señales
- ✅ Guarda historial de señales y órdenes

### 3️⃣ Monitor Orders
- ✅ Revisa órdenes abiertas cada 4 minutos
- ✅ Actualiza estados (TP/SL/TIMEOUT)
- ✅ Registra trades completados
- ✅ Solo se ejecuta si hay órdenes activas

### 4️⃣ Sync Dashboard
- ✅ Sincroniza datos al repo de GitHub Pages
- ✅ Crea placeholders si no hay datos
- ✅ Solo hace commit si hay cambios
- ✅ Mantiene metadata de sincronización

## 🎛️ Ejecución Manual

Todos los workflows pueden ejecutarse manualmente desde GitHub:

```
Actions → [Nombre del Workflow] → Run workflow
```

Útil para:
- 🧠 Forzar reentrenamiento del modelo
- 🔮 Probar predicciones inmediatamente
- 👀 Revisar órdenes fuera de horario
- 🔄 Forzar sincronización del dashboard

## 📊 Monitoreo

Puedes ver el estado de todos los workflows en:
- GitHub Actions tab
- Dashboard web (https://winningtrendingbots.github.io)
- Notificaciones de Telegram

## ⚠️ Resolución de Problemas

### "Waiting for model"
→ Ejecuta manualmente `1-train-model.yml`

### "No orders yet"
→ Normal, espera a que `predict-trade` ejecute el primer trade

### "Dashboard not updating"
→ Verifica que `DASHBOARD_TOKEN` esté configurado correctamente

### "Python errors"
→ Revisa que todos los archivos .py existan y requirements.txt esté completo
