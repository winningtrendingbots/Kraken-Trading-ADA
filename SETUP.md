# 🚀 Instrucciones de Configuración

## 📋 Problema Actual

Tu workflow tiene un error: `cache: 'pip'` requiere `requirements.txt`

## ✅ Solución Rápida

### Opción 1: Sin requirements.txt (Más Simple)

Usa el workflow simplificado que ya no requiere cache.

**Archivos a añadir/reemplazar:**
```
.github/workflows/
├── train-model-simple.yml     # Reemplaza schedule.yml
├── hourly-trading.yml          # Nuevo - Trading cada hora
├── monitor-orders.yml          # Nuevo - Monitoreo cada 15min
└── fix-conflicts.yml           # Emergencias
```

### Opción 2: Con requirements.txt (Recomendado)

1. Añade el archivo `requirements.txt` a la raíz del repo
2. Mantén los workflows actualizados

## 🔧 Pasos Para Implementar

### 1️⃣ Limpiar Estado Actual

```bash
# Opción A: Desde GitHub Actions
# Ve a Actions → Fix Merge Conflicts → Run workflow

# Opción B: Localmente
git clone https://github.com/winningtrendingbots/Kraken-Trading.git
cd Kraken-Trading
git reset --hard origin/main
git push --force
```

### 2️⃣ Añadir Archivos Nuevos

Crea esta estructura en tu repositorio:

```
Kraken-Trading/
├── .github/
│   └── workflows/
│       ├── train-model-simple.yml      ⭐ NUEVO
│       ├── hourly-trading.yml          ⭐ NUEVO
│       ├── monitor-orders.yml          ⭐ NUEVO
│       └── fix-conflicts.yml           ⭐ NUEVO
├── ethusd_lstm.py                      ✅ Ya existe
├── predict_and_filter.py               ⭐ AÑADIR (del artifact anterior)
├── kraken_trader.py                    ⭐ AÑADIR (del artifact anterior)
├── trading_orchestrator.py             ⭐ AÑADIR (opcional)
├── analytics.py                        ⭐ AÑADIR (opcional)
├── requirements.txt                    ⭐ AÑADIR (recomendado)
└── README.md                           ✅ Ya existe
```

### 3️⃣ Configurar Workflows

**Elimina o desactiva:**
- `schedule.yml` (el antiguo)

**Activa los nuevos:**
- ✅ `train-model-simple.yml` - Entrena diario a las 10 AM
- ✅ `hourly-trading.yml` - Trading cada hora
- ✅ `monitor-orders.yml` - Monitoreo cada 15 min

### 4️⃣ Variables Sensibles

⚠️ **IMPORTANTE:** Tus credenciales están en el código. Debes moverlas a **GitHub Secrets**.

**Cómo hacerlo:**

1. Ve a tu repo → **Settings** → **Secrets and variables** → **Actions**

2. Añade estos secrets:
   ```
   TELEGRAM_API=8286372753:AAF356kUIEbZRI-Crdo4jIrXc89drKGWIWY
   TELEGRAM_CHAT_ID=5825443798
   KRAKEN_API_KEY=BuVj1zFpmH8aoKXBMCfvcfmso4FD7O5tAlXDFD9aLNDc91S1wXYqNXVs
   KRAKEN_API_SECRET=XLDq0M9GmSgzjerQNiXhoq7QsHRPF2qaVowSq8He7kVrlyXnF1Lf59v3lGccCitkuki68FsJvv79idoT10OeEQ==
   ```

3. Actualiza tus scripts Python:
   ```python
   import os
   
   TELEGRAM_API = os.environ.get('TELEGRAM_API')
   CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')
   KRAKEN_API_KEY = os.environ.get('KRAKEN_API_KEY')
   KRAKEN_API_SECRET = os.environ.get('KRAKEN_API_SECRET')
   ```

4. Actualiza workflows para pasar secrets:
   ```yaml
   - name: Run script
     env:
       TELEGRAM_API: ${{ secrets.TELEGRAM_API }}
       TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
       KRAKEN_API_KEY: ${{ secrets.KRAKEN_API_KEY }}
       KRAKEN_API_SECRET: ${{ secrets.KRAKEN_API_SECRET }}
     run: python script.py
   ```

## 📊 Estructura de Workflows

### train-model-simple.yml
- **Cuándo:** Diario a las 10 AM UTC
- **Qué hace:** Entrena el modelo LSTM
- **Outputs:** ETHUSD_1h_data.csv, modelo, scalers, gráficas

### hourly-trading.yml
- **Cuándo:** Cada hora en punto
- **Qué hace:**
  1. Genera predicción con modelo LSTM
  2. Aplica filtros técnicos
  3. Ejecuta orden en Kraken si hay señal
- **Outputs:** trading_signals.csv, orders_executed.csv

### monitor-orders.yml
- **Cuándo:** Cada 15 minutos
- **Qué hace:**
  1. Revisa órdenes abiertas
  2. Cierra por TP/SL/Timeout
  3. Actualiza registros
- **Outputs:** kraken_trades.csv, open_orders.json

## 🧪 Testing

### Test Manual de Workflows

```bash
# 1. Ejecuta workflow manualmente
# GitHub → Actions → [Workflow] → Run workflow

# 2. Verifica logs
# Actions → [Ejecución] → Ver detalles

# 3. Revisa archivos generados
# Repo → Files → Verificar nuevos CSVs
```

### Test Local

```bash
# Clonar repo
git clone https://github.com/winningtrendingbots/Kraken-Trading.git
cd Kraken-Trading

# Instalar dependencias
pip install -r requirements.txt

# Test 1: Entrenamiento
python ethusd_lstm.py

# Test 2: Predicciones
python predict_and_filter.py

# Test 3: Trading (modo test)
# Edita PAPER_TRADING = True en kraken_trader.py
python kraken_trader.py
```

## 🎯 Checklist de Implementación

- [ ] Limpiar conflictos actuales
- [ ] Añadir `requirements.txt`
- [ ] Añadir scripts Python nuevos
- [ ] Actualizar workflows
- [ ] Mover credenciales a Secrets
- [ ] Actualizar código para usar secrets
- [ ] Test manual de cada workflow
- [ ] Verificar Telegram notifications
- [ ] Monitorear primera ejecución automática
- [ ] Configurar dashboard en GitHub Pages

## 🆘 Troubleshooting

### Error: "No such file"
- Verifica que todos los .py están en la raíz del repo
- Haz `git add` y `git commit` antes de ejecutar workflows

### Error: "Module not found"
- Asegúrate que requirements.txt está completo
- Verifica que el workflow instala dependencias

### Error: "Permission denied"
- Verifica que el workflow tiene `permissions: contents: write`
- Revisa Settings → Actions → General → Workflow permissions

### Conflictos de merge
- Ejecuta `fix-conflicts.yml` desde Actions
- O resetea localmente con `git reset --hard origin/main`

## 📚 Documentación Adicional

- [GitHub Actions Docs](https://docs.github.com/en/actions)
- [Kraken API Docs](https://docs.kraken.com/rest/)
- [PyTorch Docs](https://pytorch.org/docs/)

---

💡 **Tip:** Empieza con los workflows simples, verifica que funcionan, y luego añade complejidad.
