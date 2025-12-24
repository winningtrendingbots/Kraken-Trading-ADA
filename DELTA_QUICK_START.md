# 🚀 Delta System - Inicio Rápido

## ⚡ Quick Start (5 minutos)

### 1. Migrar (opcional si tienes sistema antiguo)

```bash
python migrate_to_delta.py
```

### 2. Entrenar Modelo

```bash
python adausd_lstm_5min_delta.py
```

⏱️ **Tiempo**: ~20 minutos  
📊 **Output**: `ADAUSD_MODELS/adausd_lstm_delta.pth`

### 3. Predecir

```bash
python predict_delta_5min.py
```

⏱️ **Tiempo**: <1 minuto  
📊 **Output**: `trading_signals.csv`

### 4. Tradear

```bash
python trading_orchestrator.py
```

---

## 🔥 Diferencia Clave: Antes vs Después

### ❌ Antes (Problema)

```python
Precio Actual:  $0.3577
Pred High:      $0.3924  # 😱 ¿De dónde salió esto?
Pred Low:       $0.3896  # 😱 Más alto que el actual
Pred Close:     $0.3910  # 😱 Desconectado
```

### ✅ Ahora (Solución)

```python
Precio Actual:  $0.3577
Delta High:     +1.2%  →  $0.3620  # ✅ Anclado!
Delta Low:      -0.8%  →  $0.3548  # ✅ Tiene sentido!
Delta Close:    +0.5%  →  $0.3595  # ✅ Conectado!
```

---

## 📋 Checklist de Verificación

Antes de predecir, asegúrate:

- [ ] ✅ Modelo entrenado (`adausd_lstm_delta.pth` existe)
- [ ] ✅ Config correcta (`config_delta.json` tiene `use_delta: true`)
- [ ] ✅ Scalers correctos (`scaler_*_delta.pkl` existen)
- [ ] ✅ Script correcto (usas `predict_delta_5min.py`)

---

## 🎛️ Configuración Recomendada

En `adausd_lstm_5min_delta.py`:

```python
class Config:
    USE_VOLUME = True              # ✅ OBLIGATORIO
    USE_DELTA_PREDICTION = True    # ✅ OBLIGATORIO
    VOLUME_INDICATORS = True       # ✅ RECOMENDADO
    PREDICT_VOLUME = True          # ⚠️ Opcional
    NORMALIZE_BY_WINDOW = True     # ✅ RECOMENDADO
```

---

## 📊 Output Esperado

### Entrenamiento

```
✅ Modelo Delta+Volume Entrenado
🧠 Parámetros: 2,145,923
📈 R² (delta_close): 0.9342
```

### Predicción

```
🎯 CLASIFICACIÓN CON DELTAS:
   Precio actual: $0.3577
   Delta High: +1.20% → $0.3620
   
✅ VERIFICACIÓN DE ANCLAJE:
   ¿High > Low? True
   ¿Close en rango? True
   ¿Precio actual referenciado? ✅

📊 ANÁLISIS DE VOLUMEN:
   Tendencia: STRONG_BULLISH
   Soporte: ✅

🎲 SEÑAL FINAL: BUY
🎲 CONFIANZA: 87.5%
```

---

## 🐛 Troubleshooting Rápido

### Problema: "Precio fuera del rango predicho"

❌ **Estás usando el modelo antiguo**

✅ **Solución**:
```bash
# Verifica que uses el script correcto
python predict_delta_5min.py  # ✅
# NO uses:
python predict_enhanced_5min.py  # ❌
```

### Problema: "Model file not found"

❌ **No has entrenado el modelo delta**

✅ **Solución**:
```bash
python adausd_lstm_5min_delta.py
```

### Problema: "Key error: 'use_delta'"

❌ **Config antigua**

✅ **Solución**:
```bash
# Eliminar configs antiguas
rm ADAUSD_MODELS/config.json
rm ADAUSD_MODELS/config_1h.json

# Reentrenar
python adausd_lstm_5min_delta.py
```

---

## 📚 Documentación Completa

- **Guía detallada**: `README_DELTA_SYSTEM.md`
- **Migración**: `migrate_to_delta.py`
- **Código fuente**: `adausd_lstm_5min_delta.py`
- **Predictor**: `predict_delta_5min.py`

---

## ✅ Workflow Automatizado (GitHub Actions)

### Reemplazar workflows:

1. **Entrenamiento**:
   ```
   .github/workflows/1-train-model.yml
   ```
   Reemplazar con: `1-train-model-delta.yml`

2. **Predicción**:
   ```
   .github/workflows/2-predict-and-trade.yml
   ```
   Reemplazar con: `2-predict-delta.yml`

### Ejecutar manualmente:

1. Ve a **Actions** → **Train Model (Delta + Volume)**
2. Click en **Run workflow**
3. Espera ~20 minutos
4. ✅ Listo para predecir cada 5 minutos

---

## 🎯 Comandos Esenciales

```bash
# 1. Entrenar
python adausd_lstm_5min_delta.py

# 2. Predecir
python predict_delta_5min.py

# 3. Tradear
python trading_orchestrator.py

# 4. Diagnosticar
python diagnostics.py

# 5. Analytics
python analytics.py
```

---

## 💡 Tips Rápidos

### ✅ DO:
- Usar `predict_delta_5min.py`
- Verificar anclaje en cada predicción
- Confiar en el análisis de volumen
- Validar breakouts con volumen alto

### ❌ DON'T:
- Usar `predict_and_filter.py` (obsoleto)
- Usar `predict_enhanced_5min.py` sin volumen
- Ignorar las divergencias volumen-precio
- Operar breakouts sin volumen

---

## 🔑 Conceptos Clave

### **Delta** = Cambio Relativo

```python
delta = (precio_futuro - precio_actual) / precio_actual
```

### **Anclaje** = Referencia al Precio Actual

```python
pred_high = precio_actual * (1 + delta_high)  # ✅
# NO:
pred_high = modelo.predict()  # ❌ Sin contexto
```

### **Volumen** = Confirmación

```
Alto Volumen + Subida = Tendencia Alcista Fuerte ✅
Bajo Volumen + Subida = Posible Trampa ⚠️
```

---

## 🎉 ¡Listo!

**Sistema configurado y funcionando con:**
- ✅ Predicciones ancladas
- ✅ Análisis de volumen
- ✅ Validación de breakouts
- ✅ Detección de divergencias

**¡A tradear! 🚀**
