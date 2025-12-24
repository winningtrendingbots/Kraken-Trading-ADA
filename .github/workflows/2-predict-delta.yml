name: 🔮 Predict & Trade (Delta + Volume)

on:
  schedule:
    - cron: '*/5 * * * *'  # Cada 5 minutos
  workflow_dispatch:
    inputs:
      use_legacy_predictor:
        description: 'Usar predictor antiguo (sin deltas)'
        type: boolean
        default: false

permissions:
  contents: write

jobs:
  predict-and-trade:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    
    steps:
      - name: 📋 Log execution
        run: |
          echo "🕐 Workflow ejecutado: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
          echo "📋 Trigger: ${{ github.event_name }}"
          echo "🌿 Branch: ${{ github.ref }}"
          
          if [ "${{ github.event.inputs.use_legacy_predictor }}" = "true" ]; then
            echo "📧 Predictor: Legacy (sin deltas)"
          else
            echo "🔥 Predictor: Delta + Volume"
          fi
      
      - name: 📥 Checkout
        uses: actions/checkout@v4
      
      - name: 🔍 Check if model exists
        id: check
        run: |
          echo "🔎 Verificando modelo Delta..."
          
          if [ -d "ADAUSD_MODELS" ] && [ "$(ls -A ADAUSD_MODELS/*delta.pth 2>/dev/null)" ]; then
            echo "✅ Modelo Delta encontrado"
            echo "has_model=true" >> $GITHUB_OUTPUT
            
            # Mostrar info
            if [ -f "ADAUSD_MODELS/config_delta.json" ]; then
              echo ""
              echo "📊 Información del modelo:"
              python3 << 'PYEOF'
import json
try:
    with open('ADAUSD_MODELS/config_delta.json', 'r') as f:
        config = json.load(f)
    
    print(f"   Features: {config.get('input_size', 'N/A')}")
    print(f"   Outputs: {config.get('output_size', 'N/A')}")
    print(f"   Use Delta: {config.get('use_delta', False)}")
    print(f"   Use Volume: {config.get('use_volume', False)}")
    print(f"   Volume Indicators: {config.get('volume_indicators', False)}")
    
    if 'metrics_test' in config and 'delta_close' in config['metrics_test']:
        r2 = config['metrics_test']['delta_close'].get('R2', 0)
        print(f"   R² (delta_close): {r2:.4f}")
except Exception as e:
    print(f"   (Error parseando: {e})")
PYEOF
            fi
          else
            echo "❌ No hay modelo Delta entrenado"
            echo "has_model=false" >> $GITHUB_OUTPUT
          fi
      
      - name: ⏳ Wait for model
        if: steps.check.outputs.has_model == 'false'
        run: |
          echo ""
          echo "⏳ ============================================"
          echo "   NO HAY MODELO DELTA ENTRENADO"
          echo "============================================"
          echo ""
          echo "🎯 Para empezar a predecir:"
          echo "   1. Ve a 'Actions' → 'Train Model (Delta + Volume)'"
          echo "   2. Presiona 'Run workflow'"
          echo "   3. Espera ~20-25 minutos"
          echo ""
          echo "📊 Beneficios del nuevo sistema:"
          echo "   ✅ Predicciones ancladas al precio actual"
          echo "   ✅ Análisis de volumen avanzado"
          echo "   ✅ Validación de breakouts"
          echo "   ✅ Detección de divergencias"
          echo ""
          exit 0
      
      - name: 🐍 Setup Python
        if: steps.check.outputs.has_model == 'true'
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'
      
      - name: 📦 Install dependencies
        if: steps.check.outputs.has_model == 'true'
        run: |
          echo "📦 Instalando dependencias..."
          pip install --upgrade pip
          pip install -r requirements.txt
      
      - name: 🔮 Generate predictions (Delta + Volume)
        if: steps.check.outputs.has_model == 'true' && github.event.inputs.use_legacy_predictor != 'true'
        env:
          KRAKEN_API_KEY: ${{ secrets.KRAKEN_API_KEY }}
          KRAKEN_API_SECRET: ${{ secrets.KRAKEN_API_SECRET }}
          TELEGRAM_API: ${{ secrets.TELEGRAM_API }}
          CHAT_ID: ${{ secrets.CHAT_ID }}
        run: |
          echo ""
          echo "🔮 ============================================"
          echo "   GENERANDO PREDICCIONES (DELTA + VOLUME)"
          echo "============================================"
          echo ""
          echo "✨ Características activas:"
          echo "   • Predicción de deltas (anclaje garantizado)"
          echo "   • Análisis de volumen (OBV, VWAP, PVT)"
          echo "   • Confirmación de tendencia"
          echo "   • Validación de breakouts"
          echo "   • Detección de divergencias"
          echo ""
          
          if python predict_delta_5min.py; then
            echo ""
            echo "✅ Predicción completada (Delta + Volume)"
            
            if [ -f "trading_signals.csv" ]; then
              echo "✅ Señales guardadas en trading_signals.csv"
              
              echo ""
              echo "📊 Última señal generada:"
              tail -n 1 trading_signals.csv
            else
              echo "⚠️ No se generó trading_signals.csv"
              exit 1
            fi
          else
            echo ""
            echo "❌ Error en predict_delta_5min.py"
            exit 1
          fi
      
      - name: 🔮 Generate predictions (Legacy)
        if: steps.check.outputs.has_model == 'true' && github.event.inputs.use_legacy_predictor == 'true'
        env:
          KRAKEN_API_KEY: ${{ secrets.KRAKEN_API_KEY }}
          KRAKEN_API_SECRET: ${{ secrets.KRAKEN_API_SECRET }}
          TELEGRAM_API: ${{ secrets.TELEGRAM_API }}
          CHAT_ID: ${{ secrets.CHAT_ID }}
        run: |
          echo "⚠️ Usando predictor legacy (sin deltas)"
          python predict_enhanced_5min.py
      
      - name: 💼 Execute trades
        if: steps.check.outputs.has_model == 'true'
        env:
          KRAKEN_API_KEY: ${{ secrets.KRAKEN_API_KEY }}
          KRAKEN_API_SECRET: ${{ secrets.KRAKEN_API_SECRET }}
          TELEGRAM_API: ${{ secrets.TELEGRAM_API }}
          CHAT_ID: ${{ secrets.CHAT_ID }}
        run: |
          echo ""
          echo "💼 ============================================"
          echo "   EJECUTANDO ESTRATEGIA DE TRADING"
          echo "============================================"
          echo ""
          
          if python trading_orchestrator.py; then
            echo ""
            echo "✅ Trading ejecutado correctamente"
            
            if [ -f "open_orders.json" ]; then
              echo ""
              echo "📋 Órdenes abiertas:"
              cat open_orders.json | python3 -m json.tool 2>/dev/null || cat open_orders.json
            else
              echo "ℹ️ No hay órdenes abiertas"
            fi
          else
            echo ""
            echo "⚠️ Error en trading_orchestrator.py (no crítico)"
          fi
      
      - name: 💾 Save signals and orders
        if: steps.check.outputs.has_model == 'true'
        run: |
          echo ""
          echo "💾 ============================================"
          echo "   GUARDANDO DATOS"
          echo "============================================"
          echo ""
          
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          
          git add trading_signals.csv 2>/dev/null || true
          git add orders_executed.csv 2>/dev/null || true
          git add prediction_tracker.csv 2>/dev/null || true
          git add open_orders.json 2>/dev/null || true
          git add kraken_trades.csv 2>/dev/null || true
          git add risk_config.json 2>/dev/null || true
          
          if git diff --staged --quiet; then
            echo "ℹ️ No hay nuevos datos que guardar"
          else
            echo "📋 Archivos modificados:"
            git diff --staged --name-only
            
            TIMESTAMP=$(date -u +"%Y-%m-%d %H:%M UTC")
            PREDICTOR="${{ github.event.inputs.use_legacy_predictor == 'true' && 'legacy' || 'delta+volume' }}"
            git commit -m "🤖 Trading cycle [$PREDICTOR]: $TIMESTAMP"
            
            MAX_RETRIES=3
            for i in $(seq 1 $MAX_RETRIES); do
              if git push; then
                echo ""
                echo "✅ Datos guardados correctamente"
                break
              else
                if [ $i -lt $MAX_RETRIES ]; then
                  echo "⚠️ Reintentando push ($i/$MAX_RETRIES)..."
                  sleep 2
                  git pull --rebase
                else
                  echo "❌ Error al guardar datos después de $MAX_RETRIES intentos"
                  exit 1
                fi
              fi
            done
          fi
      
      - name: 📊 Summary
        if: always() && steps.check.outputs.has_model == 'true'
        run: |
          echo ""
          echo "📊 ============================================"
          echo "   RESUMEN DE EJECUCIÓN"
          echo "============================================"
          echo ""
          
          if [ "${{ github.event.inputs.use_legacy_predictor }}" = "true" ]; then
            echo "📧 Predictor: Legacy (sin deltas)"
          else
            echo "🔥 Predictor: Delta + Volume"
          fi
          echo ""
          
          echo "📁 Archivos generados:"
          [ -f "trading_signals.csv" ] && echo "  ✅ trading_signals.csv" || echo "  ❌ trading_signals.csv"
          [ -f "prediction_tracker.csv" ] && echo "  ✅ prediction_tracker.csv" || echo "  ❌ prediction_tracker.csv"
          [ -f "orders_executed.csv" ] && echo "  ✅ orders_executed.csv" || echo "  ⚠️ orders_executed.csv (sin trades)"
          [ -f "kraken_trades.csv" ] && echo "  ✅ kraken_trades.csv" || echo "  ⚠️ kraken_trades.csv (sin trades cerrados)"
          [ -f "open_orders.json" ] && echo "  ✅ open_orders.json" || echo "  ⚠️ open_orders.json (sin posiciones)"
          
          echo ""
          
          if [ -f "prediction_tracker.csv" ]; then
            echo "📈 Estadísticas de predicciones:"
            python3 << 'PYEOF'
import pandas as pd
try:
    df = pd.read_csv('prediction_tracker.csv')
    total = len(df)
    evaluated = df['actual_close'].notna().sum()
    
    print(f"  Total predicciones: {total}")
    print(f"  Evaluadas: {evaluated}")
    
    if evaluated > 0:
        avg_accuracy = df[df['actual_close'].notna()]['pred_accuracy_%'].mean()
        print(f"  Accuracy promedio: {avg_accuracy:.2f}%")
    
    # Contar señales por tipo
    buy_count = (df['signal'] == 'BUY').sum()
    sell_count = (df['signal'] == 'SELL').sum()
    hold_count = (df['signal'] == 'HOLD').sum()
    
    print(f"\n  Señales generadas:")
    print(f"    BUY: {buy_count}")
    print(f"    SELL: {sell_count}")
    print(f"    HOLD: {hold_count}")
    
except Exception as e:
    print(f"  Error: {e}")
PYEOF
          fi
          
          echo ""
          echo "🕐 Próxima ejecución: En 5 minutos"
          echo ""
