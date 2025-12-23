"""
ORQUESTADOR DE TRADING - VERSIÓN MEJORADA

🔥 Ejecuta INMEDIATAMENTE cuando hay señal BUY/SELL
⏰ Monitorea cada 15 minutos
📊 Reportes diarios
"""

import os
import sys
import time
from datetime import datetime
import requests

# Telegram
TELEGRAM_API = os.environ.get('TELEGRAM_API', '')
CHAT_ID = os.environ.get('CHAT_ID', '')

def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_API}/sendMessage"
        requests.post(url, data={'chat_id': CHAT_ID, 'text': msg, 'parse_mode': 'Markdown'}, timeout=10)
    except Exception as e:
        print(f"❌ Telegram: {e}")

def execute_strategy():
    """
    🔥 FUNCIÓN PRINCIPAL
    
    Este script se ejecuta DESPUÉS de predict_and_filter.py
    Lee la última señal y ejecuta el trade si es válida
    """
    print(f"\n{'='*70}")
    print(f"💼 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ESTRATEGIA DE TRADING")
    print(f"{'='*70}\n")
    
    try:
        # Importar el trader
        from kraken_trader import execute_trading_strategy, monitor_orders
        
        # 1. Intentar ejecutar nuevo trade
        print("🎯 Buscando señales para ejecutar...")
        execute_trading_strategy()
        
        # 2. Monitorear órdenes existentes
        print("\n🔍 Monitoreando órdenes abiertas...")
        time.sleep(2)
        monitor_orders()
        
        print("\n" + "="*70)
        print("  ✅ ESTRATEGIA COMPLETADA")
        print("="*70)
        
    except ImportError as e:
        error_msg = f"❌ Error importando módulos: {str(e)}"
        print(error_msg)
        send_telegram(error_msg)
        sys.exit(1)
        
    except Exception as e:
        error_msg = f"❌ Error en estrategia: {str(e)}"
        print(error_msg)
        send_telegram(error_msg)
        raise

if __name__ == "__main__":
    try:
        execute_strategy()
    except KeyboardInterrupt:
        print("\n\n🛑 Ejecución interrumpida manualmente")
    except Exception as e:
        print(f"\n❌ Error crítico: {str(e)}")
        raise
