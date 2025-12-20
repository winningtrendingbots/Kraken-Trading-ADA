"""
ORQUESTADOR PRINCIPAL DE TRADING

Este script coordina:
1. Predicciones horarias con LSTM
2. Análisis cada 5 minutos con filtros técnicos
3. Ejecución de órdenes en Kraken
4. Monitoreo y cierre de órdenes cada 15 minutos
"""

import schedule
import time
from datetime import datetime
import subprocess
import requests
import pandas as pd
import os

# Telegram
TELEGRAM_API = '8286372753:AAF356kUIEbZRI-Crdo4jIrXc89drKGWIWY'
CHAT_ID = '5825443798'

def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_API}/sendMessage"
        requests.post(url, data={'chat_id': CHAT_ID, 'text': msg, 'parse_mode': 'Markdown'}, timeout=10)
    except Exception as e:
        print(f"❌ Telegram: {e}")

# Tarea 1: Predicción + Análisis (cada hora)
def hourly_prediction_task():
    print(f"\n{'='*70}")
    print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - PREDICCIÓN HORARIA")
    print(f"{'='*70}\n")
    
    try:
        result = subprocess.run(['python', 'predict_and_filter.py'], 
                              capture_output=True, text=True, timeout=300)
        print(result.stdout)
        if result.returncode != 0:
            print(f"❌ Error: {result.stderr}")
            send_telegram(f"❌ Error en predicción horaria:\n{result.stderr[:500]}")
    except Exception as e:
        error_msg = f"❌ Error ejecutando predicción: {str(e)}"
        print(error_msg)
        send_telegram(error_msg)

# Tarea 2: Ejecutar trading si hay señal (cada hora, después de predicción)
def execute_trade_task():
    print(f"\n{'='*70}")
    print(f"💼 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - EJECUTAR TRADING")
    print(f"{'='*70}\n")
    
    try:
        result = subprocess.run(['python', 'kraken_trader.py'], 
                              capture_output=True, text=True, timeout=300)
        print(result.stdout)
        if result.returncode != 0:
            print(f"❌ Error: {result.stderr}")
            send_telegram(f"❌ Error en trader:\n{result.stderr[:500]}")
    except Exception as e:
        error_msg = f"❌ Error ejecutando trader: {str(e)}"
        print(error_msg)
        send_telegram(error_msg)

# Tarea 3: Monitorear órdenes abiertas (cada 15 minutos)
def monitor_orders_task():
    print(f"\n{'='*70}")
    print(f"🔍 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - MONITOREO DE ÓRDENES")
    print(f"{'='*70}\n")
    
    try:
        # Ejecutar solo la parte de monitoreo
        from kraken_trader import monitor_orders
        monitor_orders()
    except Exception as e:
        error_msg = f"❌ Error monitoreando órdenes: {str(e)}"
        print(error_msg)
        send_telegram(error_msg)

# Tarea 4: Reporte diario
def daily_report():
    print(f"\n{'='*70}")
    print(f"📊 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - REPORTE DIARIO")
    print(f"{'='*70}\n")
    
    try:
        # Leer datos de trading
        trades_file = 'kraken_trades.csv'
        signals_file = 'trading_signals.csv'
        
        report = f"📊 *Reporte Diario - {datetime.now().strftime('%Y-%m-%d')}*\n\n"
        
        if os.path.exists(trades_file):
            df_trades = pd.read_csv(trades_file)
            df_trades['timestamp'] = pd.to_datetime(df_trades['timestamp'])
            today = df_trades[df_trades['timestamp'].dt.date == datetime.now().date()]
            
            if len(today) > 0:
                total_pnl = today['pnl_usd'].sum()
                wins = (today['pnl_usd'] > 0).sum()
                losses = (today['pnl_usd'] <= 0).sum()
                win_rate = (wins / len(today)) * 100 if len(today) > 0 else 0
                
                report += f"🔢 *Trades Hoy:* {len(today)}\n"
                report += f"✅ Ganadas: {wins}\n"
                report += f"❌ Perdidas: {losses}\n"
                report += f"📈 Win Rate: {win_rate:.1f}%\n"
                report += f"💰 P&L Día: ${total_pnl:.2f}\n\n"
                
                # Mejores y peores trades
                if len(today) > 0:
                    best = today.loc[today['pnl_usd'].idxmax()]
                    worst = today.loc[today['pnl_usd'].idxmin()]
                    
                    report += f"🏆 Mejor trade: ${best['pnl_usd']:.2f} ({best['side'].upper()})\n"
                    report += f"💔 Peor trade: ${worst['pnl_usd']:.2f} ({worst['side'].upper()})\n\n"
            else:
                report += "⏸️ No hay trades hoy\n\n"
            
            # Stats totales
            total_pnl_all = df_trades['pnl_usd'].sum()
            total_wins = (df_trades['pnl_usd'] > 0).sum()
            total_trades = len(df_trades)
            total_wr = (total_wins / total_trades) * 100 if total_trades > 0 else 0
            
            report += f"📊 *Stats Totales:*\n"
            report += f"Total trades: {total_trades}\n"
            report += f"Win rate: {total_wr:.1f}%\n"
            report += f"P&L total: ${total_pnl_all:.2f}\n"
        else:
            report += "⚠️ No hay historial de trades\n"
        
        if os.path.exists(signals_file):
            df_signals = pd.read_csv(signals_file)
            df_signals['timestamp'] = pd.to_datetime(df_signals['timestamp'])
            today_signals = df_signals[df_signals['timestamp'].dt.date == datetime.now().date()]
            
            if len(today_signals) > 0:
                buys = (today_signals['signal'] == 'BUY').sum()
                sells = (today_signals['signal'] == 'SELL').sum()
                holds = (today_signals['signal'] == 'HOLD').sum()
                
                report += f"\n🎯 *Señales Hoy:*\n"
                report += f"🟢 BUY: {buys}\n"
                report += f"🔴 SELL: {sells}\n"
                report += f"⚪ HOLD: {holds}\n"
        
        send_telegram(report)
        print(report)
        
    except Exception as e:
        error_msg = f"❌ Error generando reporte: {str(e)}"
        print(error_msg)
        send_telegram(error_msg)

# Configurar tareas programadas
def setup_scheduler():
    """Configura todas las tareas programadas"""
    
    # CADA HORA (en punto): Predicción + Ejecutar trade si hay señal
    schedule.every().hour.at(":00").do(hourly_prediction_task)
    schedule.every().hour.at(":02").do(execute_trade_task)  # 2 min después de predicción
    
    # CADA 15 MINUTOS: Monitorear órdenes abiertas
    schedule.every(15).minutes.do(monitor_orders_task)
    
    # DIARIO: Reporte a las 23:00
    schedule.every().day.at("23:00").do(daily_report)
    
    print("="*70)
    print("  🤖 TRADING BOT INICIADO")
    print("="*70)
    print("\n📅 Tareas programadas:")
    print("   🔮 Predicción + Trading: Cada hora en punto")
    print("   🔍 Monitoreo órdenes: Cada 15 minutos")
    print("   📊 Reporte diario: 23:00")
    print("\n⏰ Esperando primera ejecución...")
    print("="*70 + "\n")
    
    # Mensaje inicial
    send_telegram("""
🤖 *Trading Bot Iniciado*

📅 *Programación:*
🔮 Predicción: Cada hora
💼 Trading: Tras predicción
🔍 Monitoreo: Cada 15 min
📊 Reporte: 23:00 diario

✅ Sistema operativo
""")

# Main loop
def main():
    setup_scheduler()
    
    # Ejecutar inmediatamente primera vez
    print("🚀 Ejecutando análisis inicial...\n")
    hourly_prediction_task()
    time.sleep(5)
    execute_trade_task()
    
    # Loop principal
    while True:
        schedule.run_pending()
        time.sleep(60)  # Revisar cada minuto

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Bot detenido manualmente")
        send_telegram("🛑 Trading Bot detenido")
    except Exception as e:
        error_msg = f"❌ Error crítico: {str(e)}"
        print(error_msg)
        send_telegram(error_msg)
        raise
