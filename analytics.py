"""
ANÁLISIS Y VISUALIZACIÓN DE RENDIMIENTO

Genera reportes completos sobre:
- Performance del trading
- Efectividad de las predicciones
- Métricas por tipo de señal
- Gráficas de rendimiento
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime, timedelta
import os
import requests

# Telegram
TELEGRAM_API = '8286372753:AAF356kUIEbZRI-Crdo4jIrXc89drKGWIWY'
CHAT_ID = '5825443798'

def send_telegram(msg, photo_path=None):
    try:
        if photo_path and os.path.exists(photo_path):
            url = f"https://api.telegram.org/bot{TELEGRAM_API}/sendPhoto"
            with open(photo_path, 'rb') as f:
                files = {'photo': f}
                data = {'chat_id': CHAT_ID, 'caption': msg, 'parse_mode': 'Markdown'}
                requests.post(url, files=files, data=data, timeout=30)
        else:
            url = f"https://api.telegram.org/bot{TELEGRAM_API}/sendMessage"
            requests.post(url, data={'chat_id': CHAT_ID, 'text': msg, 'parse_mode': 'Markdown'}, timeout=10)
    except Exception as e:
        print(f"❌ Telegram: {e}")

def load_data():
    """Carga todos los CSVs disponibles"""
    data = {}
    
    if os.path.exists('kraken_trades.csv'):
        data['trades'] = pd.read_csv('kraken_trades.csv')
        data['trades']['timestamp'] = pd.to_datetime(data['trades']['timestamp'])
    
    if os.path.exists('trading_signals.csv'):
        data['signals'] = pd.read_csv('trading_signals.csv')
        data['signals']['timestamp'] = pd.to_datetime(data['signals']['timestamp'])
    
    if os.path.exists('orders_executed.csv'):
        data['executed'] = pd.read_csv('orders_executed.csv')
        data['executed']['timestamp'] = pd.to_datetime(data['executed']['timestamp'])
    
    return data

def analyze_trading_performance(df_trades):
    """Analiza el rendimiento del trading"""
    
    print("="*70)
    print("  📊 ANÁLISIS DE TRADING")
    print("="*70)
    
    if len(df_trades) == 0:
        print("⚠️ No hay trades para analizar")
        return {}
    
    metrics = {}
    
    # Métricas generales
    metrics['total_trades'] = len(df_trades)
    metrics['wins'] = (df_trades['pnl_usd'] > 0).sum()
    metrics['losses'] = (df_trades['pnl_usd'] <= 0).sum()
    metrics['win_rate'] = (metrics['wins'] / metrics['total_trades']) * 100
    
    metrics['total_pnl'] = df_trades['pnl_usd'].sum()
    metrics['avg_pnl'] = df_trades['pnl_usd'].mean()
    metrics['avg_win'] = df_trades[df_trades['pnl_usd'] > 0]['pnl_usd'].mean() if metrics['wins'] > 0 else 0
    metrics['avg_loss'] = df_trades[df_trades['pnl_usd'] <= 0]['pnl_usd'].mean() if metrics['losses'] > 0 else 0
    
    metrics['max_win'] = df_trades['pnl_usd'].max()
    metrics['max_loss'] = df_trades['pnl_usd'].min()
    
    # Profit factor
    gross_profit = df_trades[df_trades['pnl_usd'] > 0]['pnl_usd'].sum()
    gross_loss = abs(df_trades[df_trades['pnl_usd'] < 0]['pnl_usd'].sum())
    metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Por tipo de cierre
    tp_trades = df_trades[df_trades['close_reason'] == 'TP']
    sl_trades = df_trades[df_trades['close_reason'] == 'SL']
    timeout_trades = df_trades[df_trades['close_reason'] == 'TIMEOUT']
    
    metrics['tp_count'] = len(tp_trades)
    metrics['sl_count'] = len(sl_trades)
    metrics['timeout_count'] = len(timeout_trades)
    
    # Por tipo de orden
    buy_trades = df_trades[df_trades['side'] == 'buy']
    sell_trades = df_trades[df_trades['side'] == 'sell']
    
    metrics['buy_count'] = len(buy_trades)
    metrics['sell_count'] = len(sell_trades)
    metrics['buy_winrate'] = (buy_trades['pnl_usd'] > 0).sum() / len(buy_trades) * 100 if len(buy_trades) > 0 else 0
    metrics['sell_winrate'] = (sell_trades['pnl_usd'] > 0).sum() / len(sell_trades) * 100 if len(sell_trades) > 0 else 0
    
    # Tiempo promedio
    metrics['avg_time_open'] = df_trades['time_open_min'].mean()
    
    # Mostrar resultados
    print(f"\n📈 RESULTADOS GENERALES:")
    print(f"   Total trades: {metrics['total_trades']}")
    print(f"   Ganadas: {metrics['wins']} ({metrics['win_rate']:.1f}%)")
    print(f"   Perdidas: {metrics['losses']}")
    print(f"   Win Rate: {metrics['win_rate']:.1f}%")
    print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
    
    print(f"\n💰 P&L:")
    print(f"   Total: ${metrics['total_pnl']:.2f}")
    print(f"   Promedio: ${metrics['avg_pnl']:.2f}")
    print(f"   Ganancia promedio: ${metrics['avg_win']:.2f}")
    print(f"   Pérdida promedio: ${metrics['avg_loss']:.2f}")
    print(f"   Mejor trade: ${metrics['max_win']:.2f}")
    print(f"   Peor trade: ${metrics['max_loss']:.2f}")
    
    print(f"\n🎯 CIERRES:")
    print(f"   Take Profit: {metrics['tp_count']}")
    print(f"   Stop Loss: {metrics['sl_count']}")
    print(f"   Timeout: {metrics['timeout_count']}")
    
    print(f"\n📊 POR TIPO:")
    print(f"   BUY: {metrics['buy_count']} trades (WR: {metrics['buy_winrate']:.1f}%)")
    print(f"   SELL: {metrics['sell_count']} trades (WR: {metrics['sell_winrate']:.1f}%)")
    
    print(f"\n⏱️ TIEMPO:")
    print(f"   Promedio abierto: {metrics['avg_time_open']:.1f} min")
    
    return metrics

def analyze_predictions(df_signals):
    """Analiza la efectividad de las predicciones"""
    
    print("\n" + "="*70)
    print("  🔮 ANÁLISIS DE PREDICCIONES")
    print("="*70)
    
    if len(df_signals) == 0:
        print("⚠️ No hay señales para analizar")
        return {}
    
    metrics = {}
    
    # Distribución de señales
    buy_signals = (df_signals['signal'] == 'BUY').sum()
    sell_signals = (df_signals['signal'] == 'SELL').sum()
    hold_signals = (df_signals['signal'] == 'HOLD').sum()
    
    metrics['total_signals'] = len(df_signals)
    metrics['buy_signals'] = buy_signals
    metrics['sell_signals'] = sell_signals
    metrics['hold_signals'] = hold_signals
    
    # Confianza promedio
    metrics['avg_confidence_buy'] = df_signals[df_signals['signal'] == 'BUY']['confidence'].mean()
    metrics['avg_confidence_sell'] = df_signals[df_signals['signal'] == 'SELL']['confidence'].mean()
    
    # Precisión de predicciones (comparar pred vs real)
    df_signals['pred_direction'] = np.where(df_signals['pred_change_%'] > 0, 'UP', 'DOWN')
    
    print(f"\n📊 SEÑALES GENERADAS:")
    print(f"   Total: {metrics['total_signals']}")
    print(f"   BUY: {buy_signals} ({buy_signals/metrics['total_signals']*100:.1f}%)")
    print(f"   SELL: {sell_signals} ({sell_signals/metrics['total_signals']*100:.1f}%)")
    print(f"   HOLD: {hold_signals} ({hold_signals/metrics['total_signals']*100:.1f}%)")
    
    print(f"\n🎯 CONFIANZA PROMEDIO:")
    print(f"   BUY: {metrics['avg_confidence_buy']:.1f}%")
    print(f"   SELL: {metrics['avg_confidence_sell']:.1f}%")
    
    return metrics

def create_visualizations(df_trades, df_signals):
    """Crea gráficas de rendimiento"""
    
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('Trading Bot - Análisis de Rendimiento', fontsize=16, fontweight='bold')
    
    # 1. Curva de equity
    if len(df_trades) > 0:
        ax1 = plt.subplot(3, 3, 1)
        df_trades = df_trades.sort_values('timestamp')
        df_trades['cumulative_pnl'] = df_trades['pnl_usd'].cumsum()
        ax1.plot(df_trades['timestamp'], df_trades['cumulative_pnl'], 'b-', linewidth=2)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax1.set_title('Curva de Equity', fontweight='bold')
        ax1.set_xlabel('Fecha')
        ax1.set_ylabel('P&L Acumulado ($)')
        ax1.grid(alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Distribución de P&L
        ax2 = plt.subplot(3, 3, 2)
        ax2.hist(df_trades['pnl_usd'], bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax2.axvline(0, color='red', linestyle='--', linewidth=2)
        ax2.set_title('Distribución de P&L', fontweight='bold')
        ax2.set_xlabel('P&L ($)')
        ax2.set_ylabel('Frecuencia')
        ax2.grid(alpha=0.3)
        
        # 3. Win/Loss por tipo
        ax3 = plt.subplot(3, 3, 3)
        buy_wins = (df_trades[df_trades['side'] == 'buy']['pnl_usd'] > 0).sum()
        buy_losses = (df_trades[df_trades['side'] == 'buy']['pnl_usd'] <= 0).sum()
        sell_wins = (df_trades[df_trades['side'] == 'sell']['pnl_usd'] > 0).sum()
        sell_losses = (df_trades[df_trades['side'] == 'sell']['pnl_usd'] <= 0).sum()
        
        x = np.arange(2)
        width = 0.35
        ax3.bar(x - width/2, [buy_wins, sell_wins], width, label='Wins', color='green', alpha=0.7)
        ax3.bar(x + width/2, [buy_losses, sell_losses], width, label='Losses', color='red', alpha=0.7)
        ax3.set_xticks(x)
        ax3.set_xticklabels(['BUY', 'SELL'])
        ax3.set_title('Win/Loss por Tipo', fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # 4. Cierre por razón
        ax4 = plt.subplot(3, 3, 4)
        close_reasons = df_trades['close_reason'].value_counts()
        colors_pie = {'TP': 'green', 'SL': 'red', 'TIMEOUT': 'orange'}
        colors = [colors_pie.get(r, 'gray') for r in close_reasons.index]
        ax4.pie(close_reasons.values, labels=close_reasons.index, autopct='%1.1f%%', colors=colors)
        ax4.set_title('Distribución de Cierres', fontweight='bold')
        
        # 5. P&L por día
        ax5 = plt.subplot(3, 3, 5)
        df_trades['date'] = df_trades['timestamp'].dt.date
        daily_pnl = df_trades.groupby('date')['pnl_usd'].sum()
        colors_bar = ['green' if x > 0 else 'red' for x in daily_pnl.values]
        ax5.bar(range(len(daily_pnl)), daily_pnl.values, color=colors_bar, alpha=0.7)
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax5.set_title('P&L Diario', fontweight='bold')
        ax5.set_xlabel('Día')
        ax5.set_ylabel('P&L ($)')
        ax5.grid(alpha=0.3)
        
        # 6. Tiempo de trades
        ax6 = plt.subplot(3, 3, 6)
        ax6.hist(df_trades['time_open_min'], bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax6.set_title('Distribución Tiempo Abierto', fontweight='bold')
        ax6.set_xlabel('Minutos')
        ax6.set_ylabel('Frecuencia')
        ax6.grid(alpha=0.3)
    
    # 7. Señales generadas
    if len(df_signals) > 0:
        ax7 = plt.subplot(3, 3, 7)
        signal_counts = df_signals['signal'].value_counts()
        colors_signals = {'BUY': 'green', 'SELL': 'red', 'HOLD': 'gray'}
        colors = [colors_signals.get(s, 'blue') for s in signal_counts.index]
        ax7.bar(signal_counts.index, signal_counts.values, color=colors, alpha=0.7)
        ax7.set_title('Señales Generadas', fontweight='bold')
        ax7.set_ylabel('Cantidad')
        ax7.grid(alpha=0.3)
        
        # 8. Confianza por señal
        ax8 = plt.subplot(3, 3, 8)
        buy_conf = df_signals[df_signals['signal'] == 'BUY']['confidence']
        sell_conf = df_signals[df_signals['signal'] == 'SELL']['confidence']
        ax8.boxplot([buy_conf, sell_conf], labels=['BUY', 'SELL'])
        ax8.set_title('Confianza por Tipo de Señal', fontweight='bold')
        ax8.set_ylabel('Confianza (%)')
        ax8.grid(alpha=0.3)
        
        # 9. Evolución de señales
        ax9 = plt.subplot(3, 3, 9)
        df_signals_sorted = df_signals.sort_values('timestamp')
        df_signals_sorted['cumulative_buy'] = (df_signals_sorted['signal'] == 'BUY').cumsum()
        df_signals_sorted['cumulative_sell'] = (df_signals_sorted['signal'] == 'SELL').cumsum()
        ax9.plot(df_signals_sorted['timestamp'], df_signals_sorted['cumulative_buy'], 'g-', label='BUY', linewidth=2)
        ax9.plot(df_signals_sorted['timestamp'], df_signals_sorted['cumulative_sell'], 'r-', label='SELL', linewidth=2)
        ax9.set_title('Evolución de Señales', fontweight='bold')
        ax9.legend()
        ax9.grid(alpha=0.3)
        ax9.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    output_path = 'trading_analytics.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

def generate_report():
    """Genera reporte completo"""
    
    print("="*70)
    print("  📊 GENERANDO REPORTE COMPLETO")
    print("="*70 + "\n")
    
    data = load_data()
    
    report = f"📊 *Reporte de Trading Bot*\n"
    report += f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    
    # Análisis de trades
    if 'trades' in data and len(data['trades']) > 0:
        metrics_trading = analyze_trading_performance(data['trades'])
        
        report += f"💼 *TRADING:*\n"
        report += f"Trades: {metrics_trading['total_trades']}\n"
        report += f"Win Rate: {metrics_trading['win_rate']:.1f}%\n"
        report += f"P&L Total: ${metrics_trading['total_pnl']:.2f}\n"
        report += f"Profit Factor: {metrics_trading['profit_factor']:.2f}\n\n"
    
    # Análisis de señales
    if 'signals' in data and len(data['signals']) > 0:
        metrics_signals = analyze_predictions(data['signals'])
        
        report += f"🎯 *SEÑALES:*\n"
        report += f"Total: {metrics_signals['total_signals']}\n"
        report += f"BUY: {metrics_signals['buy_signals']}\n"
        report += f"SELL: {metrics_signals['sell_signals']}\n"
        report += f"HOLD: {metrics_signals['hold_signals']}\n"
    
    # Crear visualizaciones
    if 'trades' in data or 'signals' in data:
        df_trades = data.get('trades', pd.DataFrame())
        df_signals = data.get('signals', pd.DataFrame())
        
        if len(df_trades) > 0 or len(df_signals) > 0:
            chart_path = create_visualizations(df_trades, df_signals)
            print(f"\n✅ Gráficas guardadas: {chart_path}")
            send_telegram(report, chart_path)
        else:
            send_telegram(report)
    else:
        send_telegram(report)
    
    print("\n" + "="*70)
    print("  ✅ REPORTE COMPLETADO")
    print("="*70)

if __name__ == "__main__":
    try:
        generate_report()
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        print(error_msg)
        send_telegram(error_msg)
        raise
