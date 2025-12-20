import requests
import json
import hmac
import hashlib
import base64
import time
import urllib.parse
import pandas as pd
import os
from datetime import datetime
from risk_manager import get_risk_manager

# Configuración Kraken
KRAKEN_API_KEY = os.environ.get('KRAKEN_API_KEY', '')
KRAKEN_API_SECRET = os.environ.get('KRAKEN_API_SECRET', '')
KRAKEN_API_URL = "https://api.kraken.com"

# Telegram
TELEGRAM_API = os.environ.get('TELEGRAM_API', '')
CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

# Archivos
TRADES_FILE = 'kraken_trades.csv'
OPEN_ORDERS_FILE = 'open_orders.json'

def send_telegram(msg):
    if not TELEGRAM_API or not CHAT_ID:
        print("⚠️ Telegram no configurado")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_API}/sendMessage"
        requests.post(url, data={'chat_id': CHAT_ID, 'text': msg, 'parse_mode': 'Markdown'}, timeout=10)
    except Exception as e:
        print(f"❌ Telegram: {e}")

def kraken_signature(urlpath, data, secret):
    postdata = urllib.parse.urlencode(data)
    encoded = (str(data['nonce']) + postdata).encode()
    message = urlpath.encode() + hashlib.sha256(encoded).digest()
    mac = hmac.new(base64.b64decode(secret), message, hashlib.sha512)
    sigdigest = base64.b64encode(mac.digest())
    return sigdigest.decode()

def kraken_request(uri_path, data):
    headers = {
        'API-Key': KRAKEN_API_KEY,
        'API-Sign': kraken_signature(uri_path, data, KRAKEN_API_SECRET)
    }
    req = requests.post(KRAKEN_API_URL + uri_path, headers=headers, data=data)
    return req.json()

def get_current_price():
    url = f"{KRAKEN_API_URL}/0/public/Ticker?pair=ETHUSD"
    r = requests.get(url).json()
    if 'result' in r and 'XETHZUSD' in r['result']:
        return float(r['result']['XETHZUSD']['c'][0])
    return None

def get_balance():
    data = {'nonce': str(int(1000*time.time()))}
    result = kraken_request('/0/private/Balance', data)
    return result

def place_order(side, volume, price, tp_price, sl_price):
    """
    side: 'buy' o 'sell'
    volume: cantidad en ETH
    price: precio límite (None para market order)
    tp_price: take profit
    sl_price: stop loss
    """
    data = {
        'nonce': str(int(1000*time.time())),
        'ordertype': 'limit' if price else 'market',
        'type': side,
        'volume': str(volume),
        'pair': 'XETHZUSD',
    }
    
    if price:
        data['price'] = str(price)
    
    result = kraken_request('/0/private/AddOrder', data)
    return result

def cancel_order(txid):
    data = {
        'nonce': str(int(1000*time.time())),
        'txid': txid
    }
    result = kraken_request('/0/private/CancelOrder', data)
    return result

def get_open_orders():
    data = {'nonce': str(int(1000*time.time()))}
    result = kraken_request('/0/private/OpenOrders', data)
    return result

def calculate_tp_sl(entry_price, side, atr, pred_high, pred_low, tp_percentage=0.80):
    """
    Calcula TP al 80% de la predicción y SL con ATR
    """
    if side == 'buy':
        target_move = pred_high - entry_price
        tp = entry_price + (target_move * tp_percentage)
        sl = entry_price - (atr * 2)
    else:
        target_move = entry_price - pred_low
        tp = entry_price - (target_move * tp_percentage)
        sl = entry_price + (atr * 2)
    
    return round(tp, 2), round(sl, 2)

# ... (mantener todo el código anterior hasta la línea 80) ...

def monitor_orders():
    """Monitorea órdenes abiertas y cierra por TP/SL/tiempo - CADA 5 MINUTOS"""
    if not os.path.exists(OPEN_ORDERS_FILE):
        return
    
    with open(OPEN_ORDERS_FILE, 'r') as f:
        orders = json.load(f)
    
    if len(orders) == 0:
        print("ℹ️ No hay órdenes abiertas para monitorear")
        return
    
    current_price = get_current_price()
    if not current_price:
        print("❌ No se pudo obtener precio actual")
        return
    
    # Cargar Risk Manager
    risk_manager = get_risk_manager()
    
    updated_orders = []
    
    for order in orders:
        txid = order['txid']
        entry_price = order['entry_price']
        side = order['side']
        tp = order['tp']
        sl = order['sl']
        open_time = datetime.fromisoformat(order['open_time'])
        volume = order['volume']
        
        time_open = (datetime.now() - open_time).total_seconds() / 60  # en minutos
        
        should_close = False
        close_reason = None
        close_price = current_price
        
        # ═══════════════════════════════════════════════════════════
        # CONFIGURACIÓN DE CIERRES - AJUSTA AQUÍ
        # ═══════════════════════════════════════════════════════════
        
        # 1. Verificar TP (Take Profit)
        if side == 'buy' and current_price >= tp:
            should_close = True
            close_reason = 'TP'
        elif side == 'sell' and current_price <= tp:
            should_close = True
            close_reason = 'TP'
        
        # 2. Verificar SL (Stop Loss)
        elif side == 'buy' and current_price <= sl:
            should_close = True
            close_reason = 'SL'
        elif side == 'sell' and current_price >= sl:
            should_close = True
            close_reason = 'SL'
        
        # 3. TIMEOUT - Cerrar después de X minutos
        # ⚠️ AJUSTA AQUÍ EL TIMEOUT ⚠️
        elif time_open >= 300:  # 300 minutos = 5 horas
            should_close = True
            close_reason = 'TIMEOUT'
        
        # 4. STOP LOSS PROGRESIVO (Opcional)
        # Cierra si pierde más de X% en los primeros Y minutos
        elif time_open <= 10:  # Primeros 10 minutos
            loss_pct = ((current_price - entry_price) / entry_price) * 100
            if side == 'buy' and loss_pct < -1.0:  # Pierde más de 1%
                should_close = True
                close_reason = 'QUICK_LOSS'
            elif side == 'sell' and loss_pct > 1.0:
                should_close = True
                close_reason = 'QUICK_LOSS'
        
        # ═══════════════════════════════════════════════════════════
        
        if should_close:
            print(f"🔴 Cerrando orden {txid[:8]}... por {close_reason}")
            print(f"   Tiempo abierto: {time_open:.1f} min")
            print(f"   Precio entrada: ${entry_price:.2f}")
            print(f"   Precio cierre: ${close_price:.2f}")
            
            # Cancelar en Kraken
            # ⚠️ DESCOMENTAR PARA TRADING REAL ⚠️
            # cancel_result = cancel_order(txid)
            # print(f"   Kraken cancel: {cancel_result}")
            
            print("   ⚠️ MODO SIMULACIÓN - Orden NO cancelada en Kraken")
            
            # Calcular P&L
            if side == 'buy':
                pnl = (close_price - entry_price) * volume
                pnl_pct = ((close_price - entry_price) / entry_price) * 100
            else:
                pnl = (entry_price - close_price) * volume
                pnl_pct = ((entry_price - close_price) / entry_price) * 100
            
            # Actualizar capital en Risk Manager
            risk_manager.update_after_trade(pnl)
            
            # Guardar en CSV
            trade_data = {
                'timestamp': datetime.now(),
                'txid': txid,
                'side': side,
                'entry_price': entry_price,
                'close_price': close_price,
                'volume': volume,
                'tp': tp,
                'sl': sl,
                'close_reason': close_reason,
                'time_open_min': time_open,
                'pnl_usd': pnl,
                'pnl_%': pnl_pct
            }
            
            df = pd.DataFrame([trade_data])
            if os.path.exists(TRADES_FILE):
                df.to_csv(TRADES_FILE, mode='a', header=False, index=False)
            else:
                df.to_csv(TRADES_FILE, index=False)
            
            # Telegram
            emoji = "✅" if pnl > 0 else "❌"
            stats = risk_manager.get_stats()
            
            msg = f"""
🚀 *Nueva Orden Ejecutada*
 *Orden Cerrada*

📖 ID: {txid[:8]}...
📊 Tipo: {side.upper()}
💰 Entrada: ${entry_price:.2f}
💰 Salida: ${close_price:.2f}
🎯 Razón: {close_reason}
⏱️ Tiempo: {time_open:.1f} min

💵 P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)

📈 *Capital:*
   Actual: ${stats['current_capital']:.2f}
   Total: ${stats['total_profit']:+.2f} ({stats['profit_%']:+.2f}%)
   WR: {stats['win_rate']:.1f}% ({stats['win_count']}/{stats['total_trades']})
"""
            send_telegram(msg)
        else:
            # Mantener orden abierta
            updated_orders.append(order)
            
            # Log de seguimiento
            time_left = 300 - time_open  # Asumiendo timeout de 300 min
            print(f"📊 {txid[:8]}... | {side.upper()} | {time_open:.1f}min | Quedan {time_left:.1f}min")
    
    # Actualizar archivo
    with open(OPEN_ORDERS_FILE, 'w') as f:
        json.dump(updated_orders, f, indent=2)
    
    if len(updated_orders) > 0:
        print(f"✅ Monitoreo completado: {len(updated_orders)} órdenes activas")
    else:
        print("✅ Todas las órdenes fueron cerradas")

# ... (mantener resto del código igual) ...

def execute_signal():
    """Lee última señal y ejecuta si es BUY/SELL con gestión de riesgo"""
    
    # Leer última señal
    signals_file = 'trading_signals.csv'
    if not os.path.exists(signals_file):
        print("❌ No hay señales disponibles")
        return
    
    df = pd.read_csv(signals_file)
    latest = df.iloc[-1]
    
    signal = latest['signal']
    
    if signal == 'HOLD':
        print("⏸️ Señal HOLD - No hay acción")
        return
    
    # Cargar Risk Manager
    risk_manager = get_risk_manager()
    risk_manager.print_stats()
    
    # Verificar si ya hay orden abierta
    if os.path.exists(OPEN_ORDERS_FILE):
        with open(OPEN_ORDERS_FILE, 'r') as f:
            open_orders = json.load(f)
        if len(open_orders) >= risk_manager.max_open_positions:
            print(f"⚠️ Máximo de posiciones ({risk_manager.max_open_positions}) alcanzado")
            return
    
    # Obtener datos necesarios
    current_price = get_current_price()
    if not current_price:
        print("❌ No se pudo obtener precio actual")
        return
    
    atr = latest['atr']
    pred_high = latest['pred_high']
    pred_low = latest['pred_low']
    confidence = latest['confidence']
    
    # Calcular TP y SL
    side = signal.lower()
    tp, sl = calculate_tp_sl(current_price, side, atr, pred_high, pred_low, tp_percentage=0.80)
    
    # 🔥 VALIDAR TRADE CON RISK MANAGER
    print(f"\n{'='*70}")
    print(f"  VALIDANDO TRADE")
    print(f"{'='*70}")
    
    # 1. Validar Risk/Reward
    trade_validation = risk_manager.validate_trade(current_price, tp, sl, side)
    
    if not trade_validation['valid']:
        print(f"❌ Trade rechazado: {trade_validation['reason']}")
        msg = f"⛔ *Trade Rechazado*\n\n📊 {signal}\n❌ {trade_validation['reason']}"
        send_telegram(msg)
        return
    
    print(f"✅ R/R Ratio: {trade_validation['rr_ratio']:.2f}")
    print(f"   Risk: ${trade_validation['risk']:.2f}")
    print(f"   Reward: ${trade_validation['reward']:.2f}")
    
    # 2. Calcular tamaño de posición
    position = risk_manager.calculate_position_size(current_price, sl, confidence, side)
    
    if not position['valid']:
        print(f"❌ Posición rechazada: {position['reason']}")
        msg = f"⛔ *Posición Rechazada*\n\n📊 {signal}\n❌ {position['reason']}"
        send_telegram(msg)
        return
    
    volume = position['volume']
    
    print(f"\n{'='*70}")
    print(f"🚀 EJECUTANDO ORDEN CON GESTIÓN DE RIESGO")
    print(f"{'='*70}")
    print(f"📊 Señal: {signal}")
    print(f"💰 Precio: ${current_price:.2f}")
    print(f"📈 Volumen: {volume} ETH (${position['position_value']:.2f})")
    print(f"   • Riesgo: ${position['risk_amount']:.2f} ({risk_manager.risk_per_trade*100}%)")
    print(f"   • Capital usado: {position['capital_used_%']:.1f}%")
    print(f"   • Mult. confianza: {position['confidence_multiplier']:.2f}x")
    print(f"🎯 TP: ${tp:.2f} ({((tp-current_price)/current_price*100):+.2f}%)")
    print(f"🛑 SL: ${sl:.2f} ({((sl-current_price)/current_price*100):+.2f}%)")
    print(f"📊 R/R: {trade_validation['rr_ratio']:.2f}")
    print(f"🎲 Confianza: {confidence:.1f}%")
    print(f"{'='*70}\n")
    
    # SIMULACIÓN - Descomentar para trading real
    print("⚠️ MODO SIMULACIÓN - Orden NO enviada a Kraken")
    print("   Para trading real, descomentar bloque de código")
    
    # Descomentar para ejecutar en real:
    
    result = place_order(side, volume, None, tp, sl)
    
    if 'result' in result and 'txid' in result['result']:
        txid = result['result']['txid'][0]
        print(f"✅ Orden ejecutada: {txid}")
        
        # Guardar orden abierta
        order_data = {
            'txid': txid,
            'side': side,
            'entry_price': current_price,
            'volume': volume,
            'tp': tp,
            'sl': sl,
            'open_time': datetime.now().isoformat(),
            'signal_confidence': confidence,
            'rr_ratio': trade_validation['rr_ratio'],
            'risk_amount': position['risk_amount']
        }
        
        orders = []
        if os.path.exists(OPEN_ORDERS_FILE):
            with open(OPEN_ORDERS_FILE, 'r') as f:
                orders = json.load(f)
        
        orders.append(order_data)
        with open(OPEN_ORDERS_FILE, 'w') as f:
            json.dump(orders, f, indent=2)
        
        # Registro en CSV
        trade_data = {
            'timestamp': datetime.now(),
            'txid': txid,
            'side': side,
            'entry_price': current_price,
            'volume': volume,
            'tp': tp,
            'sl': sl,
            'confidence': confidence,
            'rr_ratio': trade_validation['rr_ratio'],
            'risk_amount': position['risk_amount'],
            'order_executed': 'YES',
            'order_type': signal
        }
        
        df = pd.DataFrame([trade_data])
        exec_file = 'orders_executed.csv'
        if os.path.exists(exec_file):
            df.to_csv(exec_file, mode='a', header=False, index=False)
        else:
            df.to_csv(exec_file, index=False)
        
        # Telegram
        stats = risk_manager.get_stats()
        msg = f"""
🚀 *Nueva Orden Ejecutada*

📊 Tipo: {signal}
💰 Entrada: ${current_price:.2f}
📈 Volumen: {volume} ETH (${position['position_value']:.2f})
   • Riesgo: ${position['risk_amount']:.2f}
   • Capital: {position['capital_used_%']:.1f}%

🎯 TP: ${tp:.2f} ({((tp-current_price)/current_price*100):+.2f}%)
🛑 SL: ${sl:.2f} ({((sl-current_price)/current_price*100):+.2f}%)
📊 R/R: {trade_validation['rr_ratio']:.2f}
🎲 Confianza: {confidence:.1f}%

📈 *Estado Cuenta:*
   Capital: ${stats['current_capital']:.2f}
   Posiciones: {stats['open_positions']}/{risk_manager.max_open_positions}
   WR: {stats['win_rate']:.1f}%
"""
        send_telegram(msg)
        
    else:
        error = result.get('error', 'Unknown error')
        print(f"❌ Error al ejecutar orden: {error}")
        send_telegram(f"❌ Error ejecutando orden: {error}")
    

def main():
    print("="*70)
    print("  🤖 KRAKEN TRADER BOT - CON GESTIÓN DE RIESGO")
    print("="*70)
    
    # 1. Monitorear órdenes existentes
    print("\n🔍 Monitoreando órdenes abiertas...")
    monitor_orders()
    
    # 2. Verificar nueva señal y ejecutar si corresponde
    print("\n📊 Verificando nuevas señales...")
    execute_signal()
    
    # 3. Mostrar resumen
    risk_manager = get_risk_manager()
    risk_manager.print_stats()
    
    if os.path.exists(TRADES_FILE):
        df = pd.read_csv(TRADES_FILE)
        if len(df) > 0:
            total_pnl = df['pnl_usd'].sum()
            win_rate = (df['pnl_usd'] > 0).sum() / len(df) * 100
            
            print(f"\n{'='*70}")
            print(f"📊 RESUMEN DE TRADING")
            print(f"{'='*70}")
            print(f"Total trades: {len(df)}")
            print(f"Win rate: {win_rate:.1f}%")
            print(f"P&L total: ${total_pnl:.2f}")
            print(f"{'='*70}\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        print(error_msg)
        send_telegram(error_msg)
        raise
