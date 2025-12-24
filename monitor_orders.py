"""
MONITOR DE ÓRDENES - SCRIPT INDEPENDIENTE
✅ Solo monitorea y cierra posiciones
✅ No abre nuevas posiciones
✅ Se ejecuta cada 4-5 minutos
"""

import pandas as pd
import os
import json
import time
import hmac
import hashlib
import base64
import urllib.parse
import requests
from datetime import datetime, timedelta

# Configuración
KRAKEN_API_KEY = os.environ.get('KRAKEN_API_KEY', '')
KRAKEN_API_SECRET = os.environ.get('KRAKEN_API_SECRET', '')
TELEGRAM_API = os.environ.get('TELEGRAM_API', '')
CHAT_ID = os.environ.get('CHAT_ID', '')

PAIR = 'ADAUSD'
TRADES_FILE = 'kraken_trades.csv'
OPEN_ORDERS_FILE = 'open_orders.json'
MAX_POSITION_TIME_HOURS = 3.5  # Cerrar antes del rollover

def send_telegram(msg):
    """Envía mensaje a Telegram"""
    if not TELEGRAM_API or not CHAT_ID:
        return
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_API}/sendMessage"
        data = {'chat_id': CHAT_ID, 'text': msg, 'parse_mode': 'Markdown'}
        requests.post(url, data=data, timeout=10)
    except Exception as e:
        print(f"⚠️ Error Telegram: {e}")

def kraken_request(uri_path, data):
    """Request autenticado a Kraken API"""
    if not KRAKEN_API_KEY or not KRAKEN_API_SECRET:
        raise ValueError("⚠️ API keys no configuradas")
    
    api_nonce = str(int(time.time() * 1000))
    data['nonce'] = api_nonce
    
    postdata = urllib.parse.urlencode(data)
    encoded = (api_nonce + postdata).encode()
    message = uri_path.encode() + hashlib.sha256(encoded).digest()
    
    signature = hmac.new(
        base64.b64decode(KRAKEN_API_SECRET),
        message,
        hashlib.sha512
    )
    sigdigest = base64.b64encode(signature.digest())
    
    headers = {
        'API-Key': KRAKEN_API_KEY,
        'API-Sign': sigdigest.decode()
    }
    
    url = f"https://api.kraken.com{uri_path}"
    
    try:
        response = requests.post(url, headers=headers, data=data, timeout=30)
        result = response.json()
        
        if result.get('error') and len(result['error']) > 0:
            print(f"❌ Kraken Error: {result['error']}")
            return None
        
        return result.get('result')
        
    except Exception as e:
        print(f"❌ Request error: {e}")
        return None

def get_current_price():
    """Obtiene precio actual de ADAUSD"""
    try:
        url = f"https://api.kraken.com/0/public/Ticker?pair={PAIR}"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if data.get('error') and len(data['error']) > 0:
            return None
        
        pair_key = list(data['result'].keys())[0]
        price = float(data['result'][pair_key]['c'][0])
        return price
        
    except Exception as e:
        print(f"❌ Error obteniendo precio: {e}")
        return None

def sync_open_orders_with_kraken():
    """Sincroniza open_orders.json con Kraken"""
    print("\n🔄 Sincronizando con Kraken...")
    
    if not os.path.exists(OPEN_ORDERS_FILE):
        print("✅ No hay órdenes locales")
        return {}
    
    with open(OPEN_ORDERS_FILE, 'r') as f:
        local_orders = json.load(f)
    
    if len(local_orders) == 0:
        print("✅ No hay órdenes locales")
        return {}
    
    print(f"📋 Verificando {len(local_orders)} orden(es)...")
    
    result = kraken_request('/0/private/OpenOrders', {})
    
    if not result:
        print("⚠️ No se pudo consultar Kraken")
        return local_orders
    
    kraken_open_orders = result.get('open', {})
    orders_to_remove = []
    
    for order_id in local_orders.keys():
        if order_id not in kraken_open_orders:
            print(f"🗑️ Orden {order_id} cerrada en Kraken")
            orders_to_remove.append(order_id)
    
    for order_id in orders_to_remove:
        del local_orders[order_id]
    
    with open(OPEN_ORDERS_FILE, 'w') as f:
        json.dump(local_orders, f, indent=2)
    
    if len(orders_to_remove) > 0:
        print(f"✅ {len(orders_to_remove)} orden(es) eliminada(s)")
    
    print(f"📊 {len(local_orders)} orden(es) activas")
    return local_orders

def close_position(order_id, side, volume):
    """Cierra una posición"""
    print(f"\n🔄 Cerrando posición {order_id}...")
    
    close_side = 'sell' if side == 'buy' else 'buy'
    
    close_data = {
        'pair': PAIR,
        'type': close_side,
        'ordertype': 'market',
        'volume': str(volume),
        'leverage': '0'
    }
    
    result = kraken_request('/0/private/AddOrder', close_data)
    
    if result:
        print(f"✅ Posición cerrada: {result['txid'][0]}")
        return True
    else:
        print(f"❌ Error al cerrar posición")
        return False

def save_closed_trade(order_info, close_price, pnl_usd, pnl_pct, close_reason):
    """Guarda trade cerrado en kraken_trades.csv"""
    trade_data = {
        'open_time': order_info['entry_time'],
        'close_time': datetime.now(),
        'order_id': order_info['order_id'],
        'side': order_info['side'],
        'volume': order_info['volume'],
        'leverage': order_info['leverage'],
        'entry_price': order_info['entry_price'],
        'close_price': close_price,
        'take_profit': order_info['take_profit'],
        'stop_loss': order_info['stop_loss'],
        'pnl_usd': pnl_usd,
        'pnl_%': pnl_pct,
        'close_reason': close_reason,
        'margin_used': order_info['margin_used']
    }
    
    df_trade = pd.DataFrame([trade_data])
    
    if os.path.exists(TRADES_FILE):
        df_trade.to_csv(TRADES_FILE, mode='a', header=False, index=False)
    else:
        df_trade.to_csv(TRADES_FILE, index=False)
    
    print(f"✅ Trade guardado en {TRADES_FILE}")

def monitor_orders_only():
    """
    🔥 FUNCIÓN PRINCIPAL - SOLO MONITOREO
    No ejecuta nuevas órdenes, solo monitorea las existentes
    """
    print("\n" + "="*70)
    print("  👀 MONITOREANDO ÓRDENES ABIERTAS")
    print("="*70)
    
    # 1. Sincronizar con Kraken
    orders = sync_open_orders_with_kraken()
    
    if len(orders) == 0:
        print("\n✅ No hay órdenes abiertas para monitorear")
        return
    
    # 2. Obtener precio actual
    current_price = get_current_price()
    
    if not current_price:
        print("❌ No se pudo obtener precio actual")
        return
    
    print(f"\n💰 Precio actual: ${current_price:.4f}")
    
    # 3. Revisar cada orden
    orders_closed = 0
    
    for order_id, order_info in list(orders.items()):
        print(f"\n{'─'*70}")
        print(f"📊 Orden: {order_id[:16]}...")
        print(f"   Lado: {order_info['side'].upper()}")
        print(f"   Entry: ${order_info['entry_price']:.4f}")
        print(f"   Current: ${current_price:.4f}")
        print(f"   TP: ${order_info['take_profit']:.4f}")
        print(f"   SL: ${order_info['stop_loss']:.4f}")
        
        # Calcular P&L
        if order_info['side'] == 'buy':
            pnl_pct = ((current_price - order_info['entry_price']) / order_info['entry_price']) * 100
        else:
            pnl_pct = ((order_info['entry_price'] - current_price) / order_info['entry_price']) * 100
        
        pnl_usd = (order_info['volume'] * order_info['entry_price']) * (pnl_pct / 100)
        
        print(f"   P&L: {pnl_pct:+.2f}% (${pnl_usd:+.2f})")
        
        # Determinar si cerrar
        close_reason = None
        
        # ✅ CHECK 1: Take Profit alcanzado
        if order_info['side'] == 'buy':
            if current_price >= order_info['take_profit']:
                close_reason = 'TP_HIT'
        else:
            if current_price <= order_info['take_profit']:
                close_reason = 'TP_HIT'
        
        # ✅ CHECK 2: Timeout (evitar rollover fees)
        entry_time = datetime.fromisoformat(order_info['entry_time'])
        time_open = datetime.now() - entry_time
        time_open_hours = time_open.total_seconds() / 3600
        
        print(f"   Tiempo abierto: {time_open_hours:.1f}h")
        
        if time_open_hours >= MAX_POSITION_TIME_HOURS and not close_reason:
            close_reason = 'TIMEOUT'
            print(f"   ⏰ Timeout alcanzado ({MAX_POSITION_TIME_HOURS}h)")
        
        # ✅ CHECK 3: Stop Loss (el stop-loss automático debería manejarlo, pero check)
        if order_info['side'] == 'buy':
            if current_price <= order_info['stop_loss']:
                close_reason = 'SL_HIT'
        else:
            if current_price >= order_info['stop_loss']:
                close_reason = 'SL_HIT'
        
        # 🔥 CERRAR POSICIÓN SI HAY RAZÓN
        if close_reason:
            print(f"\n🚨 Cerrando por: {close_reason}")
            
            success = close_position(
                order_id,
                order_info['side'],
                order_info['volume']
            )
            
            if success:
                # Guardar en trades
                save_closed_trade(
                    order_info,
                    current_price,
                    pnl_usd,
                    pnl_pct,
                    close_reason
                )
                
                # Eliminar de open_orders
                del orders[order_id]
                orders_closed += 1
                
                # Notificar
                msg = f"""🔔 *Posición Cerrada*

🆔 {order_id[:16]}...
📊 {order_info['side'].upper()}
💰 Entry: ${order_info['entry_price']:.4f}
💵 Close: ${current_price:.4f}

{'📈' if pnl_usd > 0 else '📉'} *P&L: {pnl_pct:+.2f}% (${pnl_usd:+.2f})*

🏷️ Razón: {close_reason}
⏱️ Duración: {time_open_hours:.1f}h
"""
                send_telegram(msg)
        else:
            print(f"   ✅ Posición sigue abierta")
    
    # 4. Guardar órdenes actualizadas
    with open(OPEN_ORDERS_FILE, 'w') as f:
        json.dump(orders, f, indent=2)
    
    # 5. Resumen
    print(f"\n{'='*70}")
    print(f"  📊 RESUMEN DE MONITOREO")
    print(f"{'='*70}")
    print(f"✅ Posiciones cerradas: {orders_closed}")
    print(f"📊 Posiciones activas: {len(orders)}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    try:
        monitor_orders_only()
    except Exception as e:
        error_msg = f"❌ Error en monitoreo: {str(e)}"
        print(error_msg)
        send_telegram(error_msg)
        raise
