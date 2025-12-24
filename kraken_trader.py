"""
KRAKEN TRADER - VERSIÓN CORREGIDA PARA MARGIN

✅ Lee balance de MARGIN (no spot)
✅ Ejecuta órdenes EN KRAKEN
✅ Protección contra comisiones del 100%
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
from risk_manager import get_risk_manager

# Configuración
KRAKEN_API_KEY = os.environ.get('KRAKEN_API_KEY', '')
KRAKEN_API_SECRET = os.environ.get('KRAKEN_API_SECRET', '')
TELEGRAM_API = os.environ.get('TELEGRAM_API', '')
CHAT_ID = os.environ.get('CHAT_ID', '')

PAIR = 'ADAUSD'
SIGNALS_FILE = 'trading_signals.csv'
ORDERS_FILE = 'orders_executed.csv'
TRADES_FILE = 'kraken_trades.csv'
OPEN_ORDERS_FILE = 'open_orders.json'

def send_telegram(msg):
    """Envía mensaje a Telegram"""
    if not TELEGRAM_API or not CHAT_ID:
        print("⚠️ Telegram no configurado")
        return
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_API}/sendMessage"
        data = {'chat_id': CHAT_ID, 'text': msg, 'parse_mode': 'Markdown'}
        requests.post(url, data=data, timeout=10)
        print("✅ Mensaje enviado a Telegram")
    except Exception as e:
        print(f"❌ Error Telegram: {e}")

def kraken_request(uri_path, data):
    """Hace request autenticado a Kraken API"""
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

def get_account_balance():
    """
    🔥 FIXED: Obtiene balance de MARGIN (no spot)
    """
    print("\n💰 Obteniendo balance de Kraken (MARGIN)...")
    
    # 1. Primero intentar TradeBalance (incluye margin)
    result = kraken_request('/0/private/TradeBalance', {'asset': 'ZUSD'})
    
    if result:
        # eb = equivalent balance (spot + margin disponible)
        margin_balance = float(result.get('eb', 0))
        free_margin = float(result.get('mf', 0))  # margin free
        used_margin = float(result.get('m', 0))   # margin used
        
        print(f"📊 Balance de Trading:")
        print(f"   💵 Total disponible: ${margin_balance:.2f}")
        print(f"   ✅ Margen libre: ${free_margin:.2f}")
        print(f"   🔒 Margen usado: ${used_margin:.2f}")
        
        if free_margin < 5:
            print(f"⚠️ Margen libre insuficiente para operar (mínimo $5)")
        
        return free_margin  # Devolver solo el margen LIBRE
    
    # 2. Fallback: Balance normal (por si acaso)
    print("⚠️ TradeBalance falló, intentando Balance normal...")
    result_spot = kraken_request('/0/private/Balance', {})
    
    if not result_spot:
        print("❌ No se pudo obtener balance")
        return None
    
    usd_balance = float(result_spot.get('ZUSD', result_spot.get('USD', 0)))
    
    print(f"📊 Balance spot:")
    for currency, amount in result_spot.items():
        if float(amount) > 0:
            print(f"   • {currency}: {float(amount):.2f}")
    
    print(f"✅ Balance USD disponible: ${usd_balance:.2f}")
    
    if usd_balance < 5:
        print(f"⚠️ Balance insuficiente para operar (mínimo $5)")
    
    return usd_balance

def get_current_price():
    """Obtiene precio actual de ADAUSD"""
    try:
        url = f"https://api.kraken.com/0/public/Ticker?pair={PAIR}"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if data.get('error') and len(data['error']) > 0:
            print(f"❌ Error obteniendo precio: {data['error']}")
            return None
        
        pair_key = list(data['result'].keys())[0]
        price = float(data['result'][pair_key]['c'][0])
        
        print(f"💲 Precio actual {PAIR}: ${price:.4f}")
        return price
        
    except Exception as e:
        print(f"❌ Error obteniendo precio: {e}")
        return None

def load_last_signal():
    """Carga la última señal generada"""
    print(f"\n🔍 Buscando señales en {SIGNALS_FILE}...")
    
    if not os.path.exists(SIGNALS_FILE):
        print(f"⚠️ No existe {SIGNALS_FILE}")
        return None
    
    try:
        df = pd.read_csv(SIGNALS_FILE)
        
        if len(df) == 0:
            print("⚠️ CSV vacío")
            return None
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp', ascending=False)
        
        last_signal = df.iloc[0]
        
        signal_age = datetime.now() - last_signal['timestamp']
        
        if signal_age > timedelta(hours=2):
            print(f"⚠️ Señal demasiado antigua ({signal_age})")
            return None
        
        print(f"✅ Señal encontrada:")
        print(f"   Timestamp: {last_signal['timestamp']}")
        print(f"   Signal: {last_signal['signal']}")
        print(f"   Confidence: {last_signal['confidence']:.1f}%")
        print(f"   Price: ${last_signal['current_price']:.4f}")
        
        return last_signal.to_dict()
        
    except Exception as e:
        print(f"❌ Error leyendo señales: {e}")
        return None

def check_existing_orders():
    """Verifica si ya hay órdenes abiertas"""
    if os.path.exists(OPEN_ORDERS_FILE):
        try:
            with open(OPEN_ORDERS_FILE, 'r') as f:
                orders = json.load(f)
            
            if len(orders) > 0:
                print(f"⚠️ Ya hay {len(orders)} orden(es) abierta(s)")
                return True
        except:
            pass
    
    return False

def place_margin_order(side, volume, leverage, entry_price=None):
    """
    Coloca orden de MARGIN en Kraken
    """
    print(f"\n📤 Colocando orden MARGIN {side.upper()}...")
    print(f"   Volumen: {volume} ADA")
    print(f"   Leverage: {leverage}x")
    
    order_data = {
        'pair': PAIR,
        'type': side,
        'ordertype': 'market' if entry_price is None else 'limit',
        'volume': str(volume),
        'leverage': str(leverage),
        'oflags': 'post'
    }
    
    if entry_price:
        order_data['price'] = str(entry_price)
    
    result = kraken_request('/0/private/AddOrder', order_data)
    
    if not result:
        print("❌ Error al colocar orden")
        return None
    
    order_id = result['txid'][0]
    
    print(f"✅ Orden colocada: {order_id}")
    
    return {
        'order_id': order_id,
        'side': side,
        'volume': volume,
        'leverage': leverage,
        'timestamp': datetime.now().isoformat()
    }

def save_order_to_tracking(order_info, signal_info, position_info):
    """Guarda orden en archivos de tracking"""
    
    order_data = {
        'timestamp': datetime.now(),
        'order_id': order_info['order_id'],
        'side': order_info['side'],
        'volume': order_info['volume'],
        'leverage': order_info['leverage'],
        'entry_price': signal_info['current_price'],
        'confidence': signal_info['confidence'],
        'margin_used': position_info['margin_required'],
        'liquidation_price': position_info['liquidation_price'],
        'expected_tp': signal_info.get('pred_close', 0),
        'expected_risk': position_info['risk_amount']
    }
    
    df_order = pd.DataFrame([order_data])
    
    if os.path.exists(ORDERS_FILE):
        df_order.to_csv(ORDERS_FILE, mode='a', header=False, index=False)
    else:
        df_order.to_csv(ORDERS_FILE, index=False)
    
    print(f"✅ Orden guardada en {ORDERS_FILE}")
    
    open_order = {
        'order_id': order_info['order_id'],
        'side': order_info['side'],
        'volume': order_info['volume'],
        'leverage': order_info['leverage'],
        'entry_price': signal_info['current_price'],
        'entry_time': datetime.now().isoformat(),
        'stop_loss': signal_info['current_price'] * 0.98 if order_info['side'] == 'buy' else signal_info['current_price'] * 1.02,
        'take_profit': signal_info.get('pred_close', signal_info['current_price'] * 1.03),
        'margin_used': position_info['margin_required'],
        'liquidation_price': position_info['liquidation_price']
    }
    
    if os.path.exists(OPEN_ORDERS_FILE):
        with open(OPEN_ORDERS_FILE, 'r') as f:
            orders = json.load(f)
    else:
        orders = {}
    
    orders[order_info['order_id']] = open_order
    
    with open(OPEN_ORDERS_FILE, 'w') as f:
        json.dump(orders, f, indent=2)
    
    print(f"✅ Orden guardada en {OPEN_ORDERS_FILE}")

def execute_trading_strategy():
    """
    🔥 FUNCIÓN PRINCIPAL - Ejecuta estrategia de trading
    """
    print("="*70)
    print("  💼 EJECUTANDO ESTRATEGIA DE TRADING")
    print("="*70 + "\n")
    
    if check_existing_orders():
        print("\n⏸️ Ya hay posiciones abiertas. Saltando ejecución.")
        return
    
    signal = load_last_signal()
    
    if not signal:
        print("\n⚠️ No hay señales válidas para ejecutar")
        return
    
    if signal['signal'] == 'HOLD':
        print(f"\n⏸️ Señal es HOLD. No se ejecuta trade.")
        return
    
    print(f"\n🎯 Procesando señal: {signal['signal']}")
    print(f"   Confianza: {signal['confidence']:.1f}%")
    
    # 🔥 AQUÍ USA LA FUNCIÓN CORREGIDA
    balance = get_account_balance()
    
    if not balance or balance < 5:
        msg = f"❌ Balance insuficiente: ${balance:.2f} (mínimo $5)"
        print(msg)
        send_telegram(msg)
        return
    
    rm = get_risk_manager()
    rm.sync_with_kraken_balance(balance)
    
    current_price = get_current_price()
    
    if not current_price:
        print("❌ No se pudo obtener precio actual")
        return
    
    side = signal['signal'].lower()
    
    if side == 'buy':
        stop_loss = current_price * 0.98
        take_profit = signal.get('pred_close', current_price * 1.03)
    else:
        stop_loss = current_price * 1.02
        take_profit = signal.get('pred_close', current_price * 0.97)
    
    print(f"\n📊 Setup del Trade:")
    print(f"   Entry: ${current_price:.4f}")
    print(f"   Stop Loss: ${stop_loss:.4f}")
    print(f"   Take Profit: ${take_profit:.4f}")
    
    trade_validation = rm.validate_trade(current_price, take_profit, stop_loss, side)
    
    if not trade_validation['valid']:
        msg = f"❌ Trade rechazado: {trade_validation['reason']}"
        print(msg)
        send_telegram(msg)
        return
    
    print(f"✅ R/R Ratio: {trade_validation['rr_ratio']:.2f}")
    
    position = rm.calculate_position_size(
        current_price,
        stop_loss,
        signal['confidence'],
        side,
        use_leverage=True
    )
    
    if not position['valid']:
        msg = f"❌ Posición rechazada: {position['reason']}"
        print(msg)
        send_telegram(msg)
        return
    
    print(f"\n🔥 POSICIÓN CALCULADA:")
    print(f"   Volumen: {position['volume']} ADA")
    print(f"   Valor: ${position['position_value']:.2f}")
    print(f"   Leverage: {position['leverage']}x")
    print(f"   Margen requerido: ${position['margin_required']:.2f}")
    print(f"   Liquidación: ${position['liquidation_price']:.4f}")
    print(f"   Fees totales: ${position['total_fees_usd']:.2f}")
    
    print(f"\n🚀 EJECUTANDO ORDEN EN KRAKEN...")
    
    order_result = place_margin_order(
        side=side,
        volume=position['volume'],
        leverage=position['leverage']
    )
    
    if not order_result:
        msg = "❌ Error al ejecutar orden en Kraken"
        print(msg)
        send_telegram(msg)
        return
    
    save_order_to_tracking(order_result, signal, position)
    rm.reserve_margin(position['margin_required'])
    
    msg = f"""
🚀 *ORDEN EJECUTADA*

📊 *Setup:*
   • Señal: {signal['signal']}
   • Confianza: {signal['confidence']:.1f}%
   • Precio: ${current_price:.4f}

💼 *Posición:*
   • Volumen: {position['volume']} ADA
   • Valor: ${position['position_value']:.2f}
   • Leverage: {position['leverage']}x
   • Margen: ${position['margin_required']:.2f}

🎯 *Objetivos:*
   • TP: ${take_profit:.4f}
   • SL: ${stop_loss:.4f}
   • R/R: {trade_validation['rr_ratio']:.2f}
   • Liquidación: ${position['liquidation_price']:.4f}

💰 *Fees:*
   • Total: ${position['total_fees_usd']:.2f}
   • Ganancia mínima: ${position['min_profit_needed_usd']:.2f}

🆔 Order ID: `{order_result['order_id']}`
"""
    
    print(msg.replace('*', '').replace('`', ''))
    send_telegram(msg)
    
    print("\n" + "="*70)
    print("  ✅ ORDEN EJECUTADA CORRECTAMENTE")
    print("="*70)

def monitor_orders():
    """Monitorea y cierra órdenes abiertas"""
    print("\n🔍 Monitoreando órdenes abiertas...")
    
    if not os.path.exists(OPEN_ORDERS_FILE):
        print("ℹ️ No hay órdenes que monitorear")
        return
    
    with open(OPEN_ORDERS_FILE, 'r') as f:
        orders = json.load(f)
    
    if len(orders) == 0:
        print("ℹ️ No hay órdenes abiertas")
        return
    
    print(f"📋 Monitoreando {len(orders)} orden(es)...")
    
    current_price = get_current_price()
    
    if not current_price:
        print("❌ No se pudo obtener precio para monitorear")
        return
    
    for order_id, order_info in list(orders.items()):
        print(f"\n📊 Orden {order_id}:")
        print(f"   Lado: {order_info['side']}")
        print(f"   Entry: ${order_info['entry_price']:.4f}")
        print(f"   Current: ${current_price:.4f}")
        print(f"   TP: ${order_info['take_profit']:.4f}")
        print(f"   SL: ${order_info['stop_loss']:.4f}")
        
        if order_info['side'] == 'buy':
            pnl_pct = ((current_price - order_info['entry_price']) / order_info['entry_price']) * 100
            
            if current_price >= order_info['take_profit']:
                print("✅ TP alcanzado - Cerrando posición")
                close_reason = 'TP'
            elif current_price <= order_info['stop_loss']:
                print("🛑 SL alcanzado - Cerrando posición")
                close_reason = 'SL'
            else:
                print(f"💹 P&L actual: {pnl_pct:+.2f}%")
                continue
        else:
            pnl_pct = ((order_info['entry_price'] - current_price) / order_info['entry_price']) * 100
            
            if current_price <= order_info['take_profit']:
                print("✅ TP alcanzado - Cerrando posición")
                close_reason = 'TP'
            elif current_price >= order_info['stop_loss']:
                print("🛑 SL alcanzado - Cerrando posición")
                close_reason = 'SL'
            else:
                print(f"💹 P&L actual: {pnl_pct:+.2f}%")
                continue
        
        entry_time = datetime.fromisoformat(order_info['entry_time'])
        time_open = datetime.now() - entry_time
        
        if time_open > timedelta(hours=3.5):
            print("⏰ Timeout alcanzado (3.5h) - Cerrando para evitar rollover")
            close_reason = 'TIMEOUT'
        
        print(f"🔄 Cerrando por {close_reason}...")
        del orders[order_id]
    
    with open(OPEN_ORDERS_FILE, 'w') as f:
        json.dump(orders, f, indent=2)

if __name__ == "__main__":
    try:
        execute_trading_strategy()
        time.sleep(2)
        monitor_orders()
        
    except Exception as e:
        error_msg = f"❌ Error en trader: {str(e)}"
        print(error_msg)
        send_telegram(error_msg)
        raise
