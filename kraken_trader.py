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
CHAT_ID = os.environ.get('CHAT_ID', '')

# Archivos
TRADES_FILE = 'kraken_trades.csv'
OPEN_ORDERS_FILE = 'open_orders.json'
PREDICTION_TRACKER_FILE = 'prediction_tracker.csv'  # 🆕 Nuevo archivo

# 🔥 MODO DE OPERACIÓN
LIVE_TRADING = True  # ⚠️ Cambiar a True para trading real

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

def detect_ada_pair():
    """Detecta el par correcto de ADA en Kraken"""
    print("\n🔍 DETECTANDO PAR CORRECTO DE ADA...")
    
    possible_pairs = ['ADAUSD', 'XADAZUSD', 'ADAUSDT', 'ADAEUR', 'ADAGBP']
    
    try:
        url = f"{KRAKEN_API_URL}/0/public/AssetPairs"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'result' in data:
                available_pairs = data['result'].keys()
                ada_pairs = [p for p in available_pairs if 'ADA' in p.upper()]
                
                print(f"✅ Pares ADA disponibles: {ada_pairs}")
                
                for pair in possible_pairs:
                    if pair in ada_pairs:
                        print(f"✅ Par detectado: {pair}")
                        return pair
                
                if ada_pairs:
                    print(f"⚠️ Usando primer par disponible: {ada_pairs[0]}")
                    return ada_pairs[0]
        
        print("❌ No se pudo detectar par ADA")
        return None
        
    except Exception as e:
        print(f"❌ Error detectando par: {e}")
        return None

def get_current_price(retries=3, delay=2):
    """Obtiene precio actual de ADA"""
    pair = detect_ada_pair()
    
    if not pair:
        print("❌ No se pudo detectar par de trading")
        return None
    
    url = f"{KRAKEN_API_URL}/0/public/Ticker?pair={pair}"
    
    for attempt in range(retries):
        try:
            print(f"📊 Obteniendo precio de {pair} (intento {attempt + 1}/{retries})...")
            
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                print(f"⚠️ Status code: {response.status_code}")
                if attempt < retries - 1:
                    time.sleep(delay)
                    continue
                return None
            
            data = response.json()
            
            if 'error' in data and len(data['error']) > 0:
                print(f"❌ Error API: {data['error']}")
                if attempt < retries - 1:
                    time.sleep(delay)
                    continue
                return None
            
            if 'result' in data:
                result_pair = list(data['result'].keys())[0]
                price = float(data['result'][result_pair]['c'][0])
                print(f"✅ Precio obtenido: ${price:.4f} (par: {result_pair})")  # ✅ 4 decimales
                return price
            
            print(f"❌ No se encontró precio en la respuesta")
            if attempt < retries - 1:
                time.sleep(delay)
                continue
            
            return None
            
        except Exception as e:
            print(f"❌ Error: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
                continue
            return None
    
    return None

def get_balance():
    """Obtiene balance completo de Kraken"""
    data = {'nonce': str(int(1000*time.time()))}
    result = kraken_request('/0/private/Balance', data)
    return result

# 🔧 REEMPLAZO PARA kraken_trader.py
# Busca la función get_margin_balance() y reemplázala con esto:

def get_margin_balance():
    """
    ✅ VERSIÓN CORREGIDA: Obtiene balance de Derivatives Wallet
    Usa TradeBalance que detecta USD, EUR, etc. automáticamente
    """
    print("\n" + "="*70)
    print("  💰 OBTENIENDO BALANCE DE DERIVATIVES WALLET")
    print("="*70)
    
    # 🆕 Usar TradeBalance en lugar de Balance
    data = {'nonce': str(int(1000*time.time()))}
    result = kraken_request('/0/private/TradeBalance', data)
    
    if 'result' in result:
        # Extraer datos clave
        equity = float(result['result'].get('eb', 0))          # Balance total (equity)
        margin_used = float(result['result'].get('m', 0))      # Margen usado
        free_margin = float(result['result'].get('mf', 0))     # Margen libre (disponible)
        
        # Detectar moneda (Kraken devuelve en la moneda base de la cuenta)
        # Por defecto asume USD si tienes > 0
        currency = "USD" if equity > 0.1 else "EUR"
        
        print(f"\n📊 Detalles de la cuenta:")
        print(f"   💰 Equity Total: ${equity:.2f} {currency}")
        print(f"   📊 Margen Usado: ${margin_used:.2f} {currency}")
        print(f"   ✅ Margen Libre: ${free_margin:.2f} {currency}")
        
        # 🎯 Retornar margen libre (lo que podemos usar)
        if free_margin > 0:
            print(f"\n✅ Balance disponible para trading: ${free_margin:.2f} {currency}")
            return free_margin
        else:
            print(f"\n⚠️ NO HAY FONDOS DISPONIBLES")
            print(f"\n📋 SOLUCIÓN:")
            print(f"   1. Ve a Kraken.com → Funding → Transfer")
            print(f"   2. Transfiere de Spot Wallet → Derivatives Wallet")
            print(f"   3. Mínimo: 10 USD/EUR para trading con leverage")
            return 0
    
    print("\n❌ Error obteniendo balance de TradeBalance")
    
    # Fallback: intentar con Balance normal
    print("\n🔄 Intentando con Balance endpoint...")
    data = {'nonce': str(int(1000*time.time()))}
    balance = kraken_request('/0/private/Balance', data)
    
    if 'result' in balance:
        # Buscar cualquier símbolo USD o EUR
        usd_symbols = ['ZUSD', 'USD', 'USDT', 'USDC']
        eur_symbols = ['ZEUR', 'EUR']
        
        total = 0
        
        print("\n📊 Balances detectados:")
        for asset, amount in balance['result'].items():
            amount_float = float(amount)
            if amount_float > 0:
                print(f"   {asset}: {amount_float:.2f}")
                
                if asset in usd_symbols or asset in eur_symbols:
                    total += amount_float
        
        if total > 0:
            print(f"\n✅ Balance total: ${total:.2f}")
            return total
        else:
            print("\n⚠️ No se encontraron fondos")
            return 0
    
    print("❌ Error obteniendo balance")
    return 0

def place_order(side, volume, price, tp_price, sl_price):
    """Coloca orden con par correcto detectado automáticamente"""
    pair = detect_ada_pair()
    
    if not pair:
        return {'error': ['No se pudo detectar par de trading']}
    
    data = {
        'nonce': str(int(1000*time.time())),
        'ordertype': 'limit' if price else 'market',
        'type': side,
        'volume': str(volume),
        'pair': pair,
        'leverage': '10'
    }
    
    if price:
        data['price'] = str(price)
    
    print(f"📤 Enviando orden a Kraken:")
    print(f"   Par: {pair}")
    print(f"   Tipo: {side}")
    print(f"   Volumen: {volume}")
    print(f"   Leverage: 10x")
    
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
    """Calcula TP al 80% de la predicción y SL con ATR"""
    if side == 'buy':
        target_move = pred_high - entry_price
        tp = entry_price + (target_move * tp_percentage)
        sl = entry_price - (atr * 2)
    else:
        target_move = entry_price - pred_low
        tp = entry_price - (target_move * tp_percentage)
        sl = entry_price + (atr * 2)
    
    return round(tp, 4), round(sl, 4)  # ✅ 4 decimales


def update_prediction_tracker_on_order_open(timestamp, order_id, entry_price):
    """
    🆕 ACTUALIZA prediction_tracker.csv cuando se abre una orden
    """
    if not os.path.exists(PREDICTION_TRACKER_FILE):
        print(f"⚠️ {PREDICTION_TRACKER_FILE} no existe aún")
        return
    
    try:
        df = pd.read_csv(PREDICTION_TRACKER_FILE)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Buscar la última predicción (la más reciente)
        latest_idx = df.index[-1]
        
        # Actualizar con datos de la orden
        df.loc[latest_idx, 'order_opened'] = 'YES'
        df.loc[latest_idx, 'order_id'] = order_id
        df.loc[latest_idx, 'entry_price'] = round(entry_price, 4)  # ✅ 4 decimales
        
        # Guardar
        df.to_csv(PREDICTION_TRACKER_FILE, index=False)
        print(f"✅ {PREDICTION_TRACKER_FILE} actualizado: orden abierta")
        
    except Exception as e:
        print(f"❌ Error actualizando tracker: {e}")


def update_prediction_tracker_on_order_close(order_id, exit_price, pnl_usd, pnl_pct, 
                                              close_reason, actual_high, actual_low, actual_close):
    """
    🆕 ACTUALIZA prediction_tracker.csv cuando se cierra una orden
    Calcula precisión de la predicción
    """
    if not os.path.exists(PREDICTION_TRACKER_FILE):
        print(f"⚠️ {PREDICTION_TRACKER_FILE} no existe")
        return
    
    try:
        df = pd.read_csv(PREDICTION_TRACKER_FILE)
        
        # Buscar la fila con este order_id
        mask = df['order_id'] == order_id
        
        if not mask.any():
            print(f"⚠️ Order {order_id} no encontrada en tracker")
            return
        
        idx = df[mask].index[0]
        
        # Actualizar datos de cierre
        df.loc[idx, 'exit_price'] = round(exit_price, 4)      # ✅ 4 decimales
        df.loc[idx, 'pnl_usd'] = round(pnl_usd, 2)
        df.loc[idx, 'pnl_%'] = round(pnl_pct, 2)
        df.loc[idx, 'close_reason'] = close_reason
        df.loc[idx, 'actual_high'] = round(actual_high, 4)    # ✅ 4 decimales
        df.loc[idx, 'actual_low'] = round(actual_low, 4)      # ✅ 4 decimales
        df.loc[idx, 'actual_close'] = round(actual_close, 4)  # ✅ 4 decimales
        
        # Calcular precisión de predicción
        pred_high = df.loc[idx, 'pred_high']
        pred_low = df.loc[idx, 'pred_low']
        pred_close = df.loc[idx, 'pred_close']
        
        # Precisión = qué tan cerca estuvo la predicción
        high_error = abs(pred_high - actual_high) / actual_high * 100
        low_error = abs(pred_low - actual_low) / actual_low * 100
        close_error = abs(pred_close - actual_close) / actual_close * 100
        
        avg_error = (high_error + low_error + close_error) / 3
        accuracy = max(0, 100 - avg_error)
        
        df.loc[idx, 'pred_accuracy_%'] = round(accuracy, 2)
        
        # Guardar
        df.to_csv(PREDICTION_TRACKER_FILE, index=False)
        print(f"✅ {PREDICTION_TRACKER_FILE} actualizado: orden cerrada")
        print(f"   Precisión predicción: {accuracy:.2f}%")
        
    except Exception as e:
        print(f"❌ Error actualizando tracker: {e}")

"""
🔧 FUNCIÓN CORREGIDA: monitor_orders()
Cierra posiciones abiertas correctamente con órdenes contrarias
"""

def close_position_in_kraken(txid, side, volume):
    """
    ✅ NUEVO: Cierra una posición abierta con una orden contraria
    """
    # Determinar el lado contrario
    close_side = 'sell' if side == 'buy' else 'buy'
    
    print(f"🔄 Cerrando posición {txid[:8]}...")
    print(f"   Original: {side.upper()} {volume} ADA")
    print(f"   Cierre: {close_side.upper()} {volume} ADA")
    
    # Colocar orden de mercado contraria para cerrar inmediatamente
    data = {
        'nonce': str(int(1000*time.time())),
        'ordertype': 'market',  # Market order para cierre inmediato
        'type': close_side,
        'volume': str(volume),
        'pair': detect_ada_pair() or 'ADAUSD',
        'leverage': '10',
        'reduce_only': True  # ⚠️ IMPORTANTE: reduce_only=True cierra la posición
    }
    
    result = kraken_request('/0/private/AddOrder', data)
    return result


def verify_position_in_kraken(txid):
    """
    ✅ NUEVO: Verifica si una posición realmente existe en Kraken
    """
    try:
        data = {'nonce': str(int(1000*time.time()))}
        result = kraken_request('/0/private/OpenPositions', data)
        
        if 'result' in result:
            # Buscar si txid existe en posiciones abiertas
            for pos_id, pos_data in result['result'].items():
                if pos_id == txid or pos_data.get('ordertxid') == txid:
                    return True, pos_data
        
        return False, None
        
    except Exception as e:
        print(f"⚠️ Error verificando posición: {e}")
        return False, None


def monitor_orders():
    """
    ✅ VERSIÓN CORREGIDA: Monitorea y cierra posiciones correctamente
    """
    if not os.path.exists(OPEN_ORDERS_FILE):
        print("ℹ️ No hay archivo de órdenes abiertas")
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
        margin_reserved = order.get('margin_required', 0)
        
        time_open = (datetime.now() - open_time).total_seconds() / 60
        
        # 🆕 VERIFICAR SI LA POSICIÓN AÚN EXISTE EN KRAKEN
        if LIVE_TRADING:
            exists, position_data = verify_position_in_kraken(txid)
            
            if not exists:
                print(f"⚠️ Posición {txid[:8]} NO existe en Kraken (ya cerrada manualmente?)")
                print(f"   Eliminando del tracking local...")
                continue  # No la agregamos a updated_orders
        
        should_close = False
        close_reason = None
        close_price = current_price
        
        # 1. Verificar TP
        if side == 'buy' and current_price >= tp:
            should_close = True
            close_reason = 'TP'
        elif side == 'sell' and current_price <= tp:
            should_close = True
            close_reason = 'TP'
        
        # 2. Verificar SL
        elif side == 'buy' and current_price <= sl:
            should_close = True
            close_reason = 'SL'
        elif side == 'sell' and current_price >= sl:
            should_close = True
            close_reason = 'SL'
        
        # 3. TIMEOUT - 5 horas
        elif time_open >= 300:
            should_close = True
            close_reason = 'TIMEOUT'
        
        # 4. STOP LOSS PROGRESIVO (primeros 10 min)
        elif time_open <= 10:
            loss_pct = ((current_price - entry_price) / entry_price) * 100
            if side == 'buy' and loss_pct < -1.0:
                should_close = True
                close_reason = 'QUICK_LOSS'
            elif side == 'sell' and loss_pct > 1.0:
                should_close = True
                close_reason = 'QUICK_LOSS'
        
        if should_close:
            print(f"🔴 Cerrando orden {txid[:8]}... por {close_reason}")
            print(f"   Tiempo abierto: {time_open:.1f} min")
            print(f"   Precio entrada: ${entry_price:.4f}")
            print(f"   Precio cierre: ${close_price:.4f}")
            
            # 🔥 CERRAR EN KRAKEN SI LIVE_TRADING
            if LIVE_TRADING:
                # ✅ USAR close_position CORRECTAMENTE
                close_result = close_position_in_kraken(txid, side, volume)
                
                if 'result' in close_result and 'txid' in close_result['result']:
                    print(f"   ✅ Posición cerrada en Kraken: {close_result['result']['txid']}")
                elif 'error' in close_result:
                    print(f"   ⚠️ Error cerrando en Kraken: {close_result['error']}")
                    print(f"   ℹ️ Intentando cancelar como orden pendiente...")
                    
                    # Fallback: intentar cancelar como orden pendiente
                    cancel_result = cancel_order(txid)
                    print(f"   Cancel result: {cancel_result}")
            else:
                print("   ⚠️ MODO SIMULACIÓN - Orden NO cerrada en Kraken")
            
            # Calcular P&L
            if side == 'buy':
                pnl = (close_price - entry_price) * volume
                pnl_pct = ((close_price - entry_price) / entry_price) * 100
            else:
                pnl = (entry_price - close_price) * volume
                pnl_pct = ((entry_price - close_price) / entry_price) * 100
            
            # Actualizar prediction tracker
            actual_high = close_price * 1.001
            actual_low = close_price * 0.999
            actual_close = close_price
            
            update_prediction_tracker_on_order_close(
                txid, close_price, pnl, pnl_pct, close_reason,
                actual_high, actual_low, actual_close
            )
            
            # Actualizar capital y liberar margen
            risk_manager.update_after_trade(pnl, margin_released=margin_reserved)
            
            # Guardar en CSV
            trade_data = {
                'timestamp': datetime.now(),
                'txid': txid,
                'side': side,
                'entry_price': round(entry_price, 4),
                'close_price': round(close_price, 4),
                'volume': volume,
                'tp': round(tp, 4),
                'sl': round(sl, 4),
                'close_reason': close_reason,
                'time_open_min': round(time_open, 1),
                'pnl_usd': round(pnl, 2),
                'pnl_%': round(pnl_pct, 2)
            }
            
            df = pd.DataFrame([trade_data])
            if os.path.exists(TRADES_FILE):
                df.to_csv(TRADES_FILE, mode='a', header=False, index=False)
            else:
                df.to_csv(TRADES_FILE, index=False)
            
            # Telegram
            emoji = "✅" if pnl > 0 else "❌"
            stats = risk_manager.get_stats()
            
            mode = "🔥 LIVE" if LIVE_TRADING else "💼 SIMULACIÓN"
            
            msg = f"""
{emoji} *Orden Cerrada* {mode}

📖 ID: {txid[:8]}...
📊 Tipo: {side.upper()}
💰 Entrada: ${entry_price:.4f}
💰 Salida: ${close_price:.4f}
🎯 Razón: {close_reason}
⏱️ Tiempo: {time_open:.1f} min

💵 P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)
🔓 Margen Liberado: ${margin_reserved:.2f}

📈 *Capital:*
   Actual: ${stats['current_capital']:.2f}
   Total: ${stats['total_profit']:+.2f} ({stats['profit_%']:+.2f}%)
   WR: {stats['win_rate']:.1f}% ({stats['win_count']}/{stats['total_trades']})
"""
            send_telegram(msg)
        else:
            updated_orders.append(order)
            time_left = 300 - time_open
            print(f"📊 {txid[:8]}... | {side.upper()} | {time_open:.1f}min | Quedan {time_left:.1f}min")
    
    # Guardar órdenes actualizadas
    with open(OPEN_ORDERS_FILE, 'w') as f:
        json.dump(updated_orders, f, indent=2)
    
    if len(updated_orders) > 0:
        print(f"✅ Monitoreo completado: {len(updated_orders)} órdenes activas")
    else:
        print("✅ Todas las órdenes fueron cerradas")


# 🆕 FUNCIÓN DE VERIFICACIÓN MANUAL
def check_kraken_positions():
    """
    Verifica posiciones reales en Kraken y sincroniza con open_orders.json
    """
    print("\n" + "="*70)
    print("  🔍 VERIFICANDO POSICIONES EN KRAKEN")
    print("="*70)
    
    try:
        data = {'nonce': str(int(1000*time.time()))}
        result = kraken_request('/0/private/OpenPositions', data)
        
        if 'error' in result and len(result['error']) > 0:
            print(f"❌ Error API: {result['error']}")
            return
        
        if 'result' in result:
            positions = result['result']
            
            if len(positions) == 0:
                print("✅ NO hay posiciones abiertas en Kraken")
            else:
                print(f"📊 Posiciones abiertas en Kraken: {len(positions)}\n")
                
                for pos_id, pos_data in positions.items():
                    pair = pos_data.get('pair', 'Unknown')
                    side = pos_data.get('type', 'Unknown')
                    volume = float(pos_data.get('vol', 0))
                    cost = float(pos_data.get('cost', 0))
                    margin = float(pos_data.get('margin', 0))
                    pnl = float(pos_data.get('net', 0))
                    
                    print(f"🔸 ID: {pos_id}")
                    print(f"   Par: {pair}")
                    print(f"   Tipo: {side.upper()}")
                    print(f"   Volumen: {volume}")
                    print(f"   Costo: ${cost:.2f}")
                    print(f"   Margen: ${margin:.2f}")
                    print(f"   P&L: ${pnl:+.2f}")
                    print()
        
        # Comparar con archivo local
        if os.path.exists(OPEN_ORDERS_FILE):
            with open(OPEN_ORDERS_FILE, 'r') as f:
                local_orders = json.load(f)
            
            print(f"📁 Órdenes en open_orders.json: {len(local_orders)}")
            
            if len(local_orders) != len(positions):
                print(f"⚠️ DESINCRONIZACIÓN DETECTADA:")
                print(f"   Kraken: {len(positions)} posiciones")
                print(f"   Local: {len(local_orders)} órdenes")
                print(f"\n💡 Ejecuta monitor_orders() para sincronizar")
        
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def execute_signal():
    """Lee señal Y sincroniza con balance REAL de Kraken"""
    
    signals_file = 'trading_signals.csv'
    if not os.path.exists(signals_file):
        print("❌ No hay señales disponibles")
        return
    
    df = pd.read_csv(signals_file)
    latest = df.iloc[-1]
    
    signal = latest['signal']
    
    if signal == 'HOLD':
        print("⸻ Señal HOLD - No hay acción")
        return
    
    # ✅ PASO 1: Obtener Risk Manager
    risk_manager = get_risk_manager()
    
    # ✅ PASO 2: SINCRONIZAR CON BALANCE REAL DE KRAKEN
    print("\n" + "="*70)
    print("  🔄 SINCRONIZANDO CON KRAKEN")
    print("="*70)
    
    if LIVE_TRADING:
        kraken_balance = get_margin_balance()
        
        if kraken_balance <= 0:
            error_msg = """
❌ *ERROR: Sin fondos en Margin Wallet*

Para usar leverage 10x necesitas:
1️⃣ Transferir fondos a Margin Wallet
2️⃣ Ve a Kraken.com → Funding → Transfer
3️⃣ De Spot Wallet → Margin Wallet
4️⃣ Mínimo: 10 EUR/USD

📋 Sin fondos en Margin = Sin trading con leverage
"""
            print(error_msg)
            send_telegram(error_msg)
            return
        
        risk_manager.sync_with_kraken_balance(kraken_balance)
        print(f"✅ Balance sincronizado: ${kraken_balance:.2f}")
    else:
        print("⚠️ MODO SIMULACIÓN - Usando capital simulado")
    
    risk_manager.print_stats()
    
    # ✅ VERIFICAR SOLO 1 ORDEN A LA VEZ
    if os.path.exists(OPEN_ORDERS_FILE):
        with open(OPEN_ORDERS_FILE, 'r') as f:
            open_orders = json.load(f)
        if len(open_orders) >= 1:
            print(f"⚠️ Ya hay {len(open_orders)} orden(es) abierta(s). Solo se permite 1 a la vez.")
            return
    
    current_price = get_current_price()
    if not current_price:
        print("❌ No se pudo obtener precio actual")
        return
    
    atr = latest['atr']
    pred_high = latest['pred_high']
    pred_low = latest['pred_low']
    confidence = latest['confidence']
    
    side = signal.lower()
    tp, sl = calculate_tp_sl(current_price, side, atr, pred_high, pred_low, tp_percentage=0.80)
    
    print(f"\n{'='*70}")
    print(f"  🔍 VALIDANDO TRADE")
    print(f"{'='*70}")
    
    # Validar R/R
    trade_validation = risk_manager.validate_trade(current_price, tp, sl, side)
    
    if not trade_validation['valid']:
        print(f"❌ Trade rechazado: {trade_validation['reason']}")
        msg = f"⛔ *Trade Rechazado*\n\n📊 {signal}\n❌ {trade_validation['reason']}"
        send_telegram(msg)
        return
    
    print(f"✅ R/R Ratio: {trade_validation['rr_ratio']:.2f}")
    print(f"   Risk: ${trade_validation['risk']:.4f}")    # ✅ 4 decimales
    print(f"   Reward: ${trade_validation['reward']:.4f}")  # ✅ 4 decimales
    
    # Calcular posición con leverage 10x
    position = risk_manager.calculate_position_size(current_price, sl, confidence, side, use_leverage=True)
    
    if not position['valid']:
        print(f"❌ Posición rechazada: {position['reason']}")
        msg = f"⛔ *Posición Rechazada*\n\n📊 {signal}\n❌ {position['reason']}"
        send_telegram(msg)
        return
    
    volume = position['volume']
    
    print(f"\n{'='*70}")
    print(f"🚀 EJECUTANDO ORDEN CON LEVERAGE 10X")
    print(f"{'='*70}")
    print(f"📊 Señal: {signal}")
    print(f"💰 Precio: ${current_price:.4f}")  # ✅ 4 decimales
    print(f"📈 Volumen: {volume} ADA (${position['position_value']:.2f})")
    print(f"   • Leverage: {position['leverage']}x")
    print(f"   • Riesgo: ${position['risk_amount']:.2f}")
    print(f"   • Margen Req: ${position['margin_required']:.2f}")
    print(f"   • Capital usado: {position['capital_used_%']:.1f}%")
    print(f"🎯 TP: ${tp:.4f} ({((tp-current_price)/current_price*100):+.2f}%)")  # ✅ 4 decimales
    print(f"🛑 SL: ${sl:.4f} ({((sl-current_price)/current_price*100):+.2f}%)")  # ✅ 4 decimales
    print(f"⚠️ Liquidación: ${position['liquidation_price']:.4f}")  # ✅ 4 decimales
    print(f"📊 R/R: {trade_validation['rr_ratio']:.2f}")
    print(f"🎲 Confianza: {confidence:.1f}%")
    print(f"{'='*70}\n")
    
    # 🔥 EJECUCIÓN REAL O SIMULADA
    if LIVE_TRADING:
        print("🔥 MODO LIVE - Enviando orden a Kraken...")
        result = place_order(side, volume, None, tp, sl)
        
        if 'result' in result and 'txid' in result['result']:
            txid = result['result']['txid'][0]
            print(f"✅ Orden ejecutada en Kraken: {txid}")
            
            # 🆕 ACTUALIZAR PREDICTION TRACKER
            timestamp = latest['timestamp']
            update_prediction_tracker_on_order_open(timestamp, txid, current_price)
            
            # Reservar margen
            risk_manager.reserve_margin(position['margin_required'])
            
            # Guardar orden abierta
            order_data = {
                'txid': txid,
                'side': side,
                'entry_price': round(current_price, 4),  # ✅ 4 decimales
                'volume': volume,
                'tp': round(tp, 4),                      # ✅ 4 decimales
                'sl': round(sl, 4),                      # ✅ 4 decimales
                'open_time': datetime.now().isoformat(),
                'signal_confidence': confidence,
                'rr_ratio': trade_validation['rr_ratio'],
                'risk_amount': position['risk_amount'],
                'margin_required': position['margin_required'],
                'leverage': position['leverage'],
                'liquidation_price': round(position['liquidation_price'], 4)  # ✅ 4 decimales
            }
            
            orders = []
            if os.path.exists(OPEN_ORDERS_FILE):
                with open(OPEN_ORDERS_FILE, 'r') as f:
                    orders = json.load(f)
            
            orders.append(order_data)
            with open(OPEN_ORDERS_FILE, 'w') as f:
                json.dump(orders, f, indent=2)
            
            # CSV de ejecución
            trade_data = {
                'timestamp': datetime.now(),
                'txid': txid,
                'side': side,
                'entry_price': round(current_price, 4),  # ✅ 4 decimales
                'volume': volume,
                'tp': round(tp, 4),                      # ✅ 4 decimales
                'sl': round(sl, 4),                      # ✅ 4 decimales
                'confidence': confidence,
                'rr_ratio': trade_validation['rr_ratio'],
                'risk_amount': position['risk_amount'],
                'leverage': position['leverage'],
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
🔥 *LIVE TRADING - Nueva Orden*

📊 Tipo: {signal}
💰 Entrada: ${current_price:.4f}
📈 Volumen: {volume} ADA
⚡ Leverage: {position['leverage']}x
   • Valor: ${position['position_value']:.2f}
   • Margen: ${position['margin_required']:.2f}
   • Riesgo: ${position['risk_amount']:.2f}

🎯 TP: ${tp:.4f} ({((tp-current_price)/current_price*100):+.2f}%)
🛑 SL: ${sl:.4f} ({((sl-current_price)/current_price*100):+.2f}%)
⚠️ Liquidación: ${position['liquidation_price']:.4f}
📊 R/R: {trade_validation['rr_ratio']:.2f}
🎲 Confianza: {confidence:.1f}%

📈 *Estado Cuenta:*
   Capital: ${stats['current_capital']:.2f}
   Margen Usado: ${stats['margin_used']:.2f}
   Posiciones: {stats['open_positions']}/1
"""
            send_telegram(msg)
        else:
            error = result.get('error', 'Unknown error')
            print(f"❌ Error al ejecutar orden: {error}")
            send_telegram(f"❌ Error ejecutando orden: {error}")
    
    else:
        print("💼 MODO SIMULACIÓN - Orden NO enviada a Kraken")
        print("   ⚠️ Para activar trading real, cambiar LIVE_TRADING = True")


def main():
    mode = "🔥 LIVE TRADING" if LIVE_TRADING else "💼 SIMULACIÓN"
    
    print("="*70)
    print(f"  🤖 KRAKEN TRADER BOT - {mode}")
    print("="*70)
    
    # 1. Monitorear órdenes
    print("\n🔍 Monitoreando órdenes abiertas...")
    monitor_orders()
    
    # 2. Verificar señal
    print("\n📊 Verificando nuevas señales...")
    execute_signal()
    
    # 3. Resumen
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
