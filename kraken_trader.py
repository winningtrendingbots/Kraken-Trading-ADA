"""
KRAKEN TRADER - VERSIÓN HÍBRIDA CON SINCRONIZACIÓN

✅ Ordenes con STOP-LOSS automático
✅ Monitoreo manual para TAKE-PROFIT
✅ Protección total: si el bot falla, el SL te salva
🆕 Sincronización con estado real de Kraken
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
    """Obtiene balance de MARGIN"""
    print("\n💰 Obteniendo balance de Kraken (MARGIN)...")
    
    result = kraken_request('/0/private/TradeBalance', {'asset': 'ZUSD'})
    
    if result:
        margin_balance = float(result.get('eb', 0))
        free_margin = float(result.get('mf', 0))
        used_margin = float(result.get('m', 0))
        
        print(f"📊 Balance de Trading:")
        print(f"   💵 Total disponible: ${margin_balance:.2f}")
        print(f"   ✅ Margen libre: ${free_margin:.2f}")
        print(f"   🔒 Margen usado: ${used_margin:.2f}")
        
        if free_margin < 5:
            print(f"⚠️ Margen libre insuficiente para operar (mínimo $5)")
        
        return free_margin
    
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
    print(f"\n🔎 Buscando señales en {SIGNALS_FILE}...")
    
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

def sync_open_orders_with_kraken():
    """
    🆕 Sincroniza open_orders.json con el estado REAL de Kraken
    Elimina órdenes que ya no existen en Kraken
    """
    print("\n🔄 Sincronizando con Kraken...")
    
    if not os.path.exists(OPEN_ORDERS_FILE):
        print("✅ No hay órdenes locales")
        return {}
    
    # Cargar órdenes locales
    with open(OPEN_ORDERS_FILE, 'r') as f:
        local_orders = json.load(f)
    
    if len(local_orders) == 0:
        print("✅ No hay órdenes locales")
        return {}
    
    # Consultar órdenes REALES en Kraken
    print(f"📋 Verificando {len(local_orders)} orden(es) local(es)...")
    
    result = kraken_request('/0/private/OpenOrders', {})
    
    if not result:
        print("⚠️ No se pudo consultar Kraken, manteniendo estado local")
        return local_orders
    
    kraken_open_orders = result.get('open', {})
    
    # Filtrar órdenes que YA NO EXISTEN en Kraken
    orders_to_remove = []
    
    for order_id in local_orders.keys():
        if order_id not in kraken_open_orders:
            print(f"🗑️ Orden {order_id} ya no existe en Kraken (cerrada manualmente)")
            orders_to_remove.append(order_id)
    
    # Eliminar órdenes cerradas
    for order_id in orders_to_remove:
        del local_orders[order_id]
    
    # Guardar estado actualizado
    with open(OPEN_ORDERS_FILE, 'w') as f:
        json.dump(local_orders, f, indent=2)
    
    if len(orders_to_remove) > 0:
        print(f"✅ {len(orders_to_remove)} orden(es) eliminada(s)")
    
    if len(local_orders) > 0:
        print(f"📊 {len(local_orders)} orden(es) realmente abierta(s)")
    else:
        print("✅ No hay órdenes abiertas")
    
    return local_orders

def check_existing_orders():
    """
    Verifica si ya hay órdenes abiertas
    🔥 AHORA sincroniza con Kraken primero
    """
    # 🆕 Sincronizar con Kraken
    orders = sync_open_orders_with_kraken()
    
    if len(orders) > 0:
        print(f"⚠️ Ya hay {len(orders)} orden(es) abierta(s)")
        return True
    
    return False

def place_margin_order_with_sl(side, volume, leverage, entry_price, stop_loss):
    """
    🔥 NUEVO: Coloca orden con STOP-LOSS AUTOMÁTICO
    
    Args:
        side: 'buy' o 'sell'
        volume: Cantidad de ADA
        leverage: Multiplicador (2-5)
        entry_price: Precio actual (para referencia)
        stop_loss: Precio del stop loss
    
    Returns:
        dict con order_id y detalles
    """
    print(f"\n📤 Colocando orden MARGIN {side.upper()} con SL automático...")
    print(f"   Volumen: {volume} ADA")
    print(f"   Leverage: {leverage}x")
    print(f"   Entry: ${entry_price:.4f}")
    print(f"   Stop Loss: ${stop_loss:.4f}")
    
    # Calcular precio límite del SL (5% peor que el trigger)
    if side == 'buy':
        sl_limit_price = stop_loss * 0.995  # 0.5% peor
    else:
        sl_limit_price = stop_loss * 1.005
    
    # Orden principal
    order_data = {
        'pair': PAIR,
        'type': side,
        'ordertype': 'market',
        'volume': str(volume),
        'leverage': str(leverage),
        'close': json.dumps({
            'ordertype': 'stop-loss-limit',
            'price': str(stop_loss),
            'price2': str(sl_limit_price)
        })
    }
    
    result = kraken_request('/0/private/AddOrder', order_data)
    
    if not result:
        print("❌ Error al colocar orden")
        return None
    
    order_id = result['txid'][0]
    
    print(f"✅ Orden colocada: {order_id}")
    print(f"🛡️ Stop-Loss automático configurado en ${stop_loss:.4f}")
    
    return {
        'order_id': order_id,
        'side': side,
        'volume': volume,
        'leverage': leverage,
        'has_auto_sl': True,
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
        'expected_risk': position_info['risk_amount'],
        'has_auto_sl': order_info.get('has_auto_sl', False)
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
        'liquidation_price': position_info['liquidation_price'],
        'has_auto_sl': order_info.get('has_auto_sl', False)
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

def close_position(order_id, side, volume):
    """
    Cierra una posición manualmente (para TP o timeout)
    """
    print(f"\n🔄 Cerrando posición {order_id}...")
    
    # Para cerrar una posición long, hacemos sell (y viceversa)
    close_side = 'sell' if side == 'buy' else 'buy'
    
    close_data = {
        'pair': PAIR,
        'type': close_side,
        'ordertype': 'market',
        'volume': str(volume),
        'leverage': '0'  # Sin leverage para cerrar
    }
    
    result = kraken_request('/0/private/AddOrder', close_data)
    
    if result:
        print(f"✅ Posición cerrada: {result['txid'][0]}")
        return True
    else:
        print(f"❌ Error al cerrar posición")
        return False

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
    
    # 🔥 NUEVO: TP DINÁMICO basado en predicciones del modelo
    if side == 'buy':
        stop_loss = current_price * 0.98  # SL fijo al -2%
        
        # Usar pred_high del modelo
        pred_high = signal.get('pred_high', current_price * 1.03)
        distance_to_high = pred_high - current_price
        
        # TP al 70% de la distancia predicha (conservador)
        take_profit = current_price + (distance_to_high * 0.70)
        
        print(f"\n📊 Análisis BUY:")
        print(f"   Precio actual: ${current_price:.4f}")
        print(f"   Pred High: ${pred_high:.4f} (+{((pred_high - current_price) / current_price * 100):.2f}%)")
        print(f"   Distancia a High: ${distance_to_high:.4f}")
        print(f"   TP (70% dist): ${take_profit:.4f} (+{((take_profit - current_price) / current_price * 100):.2f}%)")
        
    else:  # SELL
        stop_loss = current_price * 1.02  # SL fijo al +2%
        
        # Usar pred_low del modelo
        pred_low = signal.get('pred_low', current_price * 0.97)
        distance_to_low = current_price - pred_low
        
        # TP al 70% de la distancia predicha
        take_profit = current_price - (distance_to_low * 0.70)
        
        print(f"\n📊 Análisis SELL:")
        print(f"   Precio actual: ${current_price:.4f}")
        print(f"   Pred Low: ${pred_low:.4f} ({((pred_low - current_price) / current_price * 100):.2f}%)")
        print(f"   Distancia a Low: ${distance_to_low:.4f}")
        print(f"   TP (70% dist): ${take_profit:.4f} ({((take_profit - current_price) / current_price * 100):.2f}%)")
    
    print(f"\n📊 Setup del Trade:")
    print(f"   Entry: ${current_price:.4f}")
    print(f"   Stop Loss: ${stop_loss:.4f} (automático)")
    print(f"   Take Profit: ${take_profit:.4f} (monitoreo manual)")
    
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
    
    print(f"\n🚀 EJECUTANDO ORDEN EN KRAKEN CON SL AUTOMÁTICO...")
    
    # 🔥 USAR LA NUEVA FUNCIÓN CON SL
    order_result = place_margin_order_with_sl(
        side=side,
        volume=position['volume'],
        leverage=position['leverage'],
        entry_price=current_price,
        stop_loss=stop_loss
    )
    
    if not order_result:
        msg = "❌ Error al ejecutar orden en Kraken"
        print(msg)
        send_telegram(msg)
        return
    
    save_order_to_tracking(order_result, signal, position)
    rm.reserve_margin(position['margin_required'])
    
    # 🔥 MENSAJE MEJORADO con predicciones del modelo
    pred_info = ""
    if side == 'buy':
        pred_high = signal.get('pred_high', 0)
        if pred_high > 0:
            pred_info = f"\n📈 *Predicción Modelo:*\n   High: ${pred_high:.4f} (+{((pred_high - current_price) / current_price * 100):.2f}%)"
    else:
        pred_low = signal.get('pred_low', 0)
        if pred_low > 0:
            pred_info = f"\n📉 *Predicción Modelo:*\n   Low: ${pred_low:.4f} ({((pred_low - current_price) / current_price * 100):.2f}%)"
    
    msg = f"""
🚀 *ORDEN EJECUTADA*

📊 *Setup:*
   • Señal: {signal['signal']}
   • Confianza: {signal['confidence']:.1f}%
   • Precio: ${current_price:.4f}
{pred_info}

💼 *Posición:*
   • Volumen: {position['volume']} ADA
   • Valor: ${position['position_value']:.2f}
   • Leverage: {position['leverage']}x
   • Margen: ${position['margin_required']:.2f}

🎯 *Objetivos:*
   • TP: ${take_profit:.4f} (70% de pred - monitoreo manual)
   • SL: ${stop_loss:.4f} 🛡️ *AUTOMÁTICO*
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
    """
    Monitorea órdenes abiertas - Solo revisa TAKE PROFIT
    (El stop-loss es automático en Kraken)
    🔥 AHORA sincroniza con Kraken al inicio
    """
    print("\n🔍 Monitoreando órdenes abiertas (solo TP)...")
    
    # 🆕 Sincronizar con Kraken PRIMERO
    orders = sync_open_orders_with_kraken()
    
    if len(orders) == 0:
        print("ℹ️ No hay órdenes que monitorear")
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
        print(f"   SL: ${order_info['stop_loss']:.4f} {'🛡️ (auto)' if order_info.get('has_auto_sl') else ''}")
        
        close_reason = None
        
        # Solo revisar TP y TIMEOUT (el SL es automático)
        if order_info['side'] == 'buy':
            pnl_pct = ((current_price - order_info['entry_price']) / order_info['entry_price']) * 100
            
            if current_price >= order_info['take_profit']:
                print("✅ TP alcanzado - Cerrando posición")
                close_reason = 'TP'
            else:
                print(f"💹 P&L actual: {pnl_pct:+.2f}%")
        else:
            pnl_pct = ((order_info['entry_price'] - current_price) / order_info['entry_price']) * 100
            
            if current_price <= order_info['take_profit']:
                print("✅ TP alcanzado - Cerrando posición")
                close_reason = 'TP'
            else:
                print(f"💹 P&L actual: {pnl_pct:+.2f}%")
        
        # Verificar timeout (3.5 horas)
        entry_time = datetime.fromisoformat(order_info['entry_time'])
        time_open = datetime.now() - entry_time
        
        if time_open > timedelta(hours=3.5):
            print("⏰ Timeout alcanzado (3.5h) - Cerrando para evitar rollover")
            close_reason = 'TIMEOUT'
        
        # Cerrar si hay razón
        if close_reason:
            success = close_position(
                order_id,
                order_info['side'],
                order_info['volume']
            )
            
            if success:
                # Remover de open_orders
                del orders[order_id]
                
                msg = f"🔒 *Posición Cerrada*\n\n"
                msg += f"Razón: {close_reason}\n"
                msg += f"P&L: {pnl_pct:+.2f}%\n"
                msg += f"Tiempo abierto: {time_open}"
                
                send_telegram(msg)
    
    # Guardar órdenes actualizadas
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
