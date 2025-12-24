"""
KRAKEN TRADER - VERSIÓN CON VALIDACIÓN DE COHERENCIA

✅ Valida que pred_close esté entre pred_high y pred_low
✅ Detecta desincronización entre precio base y precio actual
✅ Rechaza trades si el precio actual está fuera del rango predicho
✅ TP/SL ajustados correctamente desde el precio base
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

# 🆕 Configuración de tolerancia
MAX_PRICE_DRIFT_PCT = 3.0  # Máximo 3% de diferencia entre precio base y actual
PREDICTION_MAX_AGE_MINUTES = 10  # Predicciones válidas por 90 minutos

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
        signal_age_minutes = signal_age.total_seconds() / 60
        
        if signal_age_minutes > PREDICTION_MAX_AGE_MINUTES:
            print(f"⚠️ Señal demasiado antigua ({signal_age_minutes:.1f} min > {PREDICTION_MAX_AGE_MINUTES} min)")
            return None
        
        print(f"✅ Señal encontrada:")
        print(f"   Timestamp: {last_signal['timestamp']}")
        print(f"   Edad: {signal_age_minutes:.1f} minutos")
        print(f"   Signal: {last_signal['signal']}")
        print(f"   Confidence: {last_signal['confidence']:.1f}%")
        print(f"   Price (base): ${last_signal['current_price']:.4f}")
        
        return last_signal.to_dict()
        
    except Exception as e:
        print(f"❌ Error leyendo señales: {e}")
        return None

def validate_prediction_coherence(signal):
    """
    🔥 VALIDACIÓN CRÍTICA: Verifica coherencia de las predicciones
    
    Checks:
    1. pred_close debe estar entre pred_high y pred_low
    2. pred_high > pred_low (obvio pero importante)
    3. Rango no debe ser ni muy pequeño ni muy grande
    
    Returns:
        dict con 'valid' y 'reason'
    """
    pred_high = signal.get('pred_high', 0)
    pred_low = signal.get('pred_low', 0)
    pred_close = signal.get('pred_close', 0)
    base_price = signal.get('current_price', 0)
    
    print(f"\n🔬 VALIDACIÓN DE COHERENCIA:")
    print(f"   Base Price: ${base_price:.4f}")
    print(f"   Pred High:  ${pred_high:.4f}")
    print(f"   Pred Low:   ${pred_low:.4f}")
    print(f"   Pred Close: ${pred_close:.4f}")
    
    # Check 1: High > Low
    if pred_high <= pred_low:
        return {
            'valid': False,
            'reason': f"❌ pred_high (${pred_high:.4f}) ≤ pred_low (${pred_low:.4f})"
        }
    
    # Check 2: Close entre High y Low
    if not (pred_low <= pred_close <= pred_high):
        return {
            'valid': False,
            'reason': f"❌ pred_close (${pred_close:.4f}) NO está entre high y low"
        }
    
    # Check 3: Rango razonable (0.5% - 20%)
    pred_range_pct = ((pred_high - pred_low) / base_price) * 100
    
    print(f"   Rango predicho: {pred_range_pct:.2f}%")
    
    if pred_range_pct < 0.5:
        return {
            'valid': False,
            'reason': f"⚠️ Rango muy pequeño ({pred_range_pct:.2f}% < 0.5%)"
        }
    
    if pred_range_pct > 20:
        return {
            'valid': False,
            'reason': f"⚠️ Rango muy grande ({pred_range_pct:.2f}% > 20%) - volatilidad extrema"
        }
    
    print(f"   ✅ Predicciones coherentes")
    
    return {
        'valid': True,
        'reason': 'Predicciones válidas',
        'pred_range_%': pred_range_pct
    }

def validate_price_sync(signal, current_price):
    """
    🔥 VALIDACIÓN CRÍTICA: Detecta desincronización entre precio base y actual
    
    Si el precio actual está muy lejos del precio base de la predicción,
    la señal ya no es válida.
    
    Returns:
        dict con 'valid', 'drift_%', 'reason', y 'adjusted_signal'
    """
    base_price = signal['current_price']
    pred_high = signal['pred_high']
    pred_low = signal['pred_low']
    pred_close = signal['pred_close']
    
    # Calcular drift
    price_drift = current_price - base_price
    price_drift_pct = (price_drift / base_price) * 100
    
    print(f"\n🎯 VALIDACIÓN DE SINCRONIZACIÓN:")
    print(f"   Precio BASE (predicción): ${base_price:.4f}")
    print(f"   Precio ACTUAL: ${current_price:.4f}")
    print(f"   Drift: ${price_drift:+.4f} ({price_drift_pct:+.2f}%)")
    print(f"   Tolerancia: ±{MAX_PRICE_DRIFT_PCT}%")
    
    # Check: Drift excesivo
    if abs(price_drift_pct) > MAX_PRICE_DRIFT_PCT:
        return {
            'valid': False,
            'drift_%': price_drift_pct,
            'reason': f"❌ Precio actual se alejó demasiado del base ({price_drift_pct:+.2f}% > ±{MAX_PRICE_DRIFT_PCT}%)",
            'adjusted_signal': None
        }
    
    # Check: Precio actual fuera del rango predicho
    if current_price > pred_high:
        outside_pct = ((current_price - pred_high) / base_price) * 100
        print(f"   ⚠️ Precio actual (${current_price:.4f}) > pred_high (${pred_high:.4f}) en {outside_pct:.2f}%")
        
        if outside_pct > 2.0:  # Más de 2% fuera
            return {
                'valid': False,
                'drift_%': price_drift_pct,
                'reason': f"❌ Precio actual superó pred_high en {outside_pct:.2f}%",
                'adjusted_signal': None
            }
    
    elif current_price < pred_low:
        outside_pct = ((pred_low - current_price) / base_price) * 100
        print(f"   ⚠️ Precio actual (${current_price:.4f}) < pred_low (${pred_low:.4f}) en {outside_pct:.2f}%")
        
        if outside_pct > 2.0:
            return {
                'valid': False,
                'drift_%': price_drift_pct,
                'reason': f"❌ Precio actual cayó bajo pred_low en {outside_pct:.2f}%",
                'adjusted_signal': None
            }
    
    # 🔥 AJUSTE INTELIGENTE DE SEÑAL
    # Determinar dirección basándonos en precio actual vs predicciones
    
    # Si precio actual está cerca de pred_high → posible reversión (SELL)
    distance_to_high = abs(current_price - pred_high) / base_price
    distance_to_low = abs(current_price - pred_low) / base_price
    
    adjusted_signal = signal['signal']  # Default: mantener señal original
    
    # Si el precio actual está en el 20% superior del rango predicho
    range_position = (current_price - pred_low) / (pred_high - pred_low)
    
    print(f"   Posición en rango: {range_position*100:.1f}% (0%=low, 100%=high)")
    
    if range_position > 0.8:
        print(f"   ⚠️ Precio en zona alta del rango predicho")
        if signal['signal'] == 'BUY':
            print(f"   🔄 Considerando cambiar BUY → SELL (precio ya cerca de objetivo)")
            # Pero solo si la confianza es alta
            if signal['confidence'] < 80:
                adjusted_signal = 'HOLD'
                print(f"   → Cambiado a HOLD (confianza baja)")
    
    elif range_position < 0.2:
        print(f"   ⚠️ Precio en zona baja del rango predicho")
        if signal['signal'] == 'SELL':
            print(f"   🔄 Considerando cambiar SELL → BUY (precio ya cerca de objetivo)")
            if signal['confidence'] < 80:
                adjusted_signal = 'HOLD'
                print(f"   → Cambiado a HOLD (confianza baja)")
    
    print(f"   ✅ Sincronización válida")
    print(f"   Señal final: {adjusted_signal}")
    
    return {
        'valid': True,
        'drift_%': price_drift_pct,
        'reason': 'Precios sincronizados',
        'adjusted_signal': adjusted_signal,
        'range_position': range_position
    }

def sync_open_orders_with_kraken():
    """Sincroniza open_orders.json con el estado REAL de Kraken"""
    print("\n🔄 Sincronizando con Kraken...")
    
    if not os.path.exists(OPEN_ORDERS_FILE):
        print("✅ No hay órdenes locales")
        return {}
    
    with open(OPEN_ORDERS_FILE, 'r') as f:
        local_orders = json.load(f)
    
    if len(local_orders) == 0:
        print("✅ No hay órdenes locales")
        return {}
    
    print(f"📋 Verificando {len(local_orders)} orden(es) local(es)...")
    
    result = kraken_request('/0/private/OpenOrders', {})
    
    if not result:
        print("⚠️ No se pudo consultar Kraken, manteniendo estado local")
        return local_orders
    
    kraken_open_orders = result.get('open', {})
    
    orders_to_remove = []
    
    for order_id in local_orders.keys():
        if order_id not in kraken_open_orders:
            print(f"🗑️ Orden {order_id} ya no existe en Kraken")
            orders_to_remove.append(order_id)
    
    for order_id in orders_to_remove:
        del local_orders[order_id]
    
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
    """Verifica si ya hay órdenes abiertas"""
    orders = sync_open_orders_with_kraken()
    
    if len(orders) > 0:
        print(f"⚠️ Ya hay {len(orders)} orden(es) abierta(s)")
        return True
    
    return False

def calculate_tp_sl_from_range(signal, current_price, side='buy', tp_factor=0.75):
    """
    Calcula TP/SL basados en el RANGO PREDICHO
    
    IMPORTANTE: Ahora usa 'current_price' como el precio BASE de la predicción
    (no el precio actual en tiempo real)
    """
    
    pred_high = signal.get('pred_high', current_price * 1.03)
    pred_low = signal.get('pred_low', current_price * 0.97)
    pred_close = signal.get('pred_close', current_price)
    
    # Usar precio BASE (cuando se hizo la predicción)
    base_price = signal['current_price']
    
    pred_range = pred_high - pred_low
    half_range = pred_range / 2
    
    print(f"\n🎯 CÁLCULO TP/SL BASADO EN RANGO PREDICHO:")
    print(f"   Precio BASE (predicción): ${base_price:.4f}")
    print(f"   Precio ACTUAL (ejecución): ${current_price:.4f}")
    print(f"   Pred High: ${pred_high:.4f}")
    print(f"   Pred Low: ${pred_low:.4f}")
    print(f"   Pred Close: ${pred_close:.4f}")
    print(f"   Rango predicho: ${pred_range:.4f} ({(pred_range/base_price)*100:.2f}%)")
    print(f"   Mitad del rango: ${half_range:.4f}")
    
    # 🔥 CAMBIO: Calcular desde precio ACTUAL (no base)
    # Pero usando el rango predicho
    if side == 'buy':
        take_profit = current_price + (half_range * tp_factor)
        stop_loss = current_price - half_range
        
        tp_distance = take_profit - current_price
        sl_distance = current_price - stop_loss
        
        tp_pct = (tp_distance / current_price) * 100
        sl_pct = (sl_distance / current_price) * 100
        
        print(f"\n📈 BUY Setup:")
        print(f"   Entry: ${current_price:.4f}")
        print(f"   TP: ${take_profit:.4f} (+{tp_pct:.2f}%)")
        print(f"   SL: ${stop_loss:.4f} (-{sl_pct:.2f}%)")
        
    else:  # SELL
        take_profit = current_price - (half_range * tp_factor)
        stop_loss = current_price + half_range
        
        tp_distance = current_price - take_profit
        sl_distance = stop_loss - current_price
        
        tp_pct = (tp_distance / current_price) * 100
        sl_pct = (sl_distance / current_price) * 100
        
        print(f"\n📉 SELL Setup:")
        print(f"   Entry: ${current_price:.4f}")
        print(f"   TP: ${take_profit:.4f} (-{tp_pct:.2f}%)")
        print(f"   SL: ${stop_loss:.4f} (+{sl_pct:.2f}%)")
    
    risk = abs(current_price - stop_loss)
    reward = abs(take_profit - current_price)
    rr_ratio = reward / risk if risk > 0 else 0
    
    print(f"\n💰 Risk/Reward:")
    print(f"   Riesgo: ${risk:.4f}")
    print(f"   Recompensa: ${reward:.4f}")
    print(f"   R/R: {rr_ratio:.2f}")
    
    warnings = []
    
    if pred_range / base_price < 0.01:
        warnings.append("⚠️ Rango predicho muy pequeño (<1%)")
    
    if pred_range / base_price > 0.15:
        warnings.append("⚠️ Rango predicho muy grande (>15%)")
    
    if rr_ratio < 1.0:
        warnings.append(f"⚠️ R/R bajo ({rr_ratio:.2f} < 1.0)")
    
    if warnings:
        print(f"\n⚠️ Advertencias:")
        for w in warnings:
            print(f"   {w}")
    
    return {
        'stop_loss': round(stop_loss, 4),
        'take_profit': round(take_profit, 4),
        'sl_pct': -sl_pct if side == 'buy' else sl_pct,
        'tp_pct': tp_pct if side == 'buy' else -tp_pct,
        'pred_range': pred_range,
        'pred_range_%': (pred_range / base_price) * 100,
        'half_range': half_range,
        'tp_factor': tp_factor,
        'risk_usd': risk,
        'reward_usd': reward,
        'rr_ratio': rr_ratio,
        'pred_high': pred_high,
        'pred_low': pred_low,
        'pred_close': pred_close,
        'warnings': warnings
    }

def place_market_order_with_separate_sl(side, volume, leverage, entry_price, stop_loss):
    """Coloca orden market + stop-loss separado"""
    print(f"\n📤 Colocando orden MARKET {side.upper()}...")
    print(f"   Volumen: {volume} ADA")
    print(f"   Leverage: {leverage}x")
    print(f"   Entry: ${entry_price:.4f}")
    print(f"   Stop Loss: ${stop_loss:.4f}")
    
    main_order_data = {
        'pair': PAIR,
        'type': side,
        'ordertype': 'market',
        'volume': str(volume),
        'leverage': str(leverage)
    }
    
    print("\n🚀 Ejecutando orden principal...")
    main_result = kraken_request('/0/private/AddOrder', main_order_data)
    
    if not main_result:
        print("❌ Error al colocar orden principal")
        return None
    
    main_order_id = main_result['txid'][0]
    print(f"✅ Orden ejecutada: {main_order_id}")
    
    time.sleep(2)
    
    sl_side = 'sell' if side == 'buy' else 'buy'
    
    if side == 'buy':
        sl_limit_price = stop_loss * 0.995
    else:
        sl_limit_price = stop_loss * 1.005
    
    sl_order_data = {
        'pair': PAIR,
        'type': sl_side,
        'ordertype': 'stop-loss-limit',
        'price': str(stop_loss),
        'price2': str(sl_limit_price),
        'volume': str(volume)
    }
    
    print(f"\n🛡️ Configurando stop-loss...")
    print(f"   Trigger: ${stop_loss:.4f}")
    print(f"   Limit: ${sl_limit_price:.4f}")
    
    sl_result = kraken_request('/0/private/AddOrder', sl_order_data)
    
    if sl_result:
        sl_order_id = sl_result['txid'][0]
        print(f"✅ Stop-Loss configurado: {sl_order_id}")
    else:
        sl_order_id = None
        print(f"⚠️ No se pudo configurar stop-loss")
    
    return {
        'order_id': main_order_id,
        'sl_order_id': sl_order_id,
        'side': side,
        'volume': volume,
        'leverage': leverage,
        'has_auto_sl': sl_order_id is not None,
        'timestamp': datetime.now().isoformat()
    }

def save_order_to_tracking(order_info, signal_info, position_info, tp_sl_info):
    """Guarda orden en archivos de tracking"""
    
    order_data = {
        'timestamp': datetime.now(),
        'order_id': order_info['order_id'],
        'sl_order_id': order_info.get('sl_order_id', None),
        'side': order_info['side'],
        'volume': order_info['volume'],
        'leverage': order_info['leverage'],
        'entry_price': signal_info['current_price'],
        'stop_loss': tp_sl_info['stop_loss'],
        'take_profit': tp_sl_info['take_profit'],
        'confidence': signal_info['confidence'],
        'margin_used': position_info['margin_required'],
        'liquidation_price': position_info['liquidation_price'],
        'expected_risk': position_info['risk_amount'],
        'has_auto_sl': order_info.get('has_auto_sl', False),
        'rr_ratio': tp_sl_info['rr_ratio'],
        'pred_range_%': tp_sl_info.get('pred_range_%', 0)
    }
    
    df_order = pd.DataFrame([order_data])
    
    if os.path.exists(ORDERS_FILE):
        df_order.to_csv(ORDERS_FILE, mode='a', header=False, index=False)
    else:
        df_order.to_csv(ORDERS_FILE, index=False)
    
    print(f"✅ Orden guardada en {ORDERS_FILE}")
    
    open_order = {
        'order_id': order_info['order_id'],
        'sl_order_id': order_info.get('sl_order_id', None),
        'side': order_info['side'],
        'volume': order_info['volume'],
        'leverage': order_info['leverage'],
        'entry_price': signal_info['current_price'],
        'entry_time': datetime.now().isoformat(),
        'stop_loss': tp_sl_info['stop_loss'],
        'take_profit': tp_sl_info['take_profit'],
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
    """Cierra una posición manualmente"""
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

def execute_trading_strategy():
    """🔥 FUNCIÓN PRINCIPAL - Ejecuta estrategia con validaciones"""
    print("="*70)
    print("  💼 ESTRATEGIA DE TRADING CON VALIDACIÓN")
    print("="*70 + "\n")
    
    if check_existing_orders():
        print("\n⸻ Ya hay posiciones abiertas. Saltando ejecución.")
        return
    
    signal = load_last_signal()
    
    if not signal:
        print("\n⚠️ No hay señales válidas")
        return
    
    if signal['signal'] == 'HOLD':
        print(f"\n⸻ Señal es HOLD. No se ejecuta trade.")
        return
    
    # 🔥 VALIDACIÓN 1: Coherencia de predicciones
    coherence = validate_prediction_coherence(signal)
    
    if not coherence['valid']:
        msg = f"❌ Predicción inválida: {coherence['reason']}"
        print(msg)
        send_telegram(msg)
        return
    
    # Obtener precio actual
    current_price = get_current_price()
    
    if not current_price:
        print("❌ No se pudo obtener precio actual")
        return
    
    # 🔥 VALIDACIÓN 2: Sincronización de precios
    sync_check = validate_price_sync(signal, current_price)
    
    if not sync_check['valid']:
        msg = f"❌ Desincronización: {sync_check['reason']}"
        print(msg)
        send_telegram(msg)
        return
    
    # Usar señal ajustada (si fue modificada)
    adjusted_signal = sync_check['adjusted_signal']
    
    if adjusted_signal == 'HOLD':
        print(f"\n⸻ Señal ajustada a HOLD por posición en rango")
        return
    
    signal['signal'] = adjusted_signal  # Actualizar señal
    
    print(f"\n🎯 Ejecutando señal: {signal['signal']}")
    print(f"   Confianza: {signal['confidence']:.1f}%")
    print(f"   Drift de precio: {sync_check['drift_%']:+.2f}%")
    
    balance = get_account_balance()
    
    if not balance or balance < 5:
        msg = f"❌ Balance insuficiente: ${balance:.2f}"
        print(msg)
        send_telegram(msg)
        return
    
    rm = get_risk_manager()
    rm.sync_with_kraken_balance(balance)
    
    side = signal['signal'].lower()
    
    tp_sl_info = calculate_tp_sl_from_range(signal, current_price, side, tp_factor=0.75)
    
    stop_loss = tp_sl_info['stop_loss']
    take_profit = tp_sl_info['take_profit']
    
    print(f"\n📊 RESUMEN:")
    print(f"   Entry: ${current_price:.4f}")
    print(f"   SL: ${stop_loss:.4f} ({tp_sl_info['sl_pct']:+.2f}%)")
    print(f"   TP: ${take_profit:.4f} ({tp_sl_info['tp_pct']:+.2f}%)")
    print(f"   R/R: {tp_sl_info['rr_ratio']:.2f}")
    
    trade_validation = rm.validate_trade(current_price, take_profit, stop_loss, side)
    
    if not trade_validation['valid']:
        msg = f"❌ Trade rechazado: {trade_validation['reason']}"
        print(msg)
        send_telegram(msg)
        return
    
    print(f"\n✅ Trade validado")
    
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
    
    print(f"\n🔥 POSICIÓN:")
    print(f"   Volumen: {position['volume']} ADA")
    print(f"   Leverage: {position['leverage']}x")
    print(f"   Margen: ${position['margin_required']:.2f}")
    
    print(f"\n🚀 EJECUTANDO ORDEN...")
    
    order_result = place_market_order_with_separate_sl(
        side=side,
        volume=position['volume'],
        leverage=position['leverage'],
        entry_price=current_price,
        stop_loss=stop_loss
    )
    
    if not order_result:
        msg = "❌ Error al ejecutar orden"
        print(msg)
        send_telegram(msg)
        return
    
    save_order_to_tracking(order_result, signal, position, tp_sl_info)
    rm.reserve_margin(position['margin_required'])
    
    msg = f"""
🚀 *ORDEN EJECUTADA* (Validada)

📊 *Validaciones:*
   ✅ Predicciones coherentes
   ✅ Precios sincronizados ({sync_check['drift_%']:+.2f}%)
   ✅ Close entre High/Low

🎯 *Setup:*
   • Señal: {signal['signal']}
   • Entry: ${current_price:.4f}
   • TP: ${take_profit:.4f} ({tp_sl_info['tp_pct']:+.2f}%)
   • SL: ${stop_loss:.4f} ({tp_sl_info['sl_pct']:+.2f}%)
   • R/R: {tp_sl_info['rr_ratio']:.2f}

💼 *Posición:*
   • Volumen: {position['volume']} ADA
   • Leverage: {position['leverage']}x
   • Margen: ${position['margin_required']:.2f}

🆔 `{order_result['order_id']}`
"""
    
    print(msg.replace('*', '').replace('`', ''))
    send_telegram(msg)
    
    print("\n" + "="*70)
    print("  ✅ ORDEN EJECUTADA")
    print("="*70)

def monitor_orders():
    """Monitorea órdenes abiertas"""
    print("\n🔍 Monitoreando órdenes...")
    
    orders = sync_open_orders_with_kraken()
    
    if len(orders) == 0:
        print("ℹ️ No hay órdenes")
        return
    
    current_price = get_current_price()
    
    if not current_price:
        return
    
    for order_id, order_info in list(orders.items()):
        print(f"\n📊 Orden {order_id}:")
        print(f"   Entry: ${order_info['entry_price']:.4f}")
        print(f"   Current: ${current_price:.4f}")
        print(f"   TP: ${order_info['take_profit']:.4f}")
        print(f"   SL: ${order_info['stop_loss']:.4f}")
        
        close_reason = None
        
        if order_info['side'] == 'buy':
            pnl_pct = ((current_price - order_info['entry_price']) / order_info['entry_price']) * 100
            
            if current_price >= order_info['take_profit']:
                close_reason = 'TP'
            else:
                print(f"💹 P&L: {pnl_pct:+.2f}%")
        else:
            pnl_pct = ((order_info['entry_price'] - current_price) / order_info['entry_price']) * 100
            
            if current_price <= order_info['take_profit']:
                close_reason = 'TP'
            else:
                print(f"💹 P&L: {pnl_pct:+.2f}%")
        
        entry_time = datetime.fromisoformat(order_info['entry_time'])
        time_open = datetime.now() - entry_time
        
        if time_open > timedelta(hours=3.5):
            close_reason = 'TIMEOUT'
        
        if close_reason:
            success = close_position(order_id, order_info['side'], order_info['volume'])
            
            if success:
                del orders[order_id]
                
                msg = f"🔒 *Posición Cerrada*\n\nRazón: {close_reason}\nP&L: {pnl_pct:+.2f}%"
                send_telegram(msg)
    
    with open(OPEN_ORDERS_FILE, 'w') as f:
        json.dump(orders, f, indent=2)

if __name__ == "__main__":
    try:
        execute_trading_strategy()
        time.sleep(2)
        monitor_orders()
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        print(error_msg)
        send_telegram(error_msg)
        raise
