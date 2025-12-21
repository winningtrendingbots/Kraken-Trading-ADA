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
    """
    🔍 Detecta el par correcto de ADA en Kraken
    
    Kraken usa diferentes formatos:
    - ADAUSD, XADAZUSD, ADAEUR, etc.
    
    Esta función detecta cuál está disponible
    """
    print("\n🔍 DETECTANDO PAR CORRECTO DE ADA...")
    
    # Posibles pares que Kraken usa para ADA
    possible_pairs = [
        'ADAUSD',      # Formato simple
        'XADAZUSD',    # Formato extendido
        'ADAUSDT',     # Tether
        'ADAEUR',      # Euro
        'ADAGBP'       # Libra
    ]
    
    try:
        url = f"{KRAKEN_API_URL}/0/public/AssetPairs"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'result' in data:
                available_pairs = data['result'].keys()
                
                # Buscar pares ADA
                ada_pairs = [p for p in available_pairs if 'ADA' in p.upper()]
                
                print(f"✅ Pares ADA disponibles: {ada_pairs}")
                
                # Intentar encontrar USD primero
                for pair in possible_pairs:
                    if pair in ada_pairs:
                        print(f"✅ Par detectado: {pair}")
                        return pair
                
                # Si no encontramos ninguno de los esperados, usar el primero disponible
                if ada_pairs:
                    print(f"⚠️ Usando primer par disponible: {ada_pairs[0]}")
                    return ada_pairs[0]
        
        print("❌ No se pudo detectar par ADA")
        return None
        
    except Exception as e:
        print(f"❌ Error detectando par: {e}")
        return None

def get_current_price(retries=3, delay=2):
    """
    Obtiene precio actual de ADA con detección automática del par
    """
    # 🆕 Detectar par correcto
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
                # Kraken devuelve el par en el formato que usa internamente
                result_pair = list(data['result'].keys())[0]
                price = float(data['result'][result_pair]['c'][0])
                print(f"✅ Precio obtenido: ${price:.4f} (par: {result_pair})")
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
    """Obtiene balance real de Kraken"""
    data = {'nonce': str(int(1000*time.time()))}
    result = kraken_request('/0/private/Balance', data)
    return result

def get_usd_balance():
    """
    🆕 Obtiene balance en USD disponible (mejorado)
    Maneja múltiples formatos de Kraken
    """
    balance = get_balance()
    
    if 'result' in balance:
        # Kraken puede usar diferentes símbolos para USD
        usd_symbols = ['ZUSD', 'USD', 'USDT']
        
        for symbol in usd_symbols:
            if symbol in balance['result']:
                usd = float(balance['result'][symbol])
                print(f"💰 Balance {symbol}: ${usd:.2f}")
                return usd
        
        print("⚠️ No se encontró balance USD")
        print(f"Balances disponibles: {list(balance['result'].keys())}")
    
    return 0

def place_order(side, volume, price, tp_price, sl_price):
    """
    🆕 Coloca orden con par correcto detectado automáticamente
    """
    pair = detect_ada_pair()
    
    if not pair:
        return {'error': ['No se pudo detectar par de trading']}
    
    data = {
        'nonce': str(int(1000*time.time())),
        'ordertype': 'limit' if price else 'market',
        'type': side,
        'volume': str(volume),
        'pair': pair,  # 🔥 Usar par detectado
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
    
    return round(tp, 4), round(sl, 4)

def monitor_orders():
    """Monitorea órdenes abiertas y cierra por TP/SL/tiempo - CADA 15 MINUTOS"""
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
        
        # 4. STOP LOSS PROGRESIVO
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
                cancel_result = cancel_order(txid)
                print(f"   Kraken cancel: {cancel_result}")
            else:
                print("   ⚠️ MODO SIMULACIÓN - Orden NO cancelada en Kraken")
            
            # Calcular P&L
            if side == 'buy':
                pnl = (close_price - entry_price) * volume
                pnl_pct = ((close_price - entry_price) / entry_price) * 100
            else:
                pnl = (entry_price - close_price) * volume
                pnl_pct = ((entry_price - close_price) / entry_price) * 100
            
            # Actualizar capital y liberar margen
            risk_manager.update_after_trade(pnl, margin_released=margin_reserved)
            
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
    
    with open(OPEN_ORDERS_FILE, 'w') as f:
        json.dump(updated_orders, f, indent=2)
    
    if len(updated_orders) > 0:
        print(f"✅ Monitoreo completado: {len(updated_orders)} órdenes activas")
    else:
        print("✅ Todas las órdenes fueron cerradas")

def execute_signal():
    """Lee última señal y ejecuta si es BUY/SELL con gestión de riesgo"""
    
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
    
    # 🔥 SINCRONIZAR CON BALANCE REAL DE KRAKEN
    if LIVE_TRADING:
        kraken_balance = get_usd_balance()
        if kraken_balance > 0:
            risk_manager.sync_with_kraken_balance(kraken_balance)
            print(f"✅ Balance sincronizado con Kraken: ${kraken_balance:.2f}")
    
    risk_manager.print_stats()
    
    # Verificar si ya hay una orden abierta
    if os.path.exists(OPEN_ORDERS_FILE):
        with open(OPEN_ORDERS_FILE, 'r') as f:
            open_orders = json.load(f)
        if len(open_orders) >= 1:  # 🆕 Solo 1 orden a la vez
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
    print(f"   Risk: ${trade_validation['risk']:.4f}")
    print(f"   Reward: ${trade_validation['reward']:.4f}")
    
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
    print(f"💰 Precio: ${current_price:.4f}")
    print(f"📈 Volumen: {volume} ADA (${position['position_value']:.2f})")
    print(f"   • Leverage: {position['leverage']}x")
    print(f"   • Riesgo: ${position['risk_amount']:.2f}")
    print(f"   • Margen Req: ${position['margin_required']:.2f}")
    print(f"   • Capital usado: {position['capital_used_%']:.1f}%")
    print(f"🎯 TP: ${tp:.4f} ({((tp-current_price)/current_price*100):+.2f}%)")
    print(f"🛑 SL: ${sl:.4f} ({((sl-current_price)/current_price*100):+.2f}%)")
    print(f"⚠️ Liquidación: ${position['liquidation_price']:.4f}")
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
            
            # Reservar margen
            risk_manager.reserve_margin(position['margin_required'])
            
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
                'risk_amount': position['risk_amount'],
                'margin_required': position['margin_required'],
                'leverage': position['leverage'],
                'liquidation_price': position['liquidation_price']
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
                'entry_price': current_price,
                'volume': volume,
                'tp': tp,
                'sl': sl,
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
        # MODO SIMULACIÓN
        print("💼 MODO SIMULACIÓN - Orden NO enviada a Kraken")
        print("   ⚠️ Para activar trading real, cambiar LIVE_TRADING = True")
        
        # Simular orden guardada
        txid = f"SIM_{int(time.time())}"
        
        # Reservar margen en simulación
        risk_manager.reserve_margin(position['margin_required'])
        
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
            'risk_amount': position['risk_amount'],
            'margin_required': position['margin_required'],
            'leverage': position['leverage'],
            'liquidation_price': position['liquidation_price']
        }
        
        orders = []
        if os.path.exists(OPEN_ORDERS_FILE):
            with open(OPEN_ORDERS_FILE, 'r') as f:
                orders = json.load(f)
        
        orders.append(order_data)
        with open(OPEN_ORDERS_FILE, 'w') as f:
            json.dump(orders, f, indent=2)
        
        # CSV
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
            'leverage': position['leverage'],
            'order_executed': 'SIMULATED',
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
💼 *SIMULACIÓN - Nueva Orden*

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

⚠️ *MODO SIMULACIÓN*
Para trading real: LIVE_TRADING = True
"""
        send_telegram(msg)

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
