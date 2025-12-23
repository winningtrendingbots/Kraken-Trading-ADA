"""
DIAGNÓSTICO DE BALANCE DE KRAKEN
Muestra TODOS los balances para identificar el símbolo correcto
"""

import requests
import hmac
import hashlib
import base64
import time
import urllib.parse
import os

KRAKEN_API_KEY = os.environ.get('KRAKEN_API_KEY', '')
KRAKEN_API_SECRET = os.environ.get('KRAKEN_API_SECRET', '')
KRAKEN_API_URL = "https://api.kraken.com"

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
    req = requests.post(KRAKEN_API_URL + uri_path, headers=headers, data=data, timeout=15)
    return req.json()

print("="*70)
print("  🔍 DIAGNÓSTICO COMPLETO DE BALANCE KRAKEN")
print("="*70)

# 1. Balance general
print("\n📊 BALANCE GENERAL (Balance):")
print("-"*70)

data = {'nonce': str(int(1000*time.time()))}
balance = kraken_request('/0/private/Balance', data)

if 'result' in balance:
    if not balance['result']:
        print("⚠️ Balance vacío")
    else:
        for asset, amount in balance['result'].items():
            amount_float = float(amount)
            if amount_float > 0:
                print(f"   {asset}: {amount_float:.4f}")
else:
    print(f"❌ Error: {balance.get('error', 'Unknown')}")

# 2. Balance extendido
print("\n📊 BALANCE EXTENDIDO (BalanceEx):")
print("-"*70)

data = {'nonce': str(int(1000*time.time()))}
balance_ex = kraken_request('/0/private/BalanceEx', data)

if 'result' in balance_ex:
    if not balance_ex['result']:
        print("⚠️ Balance extendido vacío")
    else:
        for asset, details in balance_ex['result'].items():
            print(f"   {asset}:")
            print(f"      Balance: {details.get('balance', 0)}")
            print(f"      Hold Trade: {details.get('hold_trade', 0)}")
else:
    print(f"❌ Error: {balance_ex.get('error', 'Unknown')}")

# 3. Trade balance (MARGEN ESPECÍFICO)
print("\n💰 TRADE BALANCE (Margen para trading):")
print("-"*70)

data = {'nonce': str(int(1000*time.time()))}
trade_balance = kraken_request('/0/private/TradeBalance', data)

if 'result' in trade_balance:
    result = trade_balance['result']
    
    equity = float(result.get('eb', 0))          # Equity (balance total)
    margin = float(result.get('m', 0))           # Margen usado
    free_margin = float(result.get('mf', 0))     # Margen libre
    margin_level = float(result.get('ml', 0))    # Nivel de margen
    unrealized = float(result.get('n', 0))       # P&L no realizado
    cost = float(result.get('c', 0))             # Costo de posiciones
    valuation = float(result.get('v', 0))        # Valoración de posiciones
    
    print(f"   💰 Equity (Total): {equity:.2f} EUR")
    print(f"   📊 Margen Usado: {margin:.2f} EUR")
    print(f"   ✅ Margen Libre: {free_margin:.2f} EUR")
    print(f"   📈 Nivel Margen: {margin_level:.2f}%")
    print(f"   💵 P&L No Realizado: {unrealized:.2f} EUR")
    print(f"   💼 Costo Posiciones: {cost:.2f} EUR")
    print(f"   📊 Valoración: {valuation:.2f} EUR")
    
    print(f"\n   🎯 BALANCE DISPONIBLE PARA TRADING: {free_margin:.2f} EUR")
    
else:
    print(f"❌ Error: {trade_balance.get('error', 'Unknown')}")

# 4. Posiciones abiertas
print("\n📍 POSICIONES ABIERTAS:")
print("-"*70)

data = {'nonce': str(int(1000*time.time()))}
positions = kraken_request('/0/private/OpenPositions', data)

if 'result' in positions:
    if not positions['result']:
        print("✅ No hay posiciones abiertas")
    else:
        for pos_id, pos_data in positions['result'].items():
            print(f"   ID: {pos_id}")
            print(f"      Par: {pos_data.get('pair', 'N/A')}")
            print(f"      Tipo: {pos_data.get('type', 'N/A')}")
            print(f"      Volumen: {pos_data.get('vol', 0)}")
            print(f"      Costo: {pos_data.get('cost', 0)}")
            print(f"      P&L: {pos_data.get('net', 0)}")
else:
    print(f"❌ Error: {positions.get('error', 'Unknown')}")

# 5. Órdenes abiertas
print("\n📋 ÓRDENES ABIERTAS:")
print("-"*70)

data = {'nonce': str(int(1000*time.time()))}
orders = kraken_request('/0/private/OpenOrders', data)

if 'result' in orders:
    open_orders = orders['result'].get('open', {})
    if not open_orders:
        print("✅ No hay órdenes abiertas")
    else:
        for order_id, order_data in open_orders.items():
            print(f"   ID: {order_id}")
            print(f"      Par: {order_data['descr'].get('pair', 'N/A')}")
            print(f"      Tipo: {order_data['descr'].get('type', 'N/A')}")
            print(f"      Volumen: {order_data.get('vol', 0)}")
else:
    print(f"❌ Error: {orders.get('error', 'Unknown')}")

print("\n" + "="*70)
print("  ✅ DIAGNÓSTICO COMPLETADO")
print("="*70)

print("\n💡 RECOMENDACIÓN:")
print("   Si ves 47€ en 'Margen Libre' pero el bot detecta $0.01,")
print("   necesitas actualizar get_margin_balance() para usar TradeBalance")
print("   en lugar de Balance normal.")
