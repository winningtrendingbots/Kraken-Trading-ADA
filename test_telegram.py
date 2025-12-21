"""
TEST DE TELEGRAM - Diagnóstico completo
Verifica por qué no llegan los mensajes
"""

import os
import requests
import time

# Credenciales
TELEGRAM_API = os.environ.get('TELEGRAM_API', '')
CHAT_ID = os.environ.get('CHAT_ID', '')

print("="*70)
print("  📱 TEST DE TELEGRAM")
print("="*70 + "\n")

# Test 1: Verificar variables de entorno
print("1️⃣ VERIFICANDO VARIABLES DE ENTORNO:")
print("-" * 70)

if not TELEGRAM_API:
    print("❌ TELEGRAM_API no encontrado")
    print("   Debe estar en GitHub Secrets como: TELEGRAM_API")
else:
    print(f"✅ TELEGRAM_API encontrado")
    print(f"   Formato: {TELEGRAM_API[:10]}...{TELEGRAM_API[-4:]}")

if not CHAT_ID:
    print("❌ TELEGRAM_CHAT_ID no encontrado")
    print("   Debe estar en GitHub Secrets como: TELEGRAM_CHAT_ID")
else:
    print(f"✅ TELEGRAM_CHAT_ID encontrado: {CHAT_ID}")

if not TELEGRAM_API or not CHAT_ID:
    print("\n⚠️  CONFIGURACIÓN INCOMPLETA")
    print("\nPara configurar:")
    print("1. Ve a GitHub → Settings → Secrets and variables → Actions")
    print("2. Añade:")
    print("   TELEGRAM_API=tu_bot_token")
    print("   TELEGRAM_CHAT_ID=tu_chat_id")
    exit(1)

print()

# Test 2: Verificar validez del bot token
print("2️⃣ VERIFICANDO BOT TOKEN:")
print("-" * 70)

try:
    url = f"https://api.telegram.org/bot{TELEGRAM_API}/getMe"
    response = requests.get(url, timeout=10)
    data = response.json()
    
    if data.get('ok'):
        bot_info = data['result']
        print(f"✅ Bot válido:")
        print(f"   Nombre: {bot_info.get('first_name', 'N/A')}")
        print(f"   Username: @{bot_info.get('username', 'N/A')}")
        print(f"   ID: {bot_info.get('id', 'N/A')}")
    else:
        print(f"❌ Token inválido: {data.get('description', 'Unknown error')}")
        exit(1)
except Exception as e:
    print(f"❌ Error conectando con Telegram: {e}")
    exit(1)

print()

# Test 3: Verificar permisos del chat
print("3️⃣ VERIFICANDO CHAT:")
print("-" * 70)

try:
    url = f"https://api.telegram.org/bot{TELEGRAM_API}/getChat"
    response = requests.post(url, data={'chat_id': CHAT_ID}, timeout=10)
    data = response.json()
    
    if data.get('ok'):
        chat_info = data['result']
        print(f"✅ Chat válido:")
        print(f"   Tipo: {chat_info.get('type', 'N/A')}")
        
        if 'title' in chat_info:
            print(f"   Título: {chat_info['title']}")
        if 'username' in chat_info:
            print(f"   Username: @{chat_info['username']}")
        if 'first_name' in chat_info:
            print(f"   Nombre: {chat_info['first_name']}")
    else:
        error_desc = data.get('description', 'Unknown error')
        print(f"❌ Chat inválido: {error_desc}")
        
        if "chat not found" in error_desc.lower():
            print("\n⚠️  SOLUCIÓN:")
            print("   1. Asegúrate de haber iniciado el bot (envía /start)")
            print("   2. Verifica que el CHAT_ID sea correcto")
        
        exit(1)
except Exception as e:
    print(f"❌ Error verificando chat: {e}")
    exit(1)

print()

# Test 4: Enviar mensaje de prueba
print("4️⃣ ENVIANDO MENSAJE DE PRUEBA:")
print("-" * 70)

test_messages = [
    "🧪 Test 1: Mensaje simple",
    "*🧪 Test 2:* Markdown básico",
    "```\n🧪 Test 3: Code block\n```"
]

for i, msg in enumerate(test_messages, 1):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_API}/sendMessage"
        
        response = requests.post(
            url,
            data={
                'chat_id': CHAT_ID,
                'text': msg,
                'parse_mode': 'Markdown'
            },
            timeout=10
        )
        
        data = response.json()
        
        if data.get('ok'):
            message_id = data['result']['message_id']
            print(f"✅ Test {i} enviado (ID: {message_id})")
        else:
            error_desc = data.get('description', 'Unknown error')
            print(f"❌ Test {i} falló: {error_desc}")
            
            if "parse_mode" in error_desc.lower():
                print("   → Problema con formato Markdown")
            
        time.sleep(1)  # Evitar rate limiting
        
    except Exception as e:
        print(f"❌ Test {i} error: {e}")

print()

# Test 5: Mensaje complejo (como los del bot)
print("5️⃣ ENVIANDO MENSAJE COMPLEJO:")
print("-" * 70)

complex_msg = """
🤖 *TEST COMPLETO - Bot Trading*

✅ Sistema operativo
📊 Modelo funcionando
💰 Balance: $10.00

📈 *Última predicción:*
   Señal: BUY
   Confianza: 75%
   Precio: $1.23

⏰ Timestamp: """ + time.strftime("%Y-%m-%d %H:%M:%S UTC")

try:
    url = f"https://api.telegram.org/bot{TELEGRAM_API}/sendMessage"
    
    response = requests.post(
        url,
        data={
            'chat_id': CHAT_ID,
            'text': complex_msg,
            'parse_mode': 'Markdown'
        },
        timeout=10
    )
    
    data = response.json()
    
    if data.get('ok'):
        message_id = data['result']['message_id']
        print(f"✅ Mensaje complejo enviado (ID: {message_id})")
    else:
        error_desc = data.get('description', 'Unknown error')
        print(f"❌ Mensaje complejo falló: {error_desc}")
        print("\n📋 Respuesta completa:")
        print(data)
        
except Exception as e:
    print(f"❌ Error: {e}")

print()

# Test 6: Verificar rate limits
print("6️⃣ VERIFICANDO RATE LIMITS:")
print("-" * 70)

print("Enviando múltiples mensajes rápidos...")

success_count = 0
fail_count = 0

for i in range(5):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_API}/sendMessage"
        response = requests.post(
            url,
            data={
                'chat_id': CHAT_ID,
                'text': f"📊 Rate limit test #{i+1}"
            },
            timeout=5
        )
        
        if response.json().get('ok'):
            success_count += 1
        else:
            fail_count += 1
            
    except Exception as e:
        fail_count += 1
        print(f"   Error en test {i+1}: {e}")

print(f"✅ Exitosos: {success_count}/5")
print(f"❌ Fallidos: {fail_count}/5")

if fail_count > 2:
    print("\n⚠️  RATE LIMIT DETECTADO")
    print("   Solución: Añadir delays entre mensajes")

print()

# Resumen final
print("="*70)
print("  📊 RESUMEN DEL DIAGNÓSTICO")
print("="*70)

issues_found = []

if not TELEGRAM_API or not CHAT_ID:
    issues_found.append("Variables de entorno faltantes")

if fail_count > 0:
    issues_found.append(f"Algunos mensajes fallaron ({fail_count}/5)")

if len(issues_found) == 0:
    print("\n✅ TODO FUNCIONA CORRECTAMENTE")
    print("\nSi aún no recibes mensajes del bot:")
    print("1. Verifica que los secrets estén en GitHub Actions")
    print("2. Revisa que el workflow tenga acceso a los secrets")
    print("3. Comprueba los logs de GitHub Actions")
else:
    print("\n⚠️  PROBLEMAS ENCONTRADOS:")
    for issue in issues_found:
        print(f"   • {issue}")

print("\n" + "="*70)
