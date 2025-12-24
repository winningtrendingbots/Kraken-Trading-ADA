"""
DIAGNÓSTICO DEL SISTEMA
✅ Verifica que todo esté configurado correctamente
✅ Chequea archivos, modelos, APIs
✅ Reporta el estado completo del sistema
"""

import os
import json
import sys
from datetime import datetime

def check_symbol(symbol='✅', text='OK'):
    """Helper para imprimir checks"""
    print(f"   {symbol} {text}")

def check_files():
    """Verifica existencia de archivos críticos"""
    print("\n" + "="*70)
    print("  📁 VERIFICANDO ARCHIVOS")
    print("="*70)
    
    required_files = {
        'Scripts Python': [
            'adausd_lstm.py',
            'predict_and_filter.py',
            'predict_enhanced.py',  # Nuevo
            'kraken_trader.py',
            'trading_orchestrator.py',
            'risk_manager.py',
            'evaluate_predictions.py',  # Nuevo
            'requirements.txt'
        ],
        'Workflows': [
            '1-train-model.yml',
            '2-predict-and-trade.yml',
            '3-monitor-orders.yml'  # Verificar si existe
        ],
        'Configuración': [
            '.gitignore'
        ]
    }
    
    all_ok = True
    
    for category, files in required_files.items():
        print(f"\n📂 {category}:")
        for file in files:
            # Buscar en raíz o en .github/workflows
            if category == 'Workflows':
                file_path = f'.github/workflows/{file}'
            else:
                file_path = file
            
            if os.path.exists(file_path):
                check_symbol('✅', f'{file}')
            else:
                check_symbol('❌', f'{file} - NO ENCONTRADO')
                all_ok = False
    
    return all_ok

def check_model():
    """Verifica modelo entrenado"""
    print("\n" + "="*70)
    print("  🤖 VERIFICANDO MODELO")
    print("="*70)
    
    model_dir = 'ADAUSD_MODELS'
    
    if not os.path.exists(model_dir):
        check_symbol('❌', f'Directorio {model_dir} no existe')
        return False
    
    check_symbol('✅', f'Directorio {model_dir} existe')
    
    required_model_files = [
        'adausd_lstm_1h.pth',
        'scaler_input_1h.pkl',
        'scaler_output_1h.pkl',
        'config_1h.json'
    ]
    
    all_ok = True
    
    for file in required_model_files:
        file_path = os.path.join(model_dir, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            check_symbol('✅', f'{file} ({size:,} bytes)')
        else:
            check_symbol('❌', f'{file} - NO ENCONTRADO')
            all_ok = False
    
    # Verificar training summary
    summary_path = os.path.join(model_dir, 'training_summary.json')
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            
            print(f"\n   📊 Última sesión de entrenamiento:")
            print(f"      Fecha: {summary.get('timestamp', 'N/A')}")
            print(f"      Epochs: {summary.get('epochs_completed', 'N/A')}")
            print(f"      Val Loss: {summary.get('best_val_loss', 'N/A'):.6f}")
            
            # Overfitting diagnosis
            if 'overfitting_diagnosis' in summary:
                diag = summary['overfitting_diagnosis']
                print(f"      Gap Train-Test: {diag.get('gap_train_test', 'N/A'):.4f}")
                print(f"      Status: {diag.get('status', 'N/A')}")
            
        except Exception as e:
            check_symbol('⚠️', f'Error leyendo summary: {e}')
    
    return all_ok

def check_data_files():
    """Verifica archivos de datos de trading"""
    print("\n" + "="*70)
    print("  📊 VERIFICANDO ARCHIVOS DE DATOS")
    print("="*70)
    
    data_files = {
        'trading_signals.csv': 'Señales de trading',
        'prediction_tracker.csv': 'Tracker de predicciones',
        'orders_executed.csv': 'Órdenes ejecutadas',
        'open_orders.json': 'Órdenes abiertas',
        'kraken_trades.csv': 'Trades cerrados',
        'risk_config.json': 'Configuración de riesgo',
        'ADAUSD_1h_data.csv': 'Datos históricos'
    }
    
    for file, description in data_files.items():
        if os.path.exists(file):
            size = os.path.getsize(file)
            
            if file.endswith('.csv'):
                try:
                    import pandas as pd
                    df = pd.read_csv(file)
                    check_symbol('✅', f'{file}: {len(df)} registros ({size:,} bytes)')
                except:
                    check_symbol('⚠️', f'{file}: existe pero error al leer')
            
            elif file.endswith('.json'):
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                    
                    if isinstance(data, dict):
                        count = len(data)
                    elif isinstance(data, list):
                        count = len(data)
                    else:
                        count = 1
                    
                    check_symbol('✅', f'{file}: {count} items ({size:,} bytes)')
                except:
                    check_symbol('⚠️', f'{file}: existe pero error al leer')
        else:
            check_symbol('⏳', f'{file}: aún no generado')

def check_environment():
    """Verifica variables de entorno y APIs"""
    print("\n" + "="*70)
    print("  🔐 VERIFICANDO CONFIGURACIÓN")
    print("="*70)
    
    env_vars = {
        'KRAKEN_API_KEY': 'Kraken API Key',
        'KRAKEN_API_SECRET': 'Kraken API Secret',
        'TELEGRAM_API': 'Telegram Bot Token',
        'CHAT_ID': 'Telegram Chat ID'
    }
    
    all_ok = True
    
    for var, description in env_vars.items():
        value = os.environ.get(var, '')
        
        if value:
            # Mostrar solo primeros/últimos caracteres
            if len(value) > 10:
                masked = f"{value[:4]}...{value[-4:]}"
            else:
                masked = "***"
            check_symbol('✅', f'{description}: {masked}')
        else:
            check_symbol('❌', f'{description}: NO CONFIGURADO')
            all_ok = False
    
    return all_ok

def check_python_packages():
    """Verifica paquetes Python instalados"""
    print("\n" + "="*70)
    print("  📦 VERIFICANDO PAQUETES PYTHON")
    print("="*70)
    
    required_packages = {
        'pandas': '>=2.0.0',
        'numpy': '>=1.24.0',
        'torch': '>=2.0.0',
        'sklearn': '>=1.3.0',
        'yfinance': '>=0.2.28',
        'matplotlib': '>=3.7.0',
        'requests': '>=2.31.0',
        'joblib': '>=1.3.0',
        'tqdm': '>=4.66.0'
    }
    
    all_ok = True
    
    for package, min_version in required_packages.items():
        try:
            if package == 'sklearn':
                import sklearn
                version = sklearn.__version__
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'N/A')
            
            check_symbol('✅', f'{package}: {version}')
        except ImportError:
            check_symbol('❌', f'{package}: NO INSTALADO')
            all_ok = False
    
    return all_ok

def check_gpu():
    """Verifica disponibilidad de GPU"""
    print("\n" + "="*70)
    print("  🖥️  VERIFICANDO GPU")
    print("="*70)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            check_symbol('✅', f'GPU disponible: {torch.cuda.get_device_name(0)}')
            check_symbol('ℹ️', f'GPUs detectadas: {gpu_count}')
            check_symbol('ℹ️', f'CUDA version: {torch.version.cuda}')
        else:
            check_symbol('⚠️', 'GPU no disponible - usando CPU')
            check_symbol('ℹ️', 'El entrenamiento será más lento en CPU')
    
    except Exception as e:
        check_symbol('❌', f'Error verificando GPU: {e}')

def check_git_config():
    """Verifica configuración de Git"""
    print("\n" + "="*70)
    print("  🔧 VERIFICANDO GIT")
    print("="*70)
    
    try:
        import subprocess
        
        # Check git user
        result = subprocess.run(['git', 'config', 'user.name'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            check_symbol('✅', f'Git user: {result.stdout.strip()}')
        else:
            check_symbol('⚠️', 'Git user no configurado')
        
        # Check git email
        result = subprocess.run(['git', 'config', 'user.email'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            check_symbol('✅', f'Git email: {result.stdout.strip()}')
        else:
            check_symbol('⚠️', 'Git email no configurado')
        
        # Check remote
        result = subprocess.run(['git', 'remote', '-v'], 
                              capture_output=True, text=True)
        if result.returncode == 0 and result.stdout:
            check_symbol('✅', 'Git remote configurado')
        else:
            check_symbol('⚠️', 'Git remote no configurado')
    
    except Exception as e:
        check_symbol('❌', f'Error verificando Git: {e}')

def generate_report():
    """Genera reporte completo del sistema"""
    print("\n" + "="*70)
    print("  🎯 DIAGNÓSTICO COMPLETO DEL SISTEMA")
    print("="*70)
    print(f"\nFecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version}")
    print(f"OS: {sys.platform}")
    
    # Run all checks
    files_ok = check_files()
    model_ok = check_model()
    check_data_files()
    env_ok = check_environment()
    packages_ok = check_python_packages()
    check_gpu()
    check_git_config()
    
    # Summary
    print("\n" + "="*70)
    print("  📋 RESUMEN")
    print("="*70)
    
    status = []
    
    if files_ok:
        status.append("✅ Archivos OK")
    else:
        status.append("❌ Faltan archivos")
    
    if model_ok:
        status.append("✅ Modelo OK")
    else:
        status.append("❌ Modelo no encontrado")
    
    if env_ok:
        status.append("✅ APIs configuradas")
    else:
        status.append("❌ APIs no configuradas")
    
    if packages_ok:
        status.append("✅ Paquetes OK")
    else:
        status.append("❌ Faltan paquetes")
    
    for s in status:
        print(f"   {s}")
    
    # Overall status
    print("\n" + "="*70)
    if all([files_ok, model_ok, env_ok, packages_ok]):
        print("  🎉 SISTEMA LISTO PARA OPERAR")
    elif model_ok and env_ok and packages_ok:
        print("  ⚠️  SISTEMA FUNCIONAL (algunos archivos opcionales faltan)")
    else:
        print("  ❌ SISTEMA REQUIERE CONFIGURACIÓN")
        print("\n  Pasos siguientes:")
        if not model_ok:
            print("     1. Ejecutar: python adausd_lstm.py")
        if not env_ok:
            print("     2. Configurar variables de entorno")
        if not packages_ok:
            print("     3. Ejecutar: pip install -r requirements.txt")
    
    print("="*70)

if __name__ == "__main__":
    try:
        generate_report()
    except Exception as e:
        print(f"\n❌ Error en diagnóstico: {e}")
        import traceback
        traceback.print_exc()
