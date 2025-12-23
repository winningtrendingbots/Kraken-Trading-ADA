"""
GESTOR DE RIESGO Y CAPITAL CON MARGEN - CONFIGURADO PARA BANCA PEQUEÑA (10€)

🆕 Optimizado para:
- Banca inicial: 10€
- Leverage: 3-5x (más seguro que 10x con banca pequeña)
- Posiciones pequeñas pero seguras
"""

import json
import os
from datetime import datetime

class RiskManager:
    def __init__(self, 
                 initial_capital=40,             # 🆕 10€ inicial (se sincroniza con Kraken)
                 risk_per_trade=0.01,             # 2% de riesgo por trade
                 max_leverage=5,                  # 🆕 5x max (más seguro con banca pequeña)
                 margin_usage_limit=0.6,          # Usar máximo 60% del margen
                 max_open_positions=3,            # 🆕 Solo 1 posición (con banca pequeña)
                 min_rr_ratio=1.5,               # Mínimo Risk/Reward 1:1.5
                 liquidation_buffer=0.30,         # 30% buffer antes de liquidación
                 max_position_size=0.40,          # Máximo 40% del capital por posición
                 confidence_threshold=75):        # 🆕 Confianza mínima 70% (más conservador)
        
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_leverage = max_leverage
        self.margin_usage_limit = margin_usage_limit
        self.max_open_positions = max_open_positions
        self.min_rr_ratio = min_rr_ratio
        self.liquidation_buffer = liquidation_buffer
        self.max_position_size = max_position_size
        self.confidence_threshold = confidence_threshold
        
        # Estado actual
        self.config_file = 'risk_config.json'
        self.load_config()
    
    def load_config(self):
        """Carga configuración guardada o usa defaults"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                self.current_capital = config.get('current_capital', self.initial_capital)
                self.total_profit = config.get('total_profit', 0)
                self.total_trades = config.get('total_trades', 0)
                self.win_count = config.get('win_count', 0)
                self.margin_used = config.get('margin_used', 0)
        else:
            self.current_capital = self.initial_capital
            self.total_profit = 0
            self.total_trades = 0
            self.win_count = 0
            self.margin_used = 0
    
    def save_config(self):
        """Guarda estado actual"""
        config = {
            'current_capital': self.current_capital,
            'total_profit': self.total_profit,
            'total_trades': self.total_trades,
            'win_count': self.win_count,
            'margin_used': self.margin_used,
            'last_update': datetime.now().isoformat(),
            'leverage_config': self.max_leverage,
            'buying_power': self.current_capital * self.max_leverage
        }
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def get_open_positions_count(self):
        """Cuenta posiciones abiertas"""
        if not os.path.exists('open_orders.json'):
            return 0
        
        with open('open_orders.json', 'r') as f:
            orders = json.load(f)
        
        return len(orders)
    
    def sync_with_kraken_balance(self, kraken_balance_usd):
        """
        🆕 Sincroniza el capital con el balance REAL de Kraken Margin Wallet
        """
        old_capital = self.current_capital
        self.current_capital = kraken_balance_usd
        
        print(f"\n💰 SINCRONIZACIÓN CON KRAKEN:")
        print(f"   Capital anterior: ${old_capital:.2f}")
        print(f"   Capital Kraken: ${kraken_balance_usd:.2f}")
        print(f"   Diferencia: ${kraken_balance_usd - old_capital:+.2f}")
        
        # Ajustar leverage dinámicamente según capital
        if kraken_balance_usd < 20:
            self.max_leverage = 3
            print(f"   ⚠️ Leverage reducido a 3x (banca < $20)")
        elif kraken_balance_usd < 50:
            self.max_leverage = 5
            print(f"   ℹ️ Leverage: 5x")
        else:
            self.max_leverage = 5
            print(f"   ℹ️ Leverage: 5x")
        
        self.save_config()
    
    def calculate_margin_requirements(self, entry_price, volume, leverage):
        """Calcula requerimientos de margen exactos de Kraken"""
        position_value = entry_price * volume
        
        # Margen inicial requerido
        margin_required = position_value / leverage
        
        # Margen de mantenimiento
        maintenance_margin = position_value / (leverage * 2)
        
        # Margen disponible
        margin_available = self.current_capital - self.margin_used
        
        return {
            'position_value': position_value,
            'margin_required': margin_required,
            'maintenance_margin': maintenance_margin,
            'margin_available': margin_available,
            'margin_after': margin_available - margin_required,
            'margin_usage_%': (margin_required / self.current_capital) * 100 if self.current_capital > 0 else 0,
            'leverage': leverage,
            'buying_power': margin_available * leverage
        }
    
    def calculate_liquidation_price(self, entry_price, stop_loss, leverage, side='buy'):
        """Calcula precio de liquidación según fórmula de Kraken"""
        
        # Maintenance margin rate de Kraken
        maintenance_rate = 1 / (leverage * 2)
        
        if side == 'buy':
            liquidation_price = entry_price * (1 - (1 - maintenance_rate))
            sl_distance_pct = ((entry_price - stop_loss) / entry_price) * 100
            liq_distance_pct = ((entry_price - liquidation_price) / entry_price) * 100
        else:
            liquidation_price = entry_price * (1 + (1 - maintenance_rate))
            sl_distance_pct = ((stop_loss - entry_price) / entry_price) * 100
            liq_distance_pct = ((liquidation_price - entry_price) / entry_price) * 100
        
        buffer = abs(liq_distance_pct - sl_distance_pct)
        
        # Con banca pequeña, necesitamos al menos 30% de buffer
        safe = buffer >= (self.liquidation_buffer * 100)
        
        return {
            'liquidation_price': round(liquidation_price, 2),
            'sl_distance_%': sl_distance_pct,
            'liquidation_distance_%': liq_distance_pct,
            'buffer_%': buffer,
            'safe': safe,
            'warning': '⚠️ SL muy cerca de liquidación' if not safe else '✅ Buffer seguro'
        }
    
    def calculate_position_size(self, entry_price, stop_loss, confidence, side='buy', use_leverage=True):
        """
        🆕 OPTIMIZADO PARA BANCA PEQUEÑA (10€)
        Calcula tamaño de posición seguro con leverage dinámico
        """
        
        result = {
            'valid': False,
            'volume': 0,
            'risk_amount': 0,
            'position_value': 0,
            'leverage': 1,
            'margin_required': 0,
            'liquidation_price': 0,
            'reason': ''
        }
        
        # 1. Verificar confianza mínima (70% con banca pequeña)
        if confidence < self.confidence_threshold:
            result['reason'] = f"Confianza {confidence:.1f}% < {self.confidence_threshold}%"
            return result
        
        # 2. Verificar máximo de posiciones (solo 1 con banca pequeña)
        open_positions = self.get_open_positions_count()
        if open_positions >= self.max_open_positions:
            result['reason'] = f"Máximo {self.max_open_positions} posición permitida"
            return result
        
        # 3. Calcular distancia al SL
        if side == 'buy':
            sl_distance = abs(entry_price - stop_loss)
        else:
            sl_distance = abs(stop_loss - entry_price)
        
        if sl_distance <= 0:
            result['reason'] = "Stop loss inválido"
            return result
        
        # 4. Determinar leverage dinámicamente (más conservador con banca pequeña)
        if use_leverage and self.max_leverage > 1:
            # Con banca pequeña, usar leverage más conservador
            if self.current_capital < 20:
                # Con menos de $20, usar máximo 3x
                confidence_factor = (confidence - 70) / 30  # 0 to 1
                base_leverage = 2 + (confidence_factor * 1)  # 2x to 3x
                leverage = min(round(base_leverage, 1), 3)
            else:
                # Con más de $20, hasta 5x
                confidence_factor = (confidence - 70) / 30
                base_leverage = 3 + (confidence_factor * 2)  # 3x to 5x
                leverage = min(round(base_leverage, 1), self.max_leverage)
        else:
            leverage = 1
        
        # 5. Calcular riesgo en USD (2% de la banca)
        risk_usd = self.current_capital * self.risk_per_trade
        
        # 6. Calcular volumen inicial
        volume = risk_usd / sl_distance
        
        # 7. Verificar liquidación ANTES de continuar
        liq_calc = self.calculate_liquidation_price(entry_price, stop_loss, leverage, side)
        
        if not liq_calc['safe']:
            # Reducir leverage automáticamente
            safe_leverage = max(1, leverage / 2)
            leverage = safe_leverage
            liq_calc = self.calculate_liquidation_price(entry_price, stop_loss, leverage, side)
            print(f"⚠️ Leverage reducido a {leverage}x por seguridad")
        
        # 8. Verificar margen disponible
        margin_calc = self.calculate_margin_requirements(entry_price, volume, leverage)
        
        if margin_calc['margin_required'] > margin_calc['margin_available'] * self.margin_usage_limit:
            # Ajustar volumen por margen
            max_margin_use = margin_calc['margin_available'] * self.margin_usage_limit
            max_position_value = max_margin_use * leverage
            volume = max_position_value / entry_price
            margin_calc = self.calculate_margin_requirements(entry_price, volume, leverage)
            print(f"⚠️ Volumen ajustado por límite de margen")
        
        position_value = entry_price * volume
        
        # 9. Verificar límite de posición (40% con banca pequeña)
        max_position_value = self.current_capital * self.max_position_size * leverage
        
        if position_value > max_position_value:
            volume = max_position_value / entry_price
            position_value = volume * entry_price
            margin_calc = self.calculate_margin_requirements(entry_price, volume, leverage)
            print(f"⚠️ Volumen ajustado por límite de posición ({self.max_position_size*100}%)")
        
        # 10. Ajustar por confianza (más conservador con banca pequeña)
        confidence_multiplier = 0.7 + (confidence / 100) * 0.5  # 0.7 a 1.2x
        volume *= confidence_multiplier
        position_value = volume * entry_price
        margin_calc = self.calculate_margin_requirements(entry_price, volume, leverage)
        
        # 11. 🆕 VERIFICACIÓN ESPECIAL PARA BANCA PEQUEÑA
        # Con menos de $20, asegurar que el tamaño mínimo sea razonable
        if self.current_capital < 20:
            min_position_value = 10  # Mínimo $10 de posición con leverage
            if position_value < min_position_value:
                volume = min_position_value / entry_price
                position_value = volume * entry_price
                margin_calc = self.calculate_margin_requirements(entry_price, volume, leverage)
                print(f"⚠️ Volumen ajustado al mínimo razonable (${min_position_value})")
        
        # 12. Validaciones finales
        if volume < 1:  # ADA tiene mínimo de 1
            result['reason'] = "Volumen menor al mínimo (1 ADA)"
            return result
        
        if margin_calc['margin_required'] > margin_calc['margin_available']:
            result['reason'] = f"Margen insuficiente (req: ${margin_calc['margin_required']:.2f}, disp: ${margin_calc['margin_available']:.2f})"
            return result
        
        # Con banca pequeña, dejar al menos 20% libre
        if margin_calc['margin_after'] < self.current_capital * 0.20:
            result['reason'] = "Dejaría menos del 20% de margen disponible"
            return result
        
        # 13. Recalcular liquidación final
        liq_calc = self.calculate_liquidation_price(entry_price, stop_loss, leverage, side)
        
        if not liq_calc['safe']:
            result['reason'] = f"SL muy cerca de liquidación (buffer: {liq_calc['buffer_%']:.1f}%)"
            return result
        
        # ✅ TODO OK
        result.update({
            'valid': True,
            'volume': round(volume, 0),  # Redondear a entero
            'risk_amount': risk_usd,
            'position_value': position_value,
            'leverage': leverage,
            'margin_required': margin_calc['margin_required'],
            'margin_available': margin_calc['margin_available'],
            'margin_after': margin_calc['margin_after'],
            'margin_usage_%': margin_calc['margin_usage_%'],
            'maintenance_margin': margin_calc['maintenance_margin'],
            'liquidation_price': liq_calc['liquidation_price'],
            'liquidation_distance_%': liq_calc['liquidation_distance_%'],
            'buffer_to_liquidation_%': liq_calc['buffer_%'],
            'capital_used_%': (position_value / (self.current_capital * leverage)) * 100,
            'confidence_multiplier': confidence_multiplier,
            'exposure_multiplier': leverage,
            'buying_power_used': margin_calc['margin_required'],
            'reason': f'Validado OK - Leverage {leverage}x'
        })
        
        return result
    
    def validate_trade(self, entry_price, take_profit, stop_loss, side='buy'):
        """Valida si el trade cumple con el Risk/Reward mínimo"""
        
        if side == 'buy':
            reward = take_profit - entry_price
            risk = entry_price - stop_loss
        else:
            reward = entry_price - take_profit
            risk = stop_loss - entry_price
        
        if risk <= 0:
            return {'valid': False, 'rr_ratio': 0, 'reason': 'Riesgo inválido'}
        
        rr_ratio = reward / risk
        
        if rr_ratio < self.min_rr_ratio:
            return {
                'valid': False,
                'rr_ratio': rr_ratio,
                'reason': f'R/R {rr_ratio:.2f} < {self.min_rr_ratio:.2f}'
            }
        
        return {
            'valid': True,
            'rr_ratio': rr_ratio,
            'risk': risk,
            'reward': reward,
            'reason': 'Trade válido'
        }
    
    def update_after_trade(self, pnl_usd, margin_released=0):
        """Actualiza capital después de un trade"""
        self.current_capital += pnl_usd
        self.total_profit += pnl_usd
        self.total_trades += 1
        self.margin_used = max(0, self.margin_used - margin_released)
        
        if pnl_usd > 0:
            self.win_count += 1
        
        self.save_config()
        
        print(f"\n{'='*70}")
        print(f"  💰 ACTUALIZACIÓN DE CAPITAL")
        print(f"{'='*70}")
        print(f"P&L Trade: ${pnl_usd:+.2f}")
        print(f"Capital Actual: ${self.current_capital:.2f}")
        print(f"Margen Liberado: ${margin_released:.2f}")
        print(f"Margen en Uso: ${self.margin_used:.2f}")
        print(f"Margen Disponible: ${self.current_capital - self.margin_used:.2f}")
        print(f"Poder de Compra: ${(self.current_capital - self.margin_used) * self.max_leverage:.2f}")
        print(f"Ganancia Total: ${self.total_profit:+.2f}")
        print(f"Win Rate: {(self.win_count/self.total_trades*100):.1f}%")
        print(f"{'='*70}\n")
    
    def reserve_margin(self, margin_amount):
        """Reserva margen para una posición abierta"""
        self.margin_used += margin_amount
        self.save_config()
        print(f"🔒 Margen reservado: ${margin_amount:.2f}")
        print(f"   Total en uso: ${self.margin_used:.2f}")
        print(f"   Disponible: ${self.current_capital - self.margin_used:.2f}")
    
    def get_stats(self):
        """Retorna estadísticas actuales"""
        win_rate = (self.win_count / self.total_trades * 100) if self.total_trades > 0 else 0
        margin_available = self.current_capital - self.margin_used
        
        return {
            'current_capital': self.current_capital,
            'total_profit': self.total_profit,
            'profit_%': ((self.current_capital - self.initial_capital) / self.initial_capital) * 100 if self.initial_capital > 0 else 0,
            'total_trades': self.total_trades,
            'win_count': self.win_count,
            'win_rate': win_rate,
            'open_positions': self.get_open_positions_count(),
            'margin_used': self.margin_used,
            'margin_available': margin_available,
            'margin_usage_%': (self.margin_used / self.current_capital * 100) if self.current_capital > 0 else 0,
            'max_leverage': self.max_leverage,
            'buying_power': margin_available * self.max_leverage,
            'effective_buying_power': margin_available * self.max_leverage - self.margin_used
        }
    
    def print_stats(self):
        """Muestra estadísticas en consola"""
        stats = self.get_stats()
        
        print(f"\n{'='*70}")
        print(f"  📊 ESTADÍSTICAS DE TRADING (LEVERAGE {self.max_leverage}X)")
        print(f"{'='*70}")
        print(f"💰 Capital Inicial:     ${self.initial_capital:.2f}")
        print(f"💵 Capital Actual:      ${stats['current_capital']:.2f}")
        print(f"📈 Ganancia Total:      ${stats['total_profit']:+.2f} ({stats['profit_%']:+.2f}%)")
        print(f"")
        print(f"📊 Trades Totales:      {stats['total_trades']}")
        print(f"✅ Trades Ganados:      {stats['win_count']}")
        print(f"📉 Win Rate:            {stats['win_rate']:.1f}%")
        print(f"")
        print(f"🔓 Posiciones Abiertas: {stats['open_positions']}/{self.max_open_positions}")
        print(f"💳 Margen Usado:        ${stats['margin_used']:.2f} ({stats['margin_usage_%']:.1f}%)")
        print(f"💰 Margen Disponible:   ${stats['margin_available']:.2f}")
        print(f"⚡ Poder de Compra:     ${stats['buying_power']:.2f} (leverage {self.max_leverage}x)")
        print(f"🎯 Poder Efectivo:      ${stats['effective_buying_power']:.2f}")
        print(f"{'='*70}\n")

# Función de utilidad
def get_risk_manager():
    """
    🆕 OPTIMIZADO PARA BANCA PEQUEÑA (10€)
    """
    return RiskManager(
        initial_capital=10,            # 🆕 Se sincroniza con Kraken automáticamente
        risk_per_trade=0.02,           # 2% riesgo por trade (0.20€)
        max_leverage=5,                # 🆕 5x max (seguro para banca pequeña)
        margin_usage_limit=0.6,        # Usar máximo 60% del margen
        max_open_positions=3,          # 🆕 Solo 1 posición a la vez
        min_rr_ratio=1.5,             # Mínimo R/R 1.5:1
        liquidation_buffer=0.30,       # 30% buffer antes de liquidación
        max_position_size=0.40,        # Máximo 40% por posición
        confidence_threshold=70        # Confianza mínima 70%
    )

if __name__ == "__main__":
    # Demo con 10€
    rm = get_risk_manager()
    rm.print_stats()
    
    print("\n" + "="*70)
    print("  🔥 EJEMPLO: Trade con 10€ y Leverage 3x")
    print("="*70)
    
    entry = 1.00
    tp = 1.03
    sl = 0.98
    confidence = 75
    
    # Validar trade
    trade_valid = rm.validate_trade(entry, tp, sl, 'buy')
    print(f"\n✅ Validación:")
    print(f"  R/R: {trade_valid.get('rr_ratio', 0):.2f}")
    print(f"  Válido: {trade_valid['valid']}")
    
    if trade_valid['valid']:
        # Con leverage
        position = rm.calculate_position_size(entry, sl, confidence, 'buy', use_leverage=True)
        
        print(f"\n🔥 Posición con Leverage {position.get('leverage', 0)}x:")
        print(f"  Volumen: {position['volume']} ADA")
        print(f"  Valor Posición: ${position['position_value']:.2f}")
        print(f"  Margen Requerido: ${position['margin_required']:.2f}")
        print(f"  Riesgo: ${position['risk_amount']:.2f} (2% de ${rm.current_capital:.2f})")
        print(f"  Precio Liquidación: ${position['liquidation_price']:.2f}")
        print(f"  Buffer: {position.get('buffer_to_liquidation_%', 0):.1f}%")
        print(f"  Estado: {position['reason']}")
