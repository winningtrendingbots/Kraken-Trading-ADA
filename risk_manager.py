"""
GESTOR DE RIESGO Y CAPITAL CON MARGEN - CONFIGURADO PARA 10X LEVERAGE

Características:
- Leverage máximo: 10x
- Gestión automática de margen de Kraken
- Cálculo de precio de liquidación
- Buffers de seguridad
- Integración con balance real de Kraken
"""

import json
import os
from datetime import datetime

class RiskManager:
    def __init__(self, 
                 initial_capital=30,           # Capital inicial en USD
                 risk_per_trade=0.015,           # 1.5% de riesgo por trade (reducido por leverage alto)
                 max_leverage=10,                # ⚡ LEVERAGE 10X ACTIVADO
                 margin_usage_limit=0.5,         # Usar máximo 50% del margen disponible
                 max_open_positions=3,           # Máximo 3 posiciones simultáneas
                 min_rr_ratio=1.5,              # Mínimo Risk/Reward 1:1.5
                 liquidation_buffer=0.25,        # 25% de buffer antes de liquidación (importante con 10x)
                 max_position_size=0.25,         # Máximo 25% del capital por posición (reducido)
                 confidence_threshold=65):       # Confianza mínima 65% (subido por leverage alto)
        
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
        Sincroniza el capital con el balance real de Kraken
        Llama esto periódicamente para mantener datos reales
        """
        old_capital = self.current_capital
        self.current_capital = kraken_balance_usd
        
        print(f"\n💰 SINCRONIZACIÓN CON KRAKEN:")
        print(f"   Capital anterior: ${old_capital:.2f}")
        print(f"   Capital Kraken: ${kraken_balance_usd:.2f}")
        print(f"   Diferencia: ${kraken_balance_usd - old_capital:+.2f}")
        
        self.save_config()
    
    def calculate_margin_requirements(self, entry_price, volume, leverage):
        """
        Calcula requerimientos de margen exactos de Kraken
        
        Kraken formula:
        - Margin Required = Position Value / Leverage
        - Maintenance Margin = Position Value / (Leverage * 2)
        """
        position_value = entry_price * volume
        
        # Margen inicial requerido
        margin_required = position_value / leverage
        
        # Margen de mantenimiento (Kraken usa leverage * 2)
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
        """
        Calcula precio de liquidación según fórmula de Kraken
        
        Para posiciones con margen:
        - LONG: Liquidación cuando equity = maintenance margin
        - SHORT: Similar pero inverso
        
        Fórmula Kraken aproximada:
        Liquidation Price = Entry * (1 ± (1 - Maintenance Margin Rate) / Leverage)
        Donde Maintenance Margin Rate ≈ 1 / (Leverage * 2)
        """
        
        # Maintenance margin rate de Kraken
        maintenance_rate = 1 / (leverage * 2)
        
        if side == 'buy':
            # Para LONG: precio baja hasta liquidación
            liquidation_price = entry_price * (1 - (1 - maintenance_rate))
            sl_distance_pct = ((entry_price - stop_loss) / entry_price) * 100
            liq_distance_pct = ((entry_price - liquidation_price) / entry_price) * 100
        else:
            # Para SHORT: precio sube hasta liquidación
            liquidation_price = entry_price * (1 + (1 - maintenance_rate))
            sl_distance_pct = ((stop_loss - entry_price) / entry_price) * 100
            liq_distance_pct = ((liquidation_price - entry_price) / entry_price) * 100
        
        buffer = abs(liq_distance_pct - sl_distance_pct)
        
        # Con 10x leverage, necesitamos al menos 25% de buffer
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
        Calcula tamaño de posición óptimo con leverage 10x
        Más conservador debido al alto apalancamiento
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
        
        # 1. Verificar confianza mínima (subida a 65% con 10x)
        if confidence < self.confidence_threshold:
            result['reason'] = f"Confianza {confidence:.1f}% < {self.confidence_threshold}%"
            return result
        
        # 2. Verificar máximo de posiciones
        open_positions = self.get_open_positions_count()
        if open_positions >= self.max_open_positions:
            result['reason'] = f"Máximo posiciones ({self.max_open_positions}) alcanzado"
            return result
        
        # 3. Calcular distancia al SL
        if side == 'buy':
            sl_distance = abs(entry_price - stop_loss)
        else:
            sl_distance = abs(stop_loss - entry_price)
        
        if sl_distance <= 0:
            result['reason'] = "Stop loss inválido"
            return result
        
        # 4. Determinar leverage (con 10x, usar dinámicamente según confianza)
        if use_leverage and self.max_leverage > 1:
            # Con confianza alta, usar más leverage
            # Con 65% confianza = 5x, con 100% = 10x
            confidence_factor = (confidence - 65) / 35  # 0 to 1
            base_leverage = 5 + (confidence_factor * 5)  # 5x to 10x
            leverage = min(round(base_leverage, 1), self.max_leverage)
        else:
            leverage = 1
        
        # 5. Calcular riesgo en USD (reducido con leverage alto)
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
        
        # 9. Verificar límite de posición (25% con 10x)
        max_position_value = self.current_capital * self.max_position_size * leverage
        
        if position_value > max_position_value:
            volume = max_position_value / entry_price
            position_value = volume * entry_price
            margin_calc = self.calculate_margin_requirements(entry_price, volume, leverage)
            print(f"⚠️ Volumen ajustado por límite de posición ({self.max_position_size*100}%)")
        
        # 10. Ajustar por confianza (más conservador)
        confidence_multiplier = 0.6 + (confidence / 100) * 0.7  # 0.6 a 1.3x
        volume *= confidence_multiplier
        position_value = volume * entry_price
        margin_calc = self.calculate_margin_requirements(entry_price, volume, leverage)
        
        # 11. Validaciones finales
        if volume < 0.001:
            result['reason'] = "Volumen menor al mínimo (0.001 ADA)"
            return result
        
        if margin_calc['margin_required'] > margin_calc['margin_available']:
            result['reason'] = f"Margen insuficiente (req: ${margin_calc['margin_required']:.2f}, disp: ${margin_calc['margin_available']:.2f})"
            return result
        
        if margin_calc['margin_after'] < self.current_capital * 0.15:  # Dejar 15% libre con 10x
            result['reason'] = "Dejaría menos del 15% de margen disponible"
            return result
        
        # 12. Recalcular liquidación final
        liq_calc = self.calculate_liquidation_price(entry_price, stop_loss, leverage, side)
        
        if not liq_calc['safe']:
            result['reason'] = f"SL muy cerca de liquidación (buffer: {liq_calc['buffer_%']:.1f}%)"
            return result
        
        # ✅ TODO OK
        result.update({
            'valid': True,
            'volume': round(volume, 4),
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
            'reason': 'Validado OK - Leverage 10x activo'
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
            'profit_%': ((self.current_capital - self.initial_capital) / self.initial_capital) * 100,
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
    """Retorna instancia del Risk Manager configurada para 10x leverage"""
    return RiskManager(
        initial_capital=1000,          # Tu capital en Kraken
        risk_per_trade=0.015,          # 1.5% riesgo (reducido por leverage alto)
        max_leverage=10,               # ⚡ 10X LEVERAGE ACTIVADO
        margin_usage_limit=0.5,        # Usar máximo 50% del margen
        max_open_positions=3,          # Máximo 3 posiciones
        min_rr_ratio=1.5,             # Mínimo R/R 1.5:1
        liquidation_buffer=0.25,       # 25% buffer antes de liquidación
        max_position_size=0.25,        # Máximo 25% por posición (reducido)
        confidence_threshold=65        # Confianza mínima 65% (subido)
    )

if __name__ == "__main__":
    # Demo con 10x leverage
    rm = get_risk_manager()
    rm.print_stats()
    
    print("\n" + "="*70)
    print("  🔥 EJEMPLO: Trade con Leverage 10x")
    print("="*70)
    
    entry = 3500
    tp = 3700
    sl = 3400
    confidence = 80
    
    # Validar trade
    trade_valid = rm.validate_trade(entry, tp, sl, 'buy')
    print(f"\n✔ Validación:")
    print(f"  R/R: {trade_valid.get('rr_ratio', 0):.2f}")
    print(f"  Válido: {trade_valid['valid']}")
    
    if trade_valid['valid']:
        # Con leverage 10x
        position = rm.calculate_position_size(entry, sl, confidence, 'buy', use_leverage=True)
        
        print(f"\n🔥 Posición CON Leverage 10x:")
        print(f"  Volumen: {position['volume']} ADA")
        print(f"  Valor Posición: ${position['position_value']:.2f}")
        print(f"  Leverage Usado: {position['leverage']}x")
        print(f"  Margen Requerido: ${position['margin_required']:.2f}")
        print(f"  Margen Disponible: ${position['margin_available']:.2f}")
        print(f"  Uso de Margen: {position.get('margin_usage_%', 0):.1f}%")
        print(f"  Precio Liquidación: ${position['liquidation_price']:.2f}")
        print(f"  Buffer a Liquidación: {position.get('buffer_to_liquidation_%', 0):.1f}%")
        print(f"  Riesgo Real: ${position['risk_amount']:.2f}")
        print(f"  Poder de Compra Usado: ${position['buying_power_used']:.2f}")
        print(f"  Estado: {position['reason']}")
