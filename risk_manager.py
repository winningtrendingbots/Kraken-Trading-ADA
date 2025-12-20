"""
GESTOR DE RIESGO Y CAPITAL CON MARGEN

Calcula el tamaño de posición considerando:
- Capital disponible
- Apalancamiento (leverage)
- Margen requerido y disponible
- Distancia a liquidación
- % de riesgo por trade
- ATR (volatilidad)
- Risk/Reward ratio
"""

import json
import os
from datetime import datetime

class RiskManager:
    def __init__(self, 
                 initial_capital=30,           # Capital inicial en USD
                 risk_per_trade=0.02,            # 2% de riesgo por trade
                 max_leverage=10,                 # Apalancamiento máximo (1x = sin margen)
                 margin_usage_limit=0.5,         # Usar máximo 50% del margen disponible
                 max_open_positions=10,           # Máximo 3 posiciones simultáneas
                 min_rr_ratio=1.5,              # Mínimo Risk/Reward 1:1.5
                 liquidation_buffer=0.2,         # 20% de buffer antes de liquidación
                 max_position_size=0.3,          # Máximo 30% del capital por posición
                 confidence_threshold=60):       # Confianza mínima 60%
        
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
            'last_update': datetime.now().isoformat()
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
    
    def calculate_margin_requirements(self, entry_price, volume, leverage):
        """
        Calcula requerimientos de margen
        
        Args:
            entry_price: Precio de entrada
            volume: Volumen en ETH
            leverage: Apalancamiento (1x, 2x, 3x, etc)
        
        Returns:
            dict con cálculos de margen
        """
        position_value = entry_price * volume
        
        # Margen requerido = Valor posición / Leverage
        margin_required = position_value / leverage
        
        # Margen disponible
        margin_available = self.current_capital - self.margin_used
        
        return {
            'position_value': position_value,
            'margin_required': margin_required,
            'margin_available': margin_available,
            'margin_after': margin_available - margin_required,
            'margin_usage_%': (margin_required / self.current_capital) * 100 if self.current_capital > 0 else 0,
            'leverage': leverage
        }
    
    def calculate_liquidation_price(self, entry_price, stop_loss, leverage, side='buy'):
        """
        Calcula precio de liquidación con margen
        
        En Kraken, liquidación ocurre cuando:
        Equity = Margen Requerido (aprox)
        
        Args:
            entry_price: Precio de entrada
            stop_loss: Stop loss
            leverage: Apalancamiento
            side: 'buy' o 'sell'
        
        Returns:
            dict con precio de liquidación y distancias
        """
        
        # Fórmula simplificada de liquidación:
        # Para LONG: Liquidación ≈ Entry * (1 - 1/Leverage * 0.9)
        # Para SHORT: Liquidación ≈ Entry * (1 + 1/Leverage * 0.9)
        
        liquidation_factor = (1 / leverage) * 0.9  # 90% del margen (10% buffer mínimo)
        
        if side == 'buy':
            liquidation_price = entry_price * (1 - liquidation_factor)
            sl_distance_pct = ((entry_price - stop_loss) / entry_price) * 100
            liq_distance_pct = ((entry_price - liquidation_price) / entry_price) * 100
        else:
            liquidation_price = entry_price * (1 + liquidation_factor)
            sl_distance_pct = ((stop_loss - entry_price) / entry_price) * 100
            liq_distance_pct = ((liquidation_price - entry_price) / entry_price) * 100
        
        return {
            'liquidation_price': round(liquidation_price, 2),
            'sl_distance_%': sl_distance_pct,
            'liquidation_distance_%': liq_distance_pct,
            'buffer_%': abs(liq_distance_pct - sl_distance_pct),
            'safe': abs(liq_distance_pct) > abs(sl_distance_pct) * 1.2  # 20% buffer mínimo
        }
    
    def calculate_position_size(self, entry_price, stop_loss, confidence, side='buy', use_leverage=True):
        """
        Calcula tamaño de posición con gestión de margen
        
        Args:
            entry_price: Precio de entrada
            stop_loss: Precio de stop loss
            confidence: Nivel de confianza (0-100)
            side: 'buy' o 'sell'
            use_leverage: Si usar apalancamiento
        
        Returns:
            dict con volume, risk_amount, margen y validación
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
        
        # 1. Verificar confianza mínima
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
        
        # 4. Determinar apalancamiento óptimo
        if use_leverage and self.max_leverage > 1:
            # Empezar con leverage moderado basado en confianza
            base_leverage = 1 + (confidence / 100) * (self.max_leverage - 1)
            leverage = min(round(base_leverage, 1), self.max_leverage)
        else:
            leverage = 1
        
        # 5. Calcular riesgo en USD (sin apalancamiento)
        risk_usd = self.current_capital * self.risk_per_trade
        
        # 6. Calcular volumen inicial
        # Volume = Risk_USD / SL_Distance
        volume = risk_usd / sl_distance
        
        # 7. Verificar liquidación
        liq_calc = self.calculate_liquidation_price(entry_price, stop_loss, leverage, side)
        
        if not liq_calc['safe']:
            # Reducir leverage si SL está muy cerca de liquidación
            max_safe_leverage = max(1, leverage / 2)
            leverage = max_safe_leverage
            liq_calc = self.calculate_liquidation_price(entry_price, stop_loss, leverage, side)
            print(f"⚠️ Leverage reducido a {leverage}x por seguridad")
        
        # 8. Verificar margen disponible
        margin_calc = self.calculate_margin_requirements(entry_price, volume, leverage)
        
        if margin_calc['margin_required'] > margin_calc['margin_available'] * self.margin_usage_limit:
            # Ajustar volumen por margen disponible
            max_margin_use = margin_calc['margin_available'] * self.margin_usage_limit
            max_position_value = max_margin_use * leverage
            volume = max_position_value / entry_price
            margin_calc = self.calculate_margin_requirements(entry_price, volume, leverage)
            print(f"⚠️ Volumen ajustado por límite de margen")
        
        position_value = entry_price * volume
        
        # 9. Verificar límite de posición
        max_position_value = self.current_capital * self.max_position_size * leverage
        
        if position_value > max_position_value:
            volume = max_position_value / entry_price
            position_value = volume * entry_price
            margin_calc = self.calculate_margin_requirements(entry_price, volume, leverage)
            print(f"⚠️ Volumen ajustado por límite de posición ({self.max_position_size*100}%)")
        
        # 10. Ajustar por confianza (escalar volumen)
        confidence_multiplier = 0.7 + (confidence / 100) * 0.6  # 0.7 a 1.3x
        volume *= confidence_multiplier
        position_value = volume * entry_price
        margin_calc = self.calculate_margin_requirements(entry_price, volume, leverage)
        
        # 11. Validaciones finales
        if volume < 0.001:
            result['reason'] = "Volumen menor al mínimo (0.001 ETH)"
            return result
        
        if margin_calc['margin_required'] > margin_calc['margin_available']:
            result['reason'] = f"Margen insuficiente (req: ${margin_calc['margin_required']:.2f}, disp: ${margin_calc['margin_available']:.2f})"
            return result
        
        if margin_calc['margin_after'] < self.current_capital * 0.1:
            result['reason'] = "Dejaría menos del 10% de margen disponible"
            return result
        
        # 12. Recalcular liquidación final
        liq_calc = self.calculate_liquidation_price(entry_price, stop_loss, leverage, side)
        
        if not liq_calc['safe']:
            result['reason'] = f"SL muy cerca de liquidación (buffer: {liq_calc['buffer_%']:.1f}%)"
            return result
        
        # Todo OK
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
            'liquidation_price': liq_calc['liquidation_price'],
            'liquidation_distance_%': liq_calc['liquidation_distance_%'],
            'buffer_to_liquidation_%': liq_calc['buffer_%'],
            'capital_used_%': (position_value / (self.current_capital * leverage)) * 100,
            'confidence_multiplier': confidence_multiplier,
            'exposure_multiplier': leverage,  # Cuánto multiplicas tu exposición
            'reason': 'Validado OK'
        })
        
        return result
    
    def validate_trade(self, entry_price, take_profit, stop_loss, side='buy'):
        """
        Valida si el trade cumple con el Risk/Reward mínimo
        """
        
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
        """
        Actualiza capital después de un trade
        
        Args:
            pnl_usd: P&L del trade
            margin_released: Margen liberado al cerrar la posición
        """
        self.current_capital += pnl_usd
        self.total_profit += pnl_usd
        self.total_trades += 1
        self.margin_used = max(0, self.margin_used - margin_released)
        
        if pnl_usd > 0:
            self.win_count += 1
        
        self.save_config()
        
        print(f"\n{'='*70}")
        print(f"  ACTUALIZACIÓN DE CAPITAL")
        print(f"{'='*70}")
        print(f"P&L Trade: ${pnl_usd:+.2f}")
        print(f"Capital Actual: ${self.current_capital:.2f}")
        print(f"Margen Liberado: ${margin_released:.2f}")
        print(f"Margen en Uso: ${self.margin_used:.2f}")
        print(f"Margen Disponible: ${self.current_capital - self.margin_used:.2f}")
        print(f"Ganancia Total: ${self.total_profit:+.2f}")
        print(f"Win Rate: {(self.win_count/self.total_trades*100):.1f}%")
        print(f"{'='*70}\n")
    
    def reserve_margin(self, margin_amount):
        """Reserva margen para una posición abierta"""
        self.margin_used += margin_amount
        self.save_config()
    
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
            'buying_power': margin_available * self.max_leverage
        }
    
    def print_stats(self):
        """Muestra estadísticas en consola"""
        stats = self.get_stats()
        
        print(f"\n{'='*70}")
        print(f"  ESTADÍSTICAS DE TRADING")
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
        print(f"{'='*70}\n")

# Función de utilidad
def get_risk_manager():
    """Retorna instancia del Risk Manager con margen"""
    return RiskManager(
        initial_capital=1000,          # Tu capital
        risk_per_trade=0.02,           # 2% riesgo
        max_leverage=3,                # Máximo 3x (ajusta según tu tolerancia)
        margin_usage_limit=0.6,        # Usar máximo 60% del margen
        max_open_positions=3,
        min_rr_ratio=1.5,
        liquidation_buffer=0.2,        # 20% buffer antes de liquidación
        max_position_size=0.3,
        confidence_threshold=60
    )

if __name__ == "__main__":
    # Ejemplo con margen
    rm = get_risk_manager()
    rm.print_stats()
    
    print("\n" + "="*70)
    print("  EJEMPLO: Trade con Margen")
    print("="*70)
    
    entry = 3500
    tp = 3700
    sl = 3400
    confidence = 80
    
    # Validar trade
    trade_valid = rm.validate_trade(entry, tp, sl, 'buy')
    print(f"\n✓ Validación:")
    print(f"  R/R: {trade_valid.get('rr_ratio', 0):.2f}")
    print(f"  Válido: {trade_valid['valid']}")
    
    if trade_valid['valid']:
        # Con margen
        position = rm.calculate_position_size(entry, sl, confidence, 'buy', use_leverage=True)
        
        print(f"\n💼 Posición CON Margen:")
        print(f"  Volumen: {position['volume']} ETH")
        print(f"  Valor Posición: ${position['position_value']:.2f}")
        print(f"  Leverage: {position['leverage']}x")
        print(f"  Margen Requerido: ${position['margin_required']:.2f}")
        print(f"  Margen Disponible: ${position['margin_available']:.2f}")
        print(f"  Uso de Margen: {position.get('margin_usage_%', 0):.1f}%")
        print(f"  Precio Liquidación: ${position['liquidation_price']:.2f}")
        print(f"  Buffer a Liquidación: {position.get('buffer_to_liquidation_%', 0):.1f}%")
        print(f"  Riesgo Real: ${position['risk_amount']:.2f}")
        print(f"  Estado: {position['reason']}")
        
        # Sin margen para comparar
        position_no_lev = rm.calculate_position_size(entry, sl, confidence, 'buy', use_leverage=False)
        print(f"\n💼 Posición SIN Margen (comparación):")
        print(f"  Volumen: {position_no_lev['volume']} ETH")
        print(f"  Valor: ${position_no_lev['position_value']:.2f}")
