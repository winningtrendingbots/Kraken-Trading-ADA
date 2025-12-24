"""
GESTOR DE RIESGO Y CAPITAL - VERSIÓN MEJORADA

✅ Optimizado para banca pequeña (10€)
✅ Leverage ENTERO (2-5x) - Compatible con Kraken
✅ Protección contra comisiones del 100%
✅ Evita rollover fees (cierre antes de 4h)
✅ Cálculo de fees totales ANTES de operar
✅ Tamaño mínimo de posición: $15
"""

import json
import os
from datetime import datetime

class RiskManager:
    def __init__(self, 
                 initial_capital=10,              
                 risk_per_trade=0.02,             # 2% riesgo
                 max_leverage=5,                  
                 margin_usage_limit=0.6,          
                 max_open_positions=1,            
                 min_rr_ratio=1.5,               
                 liquidation_buffer=0.30,         
                 max_position_size=0.40,          
                 confidence_threshold=70,         # 70% mínimo
                 min_position_value_usd=15,       # 🆕 Mínimo $15 (evita comisión 100%)
                 max_position_time_hours=3.5,     # 🆕 Cerrar antes del rollover (4h)
                 min_profit_after_fees_usd=0.50): # 🆕 Ganancia mínima después de fees
        
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_leverage = max_leverage
        self.margin_usage_limit = margin_usage_limit
        self.max_open_positions = max_open_positions
        self.min_rr_ratio = min_rr_ratio
        self.liquidation_buffer = liquidation_buffer
        self.max_position_size = max_position_size
        self.confidence_threshold = confidence_threshold
        
        # 🆕 NUEVOS parámetros de protección
        self.min_position_value_usd = min_position_value_usd
        self.max_position_time_hours = max_position_time_hours
        self.min_profit_after_fees_usd = min_profit_after_fees_usd
        
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
        
        try:
            with open('open_orders.json', 'r') as f:
                orders = json.load(f)
            return len(orders)
        except:
            return 0
    
    def sync_with_kraken_balance(self, kraken_balance_usd):
        """Sincroniza el capital con el balance REAL de Kraken"""
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
    
    def calculate_total_fees(self, entry_price, volume, time_hours=1):
        """
        🆕 Calcula TODOS los fees ANTES de abrir posición
        
        Returns:
            dict con breakdown de fees totales
        """
        position_value = entry_price * volume
        
        # 1. Trading fees (apertura + cierre)
        # Kraken: 0.26% taker fee (peor caso)
        open_fee = position_value * 0.0026
        close_fee = position_value * 0.0026
        trading_fees = open_fee + close_fee
        
        # 2. Rollover fees (si > 4 horas)
        rollover_fees = 0
        if time_hours > 4:
            rollover_periods = int(time_hours / 4)
            # Kraken cobra ~0.02% cada 4 horas en margin
            rollover_fees = position_value * 0.0002 * rollover_periods
        
        # 3. Spread estimado (0.1-0.3%)
        spread_cost = position_value * 0.002  # 0.2% conservador
        
        # Total
        total_fees = trading_fees + rollover_fees + spread_cost
        
        return {
            'position_value': position_value,
            'open_fee': open_fee,
            'close_fee': close_fee,
            'trading_fees': trading_fees,
            'rollover_fees': rollover_fees,
            'spread_cost': spread_cost,
            'total_fees': total_fees,
            'total_fees_%': (total_fees / position_value) * 100,
            'min_profit_needed': total_fees + self.min_profit_after_fees_usd
        }
    
    def calculate_take_profit_with_fees(self, entry_price, stop_loss, side='buy'):
        """
        🆕 Calcula TP que cubra TODOS los fees + ganancia mínima
        """
        # Calcular fees para una posición tipo
        test_volume = 100  # Volumen de prueba
        fees_info = self.calculate_total_fees(entry_price, test_volume, 
                                              self.max_position_time_hours)
        
        # Fees totales como %
        total_fees_pct = fees_info['total_fees_%'] / 100
        
        # Distancia al SL como %
        if side == 'buy':
            sl_distance_pct = abs(entry_price - stop_loss) / entry_price
            # TP debe ser: SL + fees + ganancia mínima
            min_profit_pct = self.min_profit_after_fees_usd / (entry_price * test_volume)
            tp_distance_pct = sl_distance_pct + total_fees_pct + min_profit_pct
            tp_price = entry_price * (1 + tp_distance_pct * self.min_rr_ratio)
        else:
            sl_distance_pct = abs(stop_loss - entry_price) / entry_price
            min_profit_pct = self.min_profit_after_fees_usd / (entry_price * test_volume)
            tp_distance_pct = sl_distance_pct + total_fees_pct + min_profit_pct
            tp_price = entry_price * (1 - tp_distance_pct * self.min_rr_ratio)
        
        return {
            'tp_price': round(tp_price, 4),
            'tp_distance_%': tp_distance_pct * self.min_rr_ratio * 100,
            'fees_included_%': total_fees_pct * 100,
            'min_profit_usd': self.min_profit_after_fees_usd
        }
    
    def calculate_margin_requirements(self, entry_price, volume, leverage):
        """Calcula requerimientos de margen exactos de Kraken"""
        position_value = entry_price * volume
        
        margin_required = position_value / leverage
        maintenance_margin = position_value / (leverage * 2)
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
        safe = buffer >= (self.liquidation_buffer * 100)
        
        return {
            'liquidation_price': round(liquidation_price, 4),
            'sl_distance_%': sl_distance_pct,
            'liquidation_distance_%': liq_distance_pct,
            'buffer_%': buffer,
            'safe': safe,
            'warning': '⚠️ SL muy cerca de liquidación' if not safe else '✅ Buffer seguro'
        }
    
    def calculate_position_size(self, entry_price, stop_loss, confidence, side='buy', use_leverage=True):
        """
        🔥 FIXED: Leverage siempre ENTERO (2, 3, 4, 5)
        Protección contra comisiones del 100%
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
            result['reason'] = f"Máximo {self.max_open_positions} posición permitida (actuales: {open_positions})"
            return result
        
        # 3. Calcular distancia al SL
        if side == 'buy':
            sl_distance = abs(entry_price - stop_loss)
        else:
            sl_distance = abs(stop_loss - entry_price)
        
        if sl_distance <= 0:
            result['reason'] = "Stop loss inválido"
            return result
        
        # 🔥 4. Determinar leverage ENTERO
        if use_leverage and self.max_leverage > 1:
            if self.current_capital < 20:
                # Banca pequeña: solo 2x o 3x
                if confidence >= 85:
                    leverage = 3
                else:
                    leverage = 2
            else:
                # Banca normal: 2-5x según confianza
                if confidence >= 90:
                    leverage = 5
                elif confidence >= 85:
                    leverage = 4
                elif confidence >= 80:
                    leverage = 3
                else:
                    leverage = 2
            
            # Limitar al máximo configurado
            leverage = min(leverage, self.max_leverage)
        else:
            leverage = 1
        
        # 5. Calcular riesgo en USD (2% de la banca)
        risk_usd = self.current_capital * self.risk_per_trade
        
        # 6. Calcular volumen inicial
        volume = risk_usd / sl_distance
        position_value = entry_price * volume
        
        # 🆕 7. VERIFICACIÓN CRÍTICA: Tamaño mínimo de posición
        if position_value < self.min_position_value_usd:
            # Ajustar volumen al mínimo
            volume = self.min_position_value_usd / entry_price
            position_value = volume * entry_price
            print(f"⚠️ Posición ajustada al mínimo: ${position_value:.2f}")
        
        # 8. Calcular fees totales
        fees_info = self.calculate_total_fees(entry_price, volume, 
                                              self.max_position_time_hours)
        
        print(f"\n💰 ANÁLISIS DE FEES:")
        print(f"   Valor posición: ${fees_info['position_value']:.2f}")
        print(f"   Trading fees: ${fees_info['trading_fees']:.2f} ({fees_info['total_fees_%']:.2f}%)")
        print(f"   Spread estimado: ${fees_info['spread_cost']:.2f}")
        print(f"   Total fees: ${fees_info['total_fees']:.2f}")
        print(f"   Ganancia mínima necesaria: ${fees_info['min_profit_needed']:.2f}")
        
        # 9. Verificar liquidación
        liq_calc = self.calculate_liquidation_price(entry_price, stop_loss, leverage, side)
        
        if not liq_calc['safe']:
            # Reducir leverage al siguiente entero inferior
            safe_leverage = max(2, leverage - 1)
            leverage = safe_leverage
            liq_calc = self.calculate_liquidation_price(entry_price, stop_loss, leverage, side)
            print(f"⚠️ Leverage reducido a {leverage}x por seguridad")
        
        # 10. Verificar margen disponible
        margin_calc = self.calculate_margin_requirements(entry_price, volume, leverage)
        
        if margin_calc['margin_required'] > margin_calc['margin_available'] * self.margin_usage_limit:
            max_margin_use = margin_calc['margin_available'] * self.margin_usage_limit
            max_position_value = max_margin_use * leverage
            volume = max_position_value / entry_price
            position_value = volume * entry_price
            margin_calc = self.calculate_margin_requirements(entry_price, volume, leverage)
            print(f"⚠️ Volumen ajustado por límite de margen")
        
        # 11. Ajustar por confianza
        confidence_multiplier = 0.7 + (confidence / 100) * 0.5
        volume *= confidence_multiplier
        position_value = volume * entry_price
        margin_calc = self.calculate_margin_requirements(entry_price, volume, leverage)
        
        # 🆕 12. VERIFICACIÓN FINAL: Posición no puede ser < $15
        if position_value < self.min_position_value_usd:
            result['reason'] = f"Posición final demasiado pequeña (${position_value:.2f} < ${self.min_position_value_usd})"
            return result
        
        # 13. Validaciones finales
        if volume < 1:
            result['reason'] = "Volumen menor al mínimo (1 ADA)"
            return result
        
        if margin_calc['margin_required'] > margin_calc['margin_available']:
            result['reason'] = f"Margen insuficiente (req: ${margin_calc['margin_required']:.2f}, disp: ${margin_calc['margin_available']:.2f})"
            return result
        
        if margin_calc['margin_after'] < self.current_capital * 0.20:
            result['reason'] = "Dejaría menos del 20% de margen disponible"
            return result
        
        # ✅ TODO OK
        result.update({
            'valid': True,
            'volume': round(volume, 0),
            'risk_amount': risk_usd,
            'position_value': position_value,
            'leverage': int(leverage),  # 🔥 Asegurar que es entero
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
            'total_fees_usd': fees_info['total_fees'],
            'total_fees_%': fees_info['total_fees_%'],
            'min_profit_needed_usd': fees_info['min_profit_needed'],
            'reason': f'Validado OK - Leverage {int(leverage)}x - Fees ${fees_info["total_fees"]:.2f}'
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
        print(f"Ganancia Total: ${self.total_profit:+.2f}")
        print(f"Win Rate: {(self.win_count/self.total_trades*100):.1f}%")
        print(f"{'='*70}\n")
    
    def reserve_margin(self, margin_amount):
        """Reserva margen para una posición abierta"""
        self.margin_used += margin_amount
        self.save_config()
        print(f"🔒 Margen reservado: ${margin_amount:.2f}")
    
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
            'buying_power': margin_available * self.max_leverage
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
        print(f"📍 Posiciones Abiertas: {stats['open_positions']}/{self.max_open_positions}")
        print(f"💳 Margen Usado:        ${stats['margin_used']:.2f}")
        print(f"💰 Margen Disponible:   ${stats['margin_available']:.2f}")
        print(f"⚡ Poder de Compra:     ${stats['buying_power']:.2f}")
        print(f"{'='*70}\n")

def get_risk_manager():
    """Retorna instancia configurada del Risk Manager"""
    return RiskManager(
        initial_capital=10,
        risk_per_trade=0.02,
        max_leverage=5,
        margin_usage_limit=0.6,
        max_open_positions=1,
        min_rr_ratio=1.5,
        liquidation_buffer=0.30,
        max_position_size=0.40,
        confidence_threshold=70,
        min_position_value_usd=15,        # 🆕 Mínimo $15
        max_position_time_hours=3.5,      # 🆕 Cerrar antes de rollover
        min_profit_after_fees_usd=0.50    # 🆕 $0.50 mínimo después de fees
    )

if __name__ == "__main__":
    rm = get_risk_manager()
    rm.print_stats()
    
    print("\n" + "="*70)
    print("  🔥 EJEMPLO: Trade Seguro con Fees Incluidos")
    print("="*70)
    
    entry = 1.00
    sl = 0.98
    confidence = 75
    
    # Calcular TP que cubra fees
    tp_info = rm.calculate_take_profit_with_fees(entry, sl, 'buy')
    tp = tp_info['tp_price']
    
    print(f"\n📊 Setup del Trade:")
    print(f"   Entry: ${entry:.4f}")
    print(f"   SL: ${sl:.4f}")
    print(f"   TP: ${tp:.4f} (incluye fees + ganancia mínima)")
    print(f"   Distancia TP: {tp_info['tp_distance_%']:.2f}%")
    
    # Validar
    trade_valid = rm.validate_trade(entry, tp, sl, 'buy')
    print(f"\n✅ Validación:")
    print(f"   R/R: {trade_valid.get('rr_ratio', 0):.2f}")
    print(f"   Válido: {trade_valid['valid']}")
    
    if trade_valid['valid']:
        position = rm.calculate_position_size(entry, sl, confidence, 'buy', use_leverage=True)
        
        if position['valid']:
            print(f"\n🔥 Posición Calculada:")
            print(f"   Volumen: {position['volume']} ADA")
            print(f"   Valor: ${position['position_value']:.2f}")
            print(f"   Leverage: {position['leverage']}x")
            print(f"   Margen: ${position['margin_required']:.2f}")
            print(f"   Fees Totales: ${position['total_fees_usd']:.2f} ({position['total_fees_%']:.2f}%)")
            print(f"   Ganancia Mínima: ${position['min_profit_needed_usd']:.2f}")
            print(f"   Liquidación: ${position['liquidation_price']:.4f}")
        else:
            print(f"\n❌ Posición rechazada: {position['reason']}")
