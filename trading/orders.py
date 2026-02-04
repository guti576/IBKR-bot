from config.settings import *

def place_buy_order(qty=1, stop_pct=0.02):

    order = MarketOrder('BUY', qty)
    trade = ib.placeOrder(contract, order)

    # esperar hasta que esté filled
    while not trade.isDone():
        ib.waitOnUpdate()

    if not trade.fills:
        raise RuntimeError("La orden no se ejecutó")

    fill_price = trade.fills[0].execution.price
    stop_price = round(fill_price * (1 - stop_pct), 2)

    stop = StopOrder('SELL', qty, stop_price)
    ib.placeOrder(contract, stop)

    log(f"Comprado a {fill_price:.2f} | Stop colocado en {stop_price:.2f}")

    return trade

def place_sell_order(qty=1):
    # Orden de venta
    order = MarketOrder('SELL', qty)
    trade = ib.placeOrder(contract, order)
    log(f"Orden de venta enviada: {trade}")


def has_open_position():
    """
    Devuelve True si hay alguna posición abierta en el símbolo indicado, False si no.
    """
    contract = Stock(symbol, 'SMART', 'USD')
    ib.qualifyContracts(contract)
    
    positions = ib.positions()  # lista de todas las posiciones abiertas
    for pos in positions:
        if pos.contract.conId == contract.conId and pos.position != 0:
            return True
    return False


def close_all_positions():
    """
    Cierra todas las posiciones abiertas del símbolo indicado.
    
    ib: instancia de IB() conectada
    symbol: ticker del activo (ej: 'AAPL', 'GOOG')
    
    Devuelve:
        Listado de dicts con resumen de cierre:
        [{'action': 'SELL', 'qty': 10, 'symbol': 'GOOG', 'status': 'filled'}, ...]
    """

    summary = []
    
    # Obtener posiciones abiertas de ese símbolo
    positions = [p for p in ib.positions() if p.contract.symbol == symbol]

    if not positions:
        print(f"No hay posiciones abiertas de {symbol}")
        return summary

    for pos in positions:
        qty = pos.position
        if qty == 0:
            continue

        # Determinar acción
        action = 'SELL' if qty > 0 else 'BUY'
        order_qty = abs(qty)

        # Crear contrato usando SMART routing para evitar error 10311
        contract = Stock(symbol, 'SMART', 'USD')
        order = MarketOrder(action, order_qty)

        try:
            trade = ib.placeOrder(contract, order)
            log(f"[{symbol}] Orden enviada: {action} {order_qty}")

            # Esperar a que la orden se complete
            while not trade.isDone():
                ib.sleep(0.5)

            log
            (f"[{symbol}] Posición cerrada: {action} {order_qty}")
            summary.append({'symbol': symbol, 'action': action, 'qty': order_qty, 'status': 'filled'})
        except Exception as e:
            log(f"[{symbol}] Error al cerrar posición: {e}")
            summary.append({'symbol': symbol, 'action': action, 'qty': order_qty, 'status': f'error: {e}'})

    log(f"Todas las posiciones de {symbol} han sido procesadas.")
    return summary