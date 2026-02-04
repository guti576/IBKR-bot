from config.settings import *
from models.train import *
from models.predict import *
from trading.orders import *
import time


model = load_model("C:/Users/krist/OneDrive/Escritorio/IBKR bot/models/pickels/xgb_model_ibkr_15m_2h.pkl")

while True:
    # Close open positions
    close_all_positions()

    # Get latest data
    last_bars = getBars()
    X_latest = createFeatures(last_bars).iloc[-1:]
    price = X_latest['close']
    log("Latest bar:{} at price {}".format(X_latest.index, price))

    # Calculate signal {BUY, NO BUY}
    signal = create_signal(X_latest, model)

    if signal == 1:
        log("BUY")
        #place_buy_order()
    else:
        log("No Buy")

    time.sleep(CHECK_INTERVAL_MINUTES * 60)