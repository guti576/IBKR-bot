from config.settings import *
from models.train import *
from trading.orders import *
from models.predict import *


model = load_model("C:/Users/krist/OneDrive/Escritorio/IBKR bot/models/xgb_model_ibkr.pkl")

# Get latest data
last_bars = getBars()
X_latest = createFeatures(last_bars).iloc[-1:]
price = X_latest['close']
log("Latest bar:{} at price {}".format(X_latest.index, price))

# Calculate signal {BUY, NO BUY}
signal = create_signal(X_latest)

if signal == 1:
    log("BUY")
else:
    log("No Buy")