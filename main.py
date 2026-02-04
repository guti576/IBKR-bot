from config.settings import *
from models.train import *
from models.predict import *
from trading.orders import *
import time

liquid_nasdaq_symbols = [
    "MSFT",  # Microsoft
    "AAPL",  # Apple
    "NVDA",  # NVIDIA
    "AMZN",  # Amazon
    "GOOGL", # Alphabet Class A
    "META"  # Meta Platforms
]

for symbol in liquid_nasdaq_symbols:

    # Define el contrato de la acción
    contract = Stock(symbol, 'SMART', 'USD')

    print(symbol)
    model = train_xgb_model(contract)

    path = "C:/Users/krist/OneDrive/Escritorio/IBKR bot/models/pickels/{}_xgb_model_ibkr_15m_2h.pkl".format(symbol)
    save_model(model, path)

