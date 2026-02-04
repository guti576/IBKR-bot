import datetime
import pandas as pd
import numpy as np
from ib_insync import *
import xgboost as xgb

# Definir fechas para los datos
endDateTime = datetime.datetime.now().strftime("%Y%m%d %H:%M:%S")
durationStr = '2 Y'  # últimos 5 días
barSizeSetting = '15 mins'
whatToShow = 'TRADES'
useRTH = True

# Target
TARGET_IN_BARS_AHEAD = 10

# Inference
PROB_THR = 0.55
CHECK_INTERVAL_MINUTES = 60 # Cada cuanto corre el bucle en minutos
STOP_LOSS_PCT = 0.98  # Stop-loss al 2% por debajo del precio de compra

# Conectar a IBKR (asegúrate de tener TWS o IB Gateway corriendo)
ib = IB()
ib.connect('127.0.0.1', 4002, clientId=21)  # Ajusta el clientId si es necesario

# Log functin
def log(msg):
    """
    Imprime log con timestamp
    """
    print(f"[{pd.Timestamp.now()}] {msg}")