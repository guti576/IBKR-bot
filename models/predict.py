import joblib
from config.settings import *

def load_model(path="C:/Users/krist/OneDrive/Escritorio/IBKR bot/models/xgb_model_ibkr.pkl"):
    # Cargar modelo
    model = joblib.load(path)
    print("Modelo cargado correctamente")

    return model


def create_signal(X_latest):
    '''
    Define the signal to buy/sell according to strategy. Pass latest data (X) available

    1: BUY
    2: SELL
    0: None
    '''

    model = load_model("C:/Users/krist/OneDrive/Escritorio/IBKR bot/models/xgb_model_ibkr.pkl")

    # Predict
    proba = model.predict_proba(X_latest)[0][1]    
    log("Uptrend probability {:.4f}".format(proba))

    if proba > PROB_THR:
        return 1
    else:
        return 0
