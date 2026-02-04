import joblib
from config.settings import *

def load_model(path):
    # Cargar modelo
    model = joblib.load(path)
    log("Modelo cargado correctamente")

    return model


def create_signal(X_latest, model):
    '''
    Define the signal to buy/sell according to strategy. Pass latest data (X) available

    1: BUY
    2: SELL
    0: None
    '''

    # Predict
    proba = model.predict_proba(X_latest)[0][1]    
    log("Uptrend probability {:.4f}".format(proba))

    if proba > PROB_THR:
        return 1
    else:
        return 0
