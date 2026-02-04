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

cheap_liquid_nasdaq = [
    "BBAI",  # BigBear.ai Holding (~$5-$7, buena cobertura y volumen) :contentReference[oaicite:1]{index=1} -> Accuracy test: 0.5051
    "LUMN",  # Lumen Technologies (~$8-$9 +/-) volumen decente :contentReference[oaicite:2]{index=2} Accuracy test: 0.5113
    "NUVB",  # Nuvation Bio (~$5-$6) cobertura de analistas y volumen medio :contentReference[oaicite:4]{index=4} Accuracy test: 0.4992
    "OPEN",  # Opendoor Technologies (~$6-$7) volumen medio decente :contentReference[oaicite:5]{index=5} Accuracy test: 0.5669
    "SIRI",  # Sirius XM (~$4-$5) volumen tradicionalmente alto :contentReference[oaicite:6]{index=6} Accuracy test: 0.4637
    "IQ",    # iQIYI (~$5) compañía bajo $10 con mayor capitalización media :contentReference[oaicite:7]{index=7} Accuracy test: 0.5374
    "YEXT",  # Yext (~$5) volumen moderado en listas de acciones baratas :contentReference[oaicite:8]{index=8} Accuracy test: 0.4778
    "COUR"  # Coursera (~$5-$6) volumen relativamente estable :contentReference[oaicite:9]{index=9} Accuracy test: 0.4595
]

for symbol in cheap_liquid_nasdaq:

    try:
        # Define el contrato de la acción
        #contract = Stock(symbol, 'SMART', 'USD')

        # Train model
        print(symbol)
        #model = train_xgb_model(contract)

        # Save model
        path = "C:/Users/krist/OneDrive/Escritorio/IBKR bot/models/pickels/cheap/{}_xgb_model_ibkr_15m_2h.pkl".format(symbol)
        #save_model(model, path)
    except:
        log("Error en el entrenamiento de:", symbol)


while True:
    
    for symbol in cheap_liquid_nasdaq:
        
        print("-"*30)
        print(symbol)
        print("-"*30)

        # Define el contrato de la acción
        contract = Stock(symbol, 'SMART', 'USD')
        
        # Close open positions
        close_all_positions(symbol)

        # Load model
        path = "C:/Users/krist/OneDrive/Escritorio/IBKR bot/models/pickels/cheap/{}_xgb_model_ibkr_15m_2h.pkl".format(symbol)
        model = load_model(path)

        # Get latest data
        last_bars = getBars(contract)
        X_latest = createFeatures(last_bars).iloc[-1:]
        last_date = X_latest.index[0].strftime('%Y-%m-%d %H:%M:%S')
        price = X_latest['close'].iloc[0]
        log("Price: {}".format(price))
        log("Date (Madrid): {}".format(last_date))

        # Calculate signal {BUY, NO BUY}
        signal = create_signal(X_latest, model)

        if signal == 1:
            log("BUY")
            place_buy_order(contract)
        else:
            log("No Buy")

    time.sleep(CHECK_INTERVAL_MINUTES * 60)