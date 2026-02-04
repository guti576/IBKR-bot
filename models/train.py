from config.settings import *
from data.feature_engineering import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


def train_xgb_model():
    
    # Training
    df_raw = getBars()
    df = createFeatures(df_raw)
        
    # Target: 1 si sube dentro de 12 barras, 0 si no
    df['future_close'] = df['close'].shift(-TARGET_IN_BARS_AHEAD)
    df['target'] = (df['future_close'] > df['close']).astype(int)

    # dropna (no target)
    df = df.dropna()

    # Features y target
    #feature_cols = ['open', 'high', 'low', 'close', 'volume', 'return', 'volatility', 'ma_12', 'ma_24']
    X = df.drop(columns=['target', 'future_close'])
    y = df['target']


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # Importante: no barajamos para series temporales
    )

    # Inicializar modelo
    model = xgb.XGBClassifier(
        n_estimators=500,          # más árboles pero más pequeños
        max_depth=2,              # ↓ MUY importante (reduce varianza)
        learning_rate=0.03,       # ↓ más suave
        subsample=0.6,            # ↓ más randomización
        colsample_bytree=0.6,     # ↓ menos features por árbol
        gamma=2,                  # penaliza splits débiles
        reg_alpha=1,              # L1
        reg_lambda=2,             # L2
        random_state=42,
        eval_metric='logloss'
    )

    # Entrenar
    model.fit(X_train, y_train)

    # Evaluar train
    y_pred = model.predict(X_train)
    acc = accuracy_score(y_train, y_pred)
    print(f'Accuracy train: {acc:.4f}')

    # Evaluar test
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'Accuracy test: {acc:.4f}')

    return model


def save_model(model, path = "C:/Users/krist/OneDrive/Escritorio/IBKR bot/models/xgb_model_ibkr.pkl"):
    joblib.dump(model, path)
    print("Modelo guardado en:", path)