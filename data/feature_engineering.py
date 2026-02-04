from config.settings import *

def getBars():

    # Descargar barras históricas
    bars = ib.reqHistoricalData(
        contract,
        endDateTime=endDateTime,
        durationStr=durationStr,
        barSizeSetting=barSizeSetting,
        whatToShow=whatToShow,
        useRTH=useRTH,
        formatDate=1
    )

    # Convertir bars a DataFrame
    data = []
    for bar in bars:
        dt = pd.Timestamp(bar.date)
        if dt.tzinfo is None:
            dt = dt.tz_localize('US/Eastern')  # Hora de NY (exchange)
        dt = dt.tz_convert('Europe/Madrid')    # Convertir a tu hora local si quieres
        data.append({
            'date': dt,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        })

    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    
    return df

def createFeatures(data):

    df = data.copy()
    df.sort_index(inplace=True)

    # =========================
    # 1. RETURNS BASE
    # =========================
    df['ret_1'] = df['close'].pct_change()
    df['ret_3'] = df['close'].pct_change(3)
    df['ret_6'] = df['close'].pct_change(6)
    df['ret_12'] = df['close'].pct_change(12)

    # =========================
    # 2. VOLATILIDAD
    # =========================
    df['vol_12'] = df['ret_1'].rolling(12).std()
    df['vol_24'] = df['ret_1'].rolling(24).std()
    df['vol_48'] = df['ret_1'].rolling(48).std()

    # =========================
    # 3. MEDIAS / TENDENCIA LARGA
    # =========================
    for w in [12, 24, 36, 48, 72]:
        df[f'sma_{w}'] = df['close'].rolling(w).mean()
        df[f'dist_sma_{w}'] = (df['close'] - df[f'sma_{w}']) / df[f'sma_{w}']

    # slopes (pendiente tendencia)
    df['slope_24'] = df['sma_24'].diff()
    df['slope_48'] = df['sma_48'].diff()

    # régimen de tendencia
    df['trend_regime'] = (df['sma_24'] > df['sma_48']).astype(int)

    # =========================
    # 4. MOMENTUM / ROC
    # =========================
    df['roc_12'] = df['close'].pct_change(12)
    df['roc_24'] = df['close'].pct_change(24)

    # =========================
    # 5. RSI
    # =========================
    def rsi(series, period=14):
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)

        ma_up = up.rolling(period).mean()
        ma_down = down.rolling(period).mean()

        rs = ma_up / ma_down
        return 100 - (100 / (1 + rs))

    df['rsi_14'] = rsi(df['close'], 14)
    df['rsi_28'] = rsi(df['close'], 28)

    # =========================
    # 6. VOLUMEN
    # =========================
    df['vol_mean_24'] = df['volume'].rolling(24).mean()
    df['rel_volume'] = df['volume'] / df['vol_mean_24']

    df['vol_zscore'] = (
        (df['volume'] - df['volume'].rolling(24).mean()) /
        df['volume'].rolling(24).std()
    )

    # OBV (On Balance Volume)
    df['obv'] = (np.sign(df['ret_1']) * df['volume']).fillna(0).cumsum()

    # presión compradora simple
    df['buy_pressure'] = df['ret_1'] * df['volume']

    # =========================
    # 7. FEATURES TEMPORALES (muy útiles intradía)
    # =========================
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute

    # =========================
    # 8. LIMPIEZA
    # =========================
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    return df