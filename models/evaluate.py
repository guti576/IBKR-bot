from config.settings import *
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.inspection import permutation_importance
import plotly.graph_objects as go
import joblib


def plot_roc_auc(model, X_test, y_test):
    """
    Plotea la curva ROC y devuelve el AUC.
    Compatible con XGBClassifier.
    """

    # Probabilidades clase positiva
    y_proba = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1], [0,1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    plt.show()

    print(f"AUC: {roc_auc:.4f}")
    return roc_auc


def permutation_importance_top_features(
    model,
    X_test,
    y_test,
    n_top=20,
    n_repeats=5,
    random_state=42,
    plot=True
):
    """
    Calcula permutation importance sobre datos OUT-OF-SAMPLE.

    Parameters
    ----------
    model : modelo entrenado
    X_test : DataFrame
    y_test : array/Series
    n_top : nº de features a devolver
    n_repeats : repeticiones permutation
    plot : mostrar gráfico

    Returns
    -------
    top_features : list[str]
    importances_df : DataFrame ordenado
    """

    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )

    importances = pd.Series(
        result.importances_mean,
        index=X_test.columns
    )

    importances_df = (
        importances
        .sort_values(ascending=False)
        .to_frame("importance")
    )

    top_features = importances_df.head(n_top).index.tolist()

    if plot:
        importances_df.head(n_top).sort_values("importance").plot(
            kind="barh",
            figsize=(8,6)
        )
        plt.title(f"Top {n_top} Permutation Importance (OOS)")
        plt.xlabel("Importance drop")
        plt.grid()
        plt.show()

    return top_features, importances_df


def plot_candles_with_signals(X, y_pred_proba, y_test, PROB_THR):

    df = X.copy().sort_index()

    y_pred_proba = np.asarray(y_pred_proba)

    # ---- prob clase positiva ----
    if y_pred_proba.ndim == 2:
        proba_buy = y_pred_proba[:, 1]
    else:
        proba_buy = y_pred_proba

    y_test = np.asarray(y_test)

    # ---- high/low simulados ----
    high = df['high']
    low  = df['low']

    # ---- señales ----
    signal_mask = proba_buy > PROB_THR

    signal_times = df.index[signal_mask]
    signal_prices = df.loc[signal_mask, 'close']

    # ---- evaluar acierto ----
    correct_mask = y_test[signal_mask] == 1

    times_correct = signal_times[correct_mask]
    prices_correct = signal_prices[correct_mask]

    times_wrong = signal_times[~correct_mask]
    prices_wrong = signal_prices[~correct_mask]

    fig = go.Figure()

    # ======================
    # Velas
    # ======================
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=high,
            low=low,
            close=df['close'],
            name="Price"
        )
    )

    # ======================
    # Señales correctas (azul)
    # ======================
    fig.add_trace(
        go.Scatter(
            x=times_correct,
            y=prices_correct,
            mode='markers',
            marker=dict(
                symbol='triangle-up',
                size=12,
                color='blue'
            ),
            name='Buy correct'
        )
    )

    # ======================
    # Señales incorrectas (negro)
    # ======================
    fig.add_trace(
        go.Scatter(
            x=times_wrong,
            y=prices_wrong,
            mode='markers',
            marker=dict(
                symbol='triangle-up',
                size=12,
                color='black'
            ),
            name='Buy wrong'
        )
    )

    fig.update_layout(
        title=f"Señales compra (azul=acierto | negro=fallo)",
        xaxis_rangeslider_visible=False,
        height=650
    )

    fig.show()

    # ======================
    # Métricas rápidas
    # ======================
    n_trades = signal_mask.sum()
    if n_trades > 0:
        winrate = correct_mask.mean()
        print(f"Trades: {n_trades}")
        print(f"Accuracy señales: {winrate:.2%}")

#plot_candles_with_signals(X_test, y_pred_proba, y_test, PROB_THR)


def evaluate_threshold_strategy(
    X,
    y_pred_proba,
    y_test,
    PROB_THR,
    horizon=4,
    initial_cash=10_000
):
    """
    Evalúa señales de compra y calcula métricas + rentabilidad.

    Returns
    -------
    summary : pd.DataFrame
    trades  : pd.DataFrame
    """

    df = X.copy().sort_index()

    y_pred_proba = np.asarray(y_pred_proba)
    y_test = np.asarray(y_test)

    # ---- prob clase positiva ----
    if y_pred_proba.ndim == 2:
        proba_buy = y_pred_proba[:, 1]
    else:
        proba_buy = y_pred_proba

    closes = df['close'].values

    # =========================
    # Señales
    # =========================
    signal_mask = proba_buy > PROB_THR
    signal_idx = np.where(signal_mask)[0]

    trades = []

    for i in signal_idx:

        if i + horizon >= len(closes):
            continue

        entry_price = closes[i]
        exit_price = closes[i + horizon]

        ret = (exit_price - entry_price) / entry_price
        correct = (y_test[i] == 1)

        trades.append([
            df.index[i],
            df.index[i + horizon],
            entry_price,
            exit_price,
            ret,
            correct
        ])

    trades = pd.DataFrame(
        trades,
        columns=[
            'entry_time',
            'exit_time',
            'entry_price',
            'exit_price',
            'ret',
            'correct'
        ]
    )

    # =========================
    # Métricas clasificación
    # =========================
    n_trades = len(trades)
    tp = trades['correct'].sum()
    fp = n_trades - tp

    precision = tp / n_trades if n_trades else 0
    coverage = n_trades / len(df)

    # =========================
    # Equity curve
    # =========================
    equity = initial_cash
    equity_curve = []

    for r in trades['ret']:
        equity *= (1 + r)
        equity_curve.append(equity)

    trades['equity'] = equity_curve

    final_cash = equity
    pnl_eur = final_cash - initial_cash
    pnl_pct = pnl_eur / initial_cash

    # =========================
    # Tabla resumen
    # =========================
    summary = pd.DataFrame({
        "metric": [
            "threshold",
            "total_bars",
            "signals_taken",
            "coverage_%",
            "correct",
            "wrong",
            "precision_% (hit rate)",
            "avg_return_per_trade_%",
            "final_equity_€",
            "pnl_€",
            "pnl_%"
        ],
        "value": [
            PROB_THR,
            len(df),
            n_trades,
            round(coverage * 100, 2),
            int(tp),
            int(fp),
            round(precision * 100, 2),
            round(trades['ret'].mean() * 100 if n_trades else 0, 3),
            round(final_cash, 2),
            round(pnl_eur, 2),
            round(pnl_pct * 100, 2)
        ]
    })

    print("\n===== BACKTEST SUMMARY =====")
    print(summary.to_string(index=False))

    return summary, trades