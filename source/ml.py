import pandas as pd
import ta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt


def load_data_1h(ticker: str): 
    """
    Loads data for a given ticker symbol from a CSV file.

    Args:
        ticker (str): The ticker symbol for which to load data.

    Returns:
        pd.DataFrame: A DataFrame containing the close and volume data for the given ticker.
    """
    df = pd.read_csv(f'data_1h_moex/{ticker}_volume.csv')
    df = df.loc[:,['close', 'volume']]
    return df 



def data_preprocessing(ticker : str):

    """
    This function takes a ticker symbol as an argument and performs the following technical indicator calculations:
    - Simple Moving Average (SMA)
    - Exponential Moving Average (EMA)
    - Relative Strength Index (RSI)
    - Moving Average Convergence Divergence (MACD)
    - Bollinger Bands
    - Rate of Change (ROC)
    - Stochastic Oscillator
    - Momentum
    - Detrended Price Oscillator (DPO)

    The function then drops any rows with NaN values and returns the DataFrame with the calculated technical indicators.

    Parameters:
    ticker (str): The ticker symbol of the stock to perform the calculations on.

    Returns:
    DataFrame: A DataFrame with the calculated technical indicators.
    """
    data = load_data_1h(ticker)
    # 1. Simple Moving Average (SMA)
    data['sma_10'] = data['close'].rolling(window=10).mean()
    data['sma_20'] = data['close'].rolling(window=20).mean()

    # 2. Exponential Moving Average (EMA)
    data['ema_10'] = data['close'].ewm(span=10, adjust=False).mean()
    data['ema_20'] = data['close'].ewm(span=20, adjust=False).mean()

    # 3. Relative Strength Index (RSI)
    data['rsi'] = ta.momentum.RSIIndicator(close=data['close'], window=14).rsi()

    # 4. Moving Average Convergence Divergence (MACD)
    macd = ta.trend.MACD(close=data['close'])
    data['macd'] = macd.macd()
    data['macd_signal'] = macd.macd_signal()

    # 5. Bollinger Bands
    bollinger = ta.volatility.BollingerBands(close=data['close'], window=20, window_dev=2)
    data['bollinger_mavg'] = bollinger.bollinger_mavg()
    data['bollinger_hband'] = bollinger.bollinger_hband()
    data['bollinger_lband'] = bollinger.bollinger_lband()

    # 6. Rate of Change (ROC)
    data['roc'] = ta.momentum.ROCIndicator(close=data['close'], window=12).roc()

    # 7. Stochastic Oscillator
    data['stoch_k'] = ((data['close'] - data['close'].rolling(window=14).min()) /
                    (data['close'].rolling(window=14).max() - data['close'].rolling(window=14).min())) * 100

    # 8. Momentum
    data['momentum'] = data['close'] - data['close'].shift(4)

    # 9. Detrended Price Oscillator (DPO)
    data['dpo'] = data['close'] - data['close'].rolling(window=14).mean()

    data[['sma_10', 'ema_10', 'rsi', 'macd', 'bollinger_mavg', 'roc', 'stoch_k', 'momentum', 'dpo']].head()
    data = data.dropna(axis=0)
    return data




def future_selection(X, y):
    """
    Select features for future prediction using XGBoost.

    Parameters
    ----------
    X_train : pandas.DataFrame
        DataFrame containing features.
    y_train : pandas.Series
        Series containing target values.

    Returns
    -------
    None

    Notes
    -----
    This function uses XGBoost to select the most important features, and then
    plots the feature importances using `xgb.plot_importance`.
    """
    model = xgb.XGBRegressor()
    model.fit(X,y)
    xgb.plot_importance(model)
    plt.show()


def stacking(ticker : str):
    """
    Train a stacking model for a given ticker.

    Parameters
    ----------
    ticker : str
        Ticker symbol for the stock.

    Returns
    -------
    None

    Notes
    -----
    This function trains a stacking model using a random forest, gradient
    boosting, and linear regression as the base models. It then plots the
    actual vs predicted values.
    """
    data = data_preprocessing(ticker)
    y = data['volume']
    X = data.drop(columns=['volume', 'ema_10', 'ema_20', 'sma_20', 'sma_10', 'bollinger_hband', 'bollinger_lband', 'stoch_k', 'roc'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.052375607931163484,shuffle=False, random_state=42)

    rf_model = RandomForestRegressor(max_depth= 7,
        max_features= 0.6705241236872915,
        max_leaf_nodes= 48,
        min_samples_leaf= 2,
        min_samples_split= 5,
        n_estimators= 138,
        random_state=42)

    gb_model = xgb.XGBRegressor(colsample_bytree= 0.7865412521282844, 
        gamma=0.5908332605690108, 
        learning_rate= 0.04050024993904943, 
        max_depth= 4, 
        n_estimators= 219, 
        reg_alpha= 0.8226005606596583, 
        reg_lambda= 0.3601906414112629, 
        subsample= 0.6270605126518848,
        random_state=42)
    lr_model = LinearRegression()

    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)
    lr_model.fit(X_train, y_train)

    rf_pred = rf_model.predict(X_test)
    gb_pred = gb_model.predict(X_test)
    lr_pred = lr_model.predict(X_test)
    dd = load_data_1h(ticker)
    stack = pd.DataFrame({'rf': rf_pred, 'gb': gb_pred, 'lr': lr_pred, 'actual_volume': dd.iloc[-280:]['volume'].to_numpy()})

    return stack