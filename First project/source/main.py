import Parser
import ml
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pmdarima as pm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
pd.options.mode.copy_on_write = True
from sklearn.metrics import mean_squared_error, mean_absolute_error
from itertools import product
import torch.nn as nn
import torch
import json
from torch.utils.data import DataLoader, TensorDataset

def load_data_for_fe(ticker: str, hour: int):
    """
    Loads data for a given ticker symbol and hour from the CSV file.

    Args:
        ticker (str): The ticker symbol for which to load data.
        hour (int): The hour of the day for which to load data.

    Returns:
        pd.DataFrame: A DataFrame containing the volume data for the given ticker and hour.
    """
    # Read the data from the CSV file
    return pd.read_csv(f'data_moex/{ticker}_{hour}.csv')

def calculate_moving_average_and_error(time_series, cycle_size):
    """
    Calculate the moving average and error for the given time series and cycle size.

    Args:
        time_series (pd.Series): The time series for which to calculate the moving average and error.
        cycle_size (int): The size of the cycle for the moving average.

    Returns:
        tuple: A tuple of two pd.Series objects. The first is the moving average of the time series,
        and the second is the error of the moving average.
    """
    # Calculate the moving average
    time_series_ma = time_series.rolling(cycle_size).mean().dropna()
    # Calculate the error of the moving average
    time_series_error = (time_series - time_series_ma).dropna()
    return time_series_ma, time_series_error

def normalize_data_standard(dd_train):
    """
    Normalize the data using the StandardScaler.

    Args:
        dd_train (pd.DataFrame): The DataFrame with the data to be normalized.

    Returns:
        tuple: A tuple of the normalized DataFrame, the mean of the error, and the standard deviation of the error.
    """
    dd_train_norm = dd_train.copy()
    # Normalize the error column
    scaler_error = StandardScaler()
    dd_train['error'] = scaler_error.fit_transform(dd_train_norm['error'].values.reshape(-1, 1))
    error_mean, error_std = scaler_error.mean_, scaler_error.scale_

    # Normalize the close column
    scaler_close = StandardScaler()
    dd_train['close'] = scaler_close.fit_transform(dd_train['close'].values.reshape(-1, 1))

    return dd_train, error_mean, error_std

def train_arima_model(dd_train_norm):
    """
    Train an ARIMA model.

    Args:
        dd_train_norm (pd.DataFrame): The normalized DataFrame with the data to be used for training.

    Returns:
        pm.AutoARIMA: The trained ARIMA model.
    """
    # Train an ARIMA model using the auto_arima function from the pmdarima library
    ARIMA_model = pm.auto_arima(
        dd_train_norm[['error']],  # Use the 'error' column as the time series to model
        test='adf',  # Use the Augmented Dickey-Fuller test to determine the order of differencing
        start_p=1, start_q=1,  # Set the starting values for the model parameters
        max_p=6, max_q=6,  # Set the maximum values for the model parameters
        seasonal=False,  # Do not use seasonal differencing (i.e. do not account for seasonality)
        trace=False,  # Do not display the results of the model selection
        error_action='ignore',  # Ignore any errors that occur during the model selection
        suppress_warnings=True,  # Suppress any warnings that occur during the model selection
        stepwise=True)  # Use a stepwise approach to select the model parameters
    return ARIMA_model

def forecast_arima(arma_model, test_size: int, error_mean, error_std, time_series_ma, cycle_size):
    """
    Generate a forecast using the trained ARIMA model.

    Args:
        arma_model (pm.AutoARIMA): The trained ARIMA model.
        test_size (int): The number of samples to forecast.
        error_mean (float): The mean of the error.
        error_std (float): The standard deviation of the error.
        time_series_ma (pd.Series): The moving average of the time series.
        cycle_size (int): The size of the cycle for the moving average.

    Returns:
        np.ndarray: The forecasted values of the time series.
    """
    # Generate a forecast using the trained ARIMA model
    forecast = arma_model.predict(n_periods=test_size)

    # Denormalize the forecasted values
    forecast_denorm = forecast * error_std + error_mean

    # Calculate the forecasted values of the time series by adding the denormalized forecast
    # to the moving average
    forecast_volume = time_series_ma.values.reshape(-1)[-cycle_size:-cycle_size + test_size] + forecast_denorm
    return forecast_volume

def forecaster_fe(tickers: list, hours, cycle_size: int, n_splits: int = 20, test_size: int = 1):
    """
    Conduct forecasting using feature engineering (FE) for the given tickers and hours.

    Parameters
    ----------
    tickers : list
        List of ticker symbols to forecast.
    hours : list
        List of hours to forecast.
    cycle_size : int
        Size of the cycle for the moving average.
    n_splits : int, optional
        Number of times to split the data for forecasting. Defaults to 20.
    test_size : int, optional
        Number of samples to forecast. Defaults to 1.

    Returns
    -------
    list
        List of dictionaries containing the forecasted values for each ticker and hour.
    """
    results = []
    for ticker, hour in product(tickers, hours):
        # Get the data for the current ticker and hour
        dd = load_data_for_fe(ticker, hour)

        total_length = len(dd)

        train_size = total_length - n_splits - 1
        # Iterate over the number of splits
        for i in range(n_splits):
            # Calculate the size of the training set
            train_size += 1
            
            # Split the data into training and test sets
            dd_train = dd.iloc[:train_size]
            dd_test = dd.iloc[train_size:train_size + test_size]

            # Check that the test set is not empty
            if len(dd_test) < test_size:
                continue
            
            # Calculate the moving average and error for the time series
            time_series = dd_train["volume"].copy()
            time_series_ma, time_series_error = calculate_moving_average_and_error(time_series, cycle_size)

            # Add the moving average and error to the data
            dd_train.loc[time_series_ma.index, 'trend'] = time_series_ma.values.astype(float)
            dd_train.loc[time_series_error.index, 'error'] = time_series_error.values.astype(float)

            # Drop the first cycle_size rows of the data
            dd_train = dd_train.loc[cycle_size:].copy()

            # Normalize the data using standard normalization
            dd_train_norm, error_mean, error_std = normalize_data_standard(dd_train)

            # Train an ARIMA model on the normalized data
            arma_model = train_arima_model(dd_train_norm)

            # Generate a forecast using the trained ARIMA model
            forecast_volume = forecast_arima(arma_model, test_size, error_mean, error_std, time_series_ma, cycle_size)

            # Drop the last test_size rows of the data
            dd_train = dd[:len(dd)-test_size]
            
            # Append the forecasted values to the results list
            results.append({
                "ticker": ticker,
                "hour": hour,
                "forecast_volume": forecast_volume,
                "i": i
            })
        
    return results


def aggregate_fe_forecasts(results, n_splits=20):
    """
    Aggregate the feature engineering forecasts and compare with actual data.

    Parameters
    ----------
    results : list
        List of dictionaries containing the forecasted values for each ticker and hour.
    n_splits : int, optional
        Number of splits to use for aggregation. Defaults to 20.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the aggregated forecasts and actual volumes.
    """
    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results)

    # Group by ticker, hour and day, and take the mean of the forecasted volumes
    aggregated_forecasts = results_df.groupby(['ticker', 'hour', 'i'], as_index=False)['forecast_volume'].mean()

    # Compare the aggregated forecasts with actual data
    actual_volumes = []
    for _, row in aggregated_forecasts.iterrows():
        ticker = row['ticker']
        hour = row['hour']
        day = row['i']
        
        # Load the actual data for the given ticker, hour and day
        actual_data = load_data_for_fe(ticker, hour)
        total_length = len(actual_data)
        # Sum the volume of the day, filtering by date
        actual_volume = actual_data[actual_data.index == day + total_length - n_splits]['volume'].sum()
        actual_volumes.append(actual_volume)
    
    # Add the actual volumes to the DataFrame
    aggregated_forecasts['actual_volume'] = actual_volumes
    
    return aggregated_forecasts

def load_data_for_ml(ticker: str) -> pd.DataFrame:
    """
    Loads the data for machine learning tasks for a given ticker symbol.

    Parameters
    ----------
    ticker : str
        Ticker symbol of the stock.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the data for the given ticker.
    """
    # Load the data from the CSV file
    data = pd.read_csv(f'data_1h_moex/{ticker}_volume.csv')
    
    return data

def create_dataset(dd, n_splits, forecast_horizon, window_size):
    """
    Creates a dataset for machine learning tasks from a given DataFrame.

    Parameters
    ----------
    dd : pd.DataFrame
        DataFrame containing the data.
    n_splits : int
        Number of splits to create in the dataset.
    forecast_horizon : int
        Forecast horizon for the task.
    window_size : int
        Window size for each split.

    Returns
    -------
    tuple
        Tuple containing two numpy arrays, X and y, where X is the input data
        and y is the target data.
    """
    X, y = [], []
    start = len(dd) - (n_splits + window_size) * forecast_horizon
    end = len(dd) - n_splits * forecast_horizon - forecast_horizon

    # Iterate over the number of splits
    for i in range(n_splits):   
        # Increment the start and end indices by the forecast horizon
        start += forecast_horizon
        end += forecast_horizon
        # Append the current window of data to X
        X.append(dd.iloc[start:end]['volume'].values)
        # Append the current window of targets to y
        y.append(dd.iloc[end:end + forecast_horizon]['volume'].values)

    # Convert X and y to numpy arrays and return them
    return np.array(X), np.array(y)

def normalize_data_minmax(dd_train):
    """
    Normalize the data using the MinMaxScaler.

    Parameters
    ----------
    dd_train : pd.DataFrame
        DataFrame containing the data to be normalized.

    Returns
    -------
    tuple
        Tuple containing the normalized DataFrame, the minimum of the data,
        the maximum of the data, and the scaler object.
    """
    scaler = MinMaxScaler()
    dd_train_norm = dd_train.copy()
    dd_train_norm['volume'] = scaler.fit_transform(
        dd_train_norm['volume'].values.reshape(-1, 1)
    )
    data_min, data_max = scaler.data_min_, scaler.data_max_
    return dd_train_norm, data_min, data_max, scaler

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=14, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Perform a forward pass through the LSTM model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor of shape (batch_size, sequence_length, input_size).

        Returns
        -------
        torch.Tensor
            The output tensor of shape (batch_size, output_size).
        """
        batch_size = x.size(0)
        
        # Initialize the hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Pass the input through the LSTM layers
        out, _ = self.lstm(x, (h0, c0))
        
        # Use the last hidden state to pass through the fully connected layer
        out = self.fc(out[:, -1, :])
        
        return out

def forcaster_lstm(tickers: list, best_params_results: list, window_size: int = 500, n_splits: int = 20, forecast_horizon: int = 14):
    """
    Train an LSTM model for each ticker and make predictions for the specified forecast horizon.

    Parameters
    ----------
    tickers : list
        List of tickers to forecast.
    best_params_results : list
        List of dictionaries with the best parameters for each ticker.
    window_size : int, optional
        Size of the window for the sliding window approach. Defaults to 500.
    n_splits : int, optional
        Number of splits for the time series cross-validation. Defaults to 20.
    forecast_horizon : int, optional
        Forecast horizon, i.e. how many days in the future to forecast. Defaults to 14.

    Returns
    -------
    dict
        Dictionary with the forecasted values for each ticker.
    """
    results = {}

    for ticker in tickers:
        # Set random seed for reproducibility
        torch.manual_seed(342)

        # Load and normalize data for the current ticker
        dd = load_data_for_ml(ticker)
        dd, min, max, scaler = normalize_data_minmax(dd)

        # Retrieve best parameters or use default values
        if ticker in best_params_results:
            best_params = best_params_results[ticker][0]
            best_hidden_size = best_params["hidden_size"]
            best_num_layers = best_params["num_layers"]
            best_learning_rate = best_params["learning_rate"]
            best_window_size = best_params['window_size']
            print(f"Parameters for ticker {ticker}: hidden_size={best_hidden_size}, num_layers={best_num_layers}, learning_rate={best_learning_rate}, window_size={best_window_size}")
        else:
            print(f"Parameters for ticker {ticker} not found, using default values.")
            best_hidden_size = 100
            best_num_layers = 3
            best_learning_rate = 0.0005599305480432884
            best_window_size = 33

        # Create dataset for training
        X, y = create_dataset(dd.iloc[:-n_splits*forecast_horizon], n_splits, forecast_horizon, best_window_size)
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(y, dtype=torch.float32)

        # Initialize DataLoader
        train_loader = DataLoader(TensorDataset(X, y), batch_size=n_splits, shuffle=True)

        # Initialize model, loss function and optimizer
        model = LSTMModel(input_size=1, hidden_size=best_hidden_size, num_layers=best_num_layers).to('cpu')
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=best_learning_rate)

        # Train the model
        num_epochs = 50
        for epoch in range(num_epochs):
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        # Evaluate the model and make predictions
        model.eval()
        last_sequence = torch.tensor(dd.iloc[-(n_splits + best_window_size) * forecast_horizon:-n_splits * forecast_horizon]['volume'].values, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        prediction = []
        actual_values = dd.iloc[-n_splits * forecast_horizon:]['volume'].values
        with torch.no_grad():
            for i in range(n_splits):
                pred = model(last_sequence)
                prediction.append(pred[0].cpu().numpy())
                real_value = torch.tensor([actual_values[i]], dtype=torch.float32).unsqueeze(-1)
                last_sequence = torch.cat((last_sequence[:, 1:, :], real_value.unsqueeze(0)), dim=1)

        # Rescale predictions and calculate RMSE
        prediction = prediction * (max - min) + min
        dd['volume'] = scaler.inverse_transform(dd['volume'].values.reshape(-1, 1))
        score = np.sqrt(np.mean((np.array(prediction).flatten() - dd.iloc[-n_splits * forecast_horizon:]['volume'].values) ** 2))

        print(f'Score: {score:.4f} RMSE')

        # Store the forecasted and actual volumes in a DataFrame
        new_rows = pd.DataFrame({
            'forecast_volume': prediction.ravel(),
            'actual_volume': dd.iloc[-n_splits * forecast_horizon:]['volume'].to_numpy()
        })
        
        results[ticker] = new_rows

    return results

def get_agregate_forecasts():
    """
    This function aggregates the forecasts of the LSTM and FE models, and
    returns a dictionary of DataFrames with the forecasted and actual values
    for each ticker.

    Returns
    -------
    dict
        A dictionary of DataFrames with the forecasted and actual values for each ticker.
    """
    tickers = Parser.get_tickers(Parser.get_ticker_names())

    if Path('best_params_results.json').exists():
        with open("best_params_results.json", "r") as f:
            best_params_results = json.load(f)
    else:
        best_params_results = {}

    # Get the forecasts from the LSTM model
    results_ml = forcaster_lstm(tickers, best_params_results)

    # Get the forecasts from the FE model
    results_fe = forecaster_fe(tickers, range(10, 24), cycle_size=20)
    results_fe_sort = aggregate_fe_forecasts(results_fe).sort_values(by=['ticker', 'i', 'hour']).reset_index(drop=True).drop(['hour', 'i'], axis=1)

    # Get the forecasts from the stacking model
    all_stacks = []

    for ticker in tickers:
        stack = ml.stacking(ticker)
        all_stacks.append(stack)

    # Combine all the data into one DataFrame
    combined_data = pd.concat(all_stacks, ignore_index=True)

    # Add a column for the ticker
    ticker_column = []
    rows_per_ticker = 280  # количество строк на тикер

    # Create a list of tickers with repeated values for each ticker
    ticker_column = [ticker for ticker in tickers for _ in range(rows_per_ticker)]

    # Check that the size of the list matches the number of rows in the DataFrame
    if len(ticker_column) == len(combined_data):
        combined_data['ticker'] = ticker_column
    else:
        print("Sizes do not match, check the number of rows and tickers.")

    # Add the LSTM forecasts to the DataFrame
    for ticker, lstm_df in results_ml.items():
        # Select only the necessary columns from the current DataFrame for the ticker and add to the 'lstm' column
        lstm_values = lstm_df['forecast_volume'].values
        # Compare the number of values in combined_data for the current ticker and the number of values in lstm_df
        if len(lstm_values) == combined_data[combined_data['ticker'] == ticker].shape[0]:
            # Fill the 'lstm' column with the values from lstm_values for the current ticker
            combined_data.loc[combined_data['ticker'] == ticker, 'lstm'] = lstm_values
        else:
            print(f"Data sizes for ticker {ticker} do not match.")

    # Create an empty column 'fe' with type float
    combined_data['fe'] = np.nan

    # Fill the 'fe' column with the values from results_fe_sort for each ticker row by row
    for ticker in combined_data['ticker'].unique():
        # Values 'fe' for the current ticker
        fe_values = results_fe_sort[results_fe_sort['ticker'] == ticker]['forecast_volume'].astype(float).values

        # Update the values 'fe' for the rows where ticker matches
        combined_data.loc[combined_data['ticker'] == ticker, 'fe'] = fe_values

    # Return a dictionary of DataFrames with the forecasted and actual values for each ticker
    return {ticker: df for ticker, df in combined_data.groupby('ticker')}

def get_results():
    """
    Aggregates forecast results for each ticker and calculates various error
    metrics. It combines the forecasts from different models using weighted
    averaging and plots the results.

    Returns
    -------
    pd.DataFrame
        DataFrame containing RMSE, MSE, MAE, MAPE, and RMSE for each model and
        combined forecast for each ticker.
    """
    results = []

    # Iterate over each ticker
    for ticker in Parser.get_tickers(Parser.get_ticker_names()):
        
        # Get aggregated forecasts for the current ticker
        df = get_agregate_forecasts()[ticker]

        # Calculate absolute errors for each model
        df['error_lstm'] = np.abs(df['lstm'] - df['actual_volume'])
        df['error_fe'] = np.abs(df['fe'] - df['actual_volume'])
        df['error_rf'] = np.abs(df['rf'] - df['actual_volume'])
        df['error_gb'] = np.abs(df['gb'] - df['actual_volume'])

        # Calculate weights based on inverse error
        df['weight_lstm'] = 1 / df['error_lstm']
        df['weight_fe'] = 1 / df['error_fe']
        df['weight_rf'] = 1 / df['error_rf']
        df['weight_gb'] = 1 / df['error_gb']

        # Normalize the weights
        total_weights = (df['weight_lstm'] + df['weight_fe'] +
                        df['weight_rf'] + df['weight_gb'])

        df['norm_weight_lstm'] = df['weight_lstm'] / total_weights
        df['norm_weight_fe'] = df['weight_fe'] / total_weights
        df['norm_weight_rf'] = df['weight_rf'] / total_weights
        df['norm_weight_gb'] = df['weight_gb'] / total_weights

        # Calculate the combined forecast using normalized weights
        combined_forecast = (df['norm_weight_lstm'] * df['lstm'] +
                            df['norm_weight_fe'] * df['fe'] +
                            df['norm_weight_rf'] * df['rf'] +
                            df['norm_weight_gb'] * df['gb'])

        # Calculate RMSE for each model and the combined forecast
        rmse_lstm = np.sqrt(mean_squared_error(df['actual_volume'], df['lstm']))
        rmse_fe = np.sqrt(mean_squared_error(df['actual_volume'], df['fe']))
        rmse_rf = np.sqrt(mean_squared_error(df['actual_volume'], df['rf']))
        rmse_gb = np.sqrt(mean_squared_error(df['actual_volume'], df['gb']))
        rmse_combined = np.sqrt(mean_squared_error(df['actual_volume'], combined_forecast))

        # Calculate additional error metrics
        mse = mean_squared_error(df['actual_volume'], combined_forecast)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(df['actual_volume'], combined_forecast)
        mape = np.mean(np.abs((df['actual_volume'] - combined_forecast) / df['actual_volume'])) * 100  # In percentage

        # Append results to the list
        results.append({
            'ticker': ticker,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'rmse_lstm': rmse_lstm,
            'rmse_fe': rmse_fe,
            'rmse_rf': rmse_rf,
            'rmse_gb': rmse_gb,
            'rmse_combined': rmse_combined
        })

        # Plotting the forecasts and combined forecast
        plt.figure(figsize=(15, 10))
        plt.plot(df['actual_volume'], label='Actual Volume', color='black', linestyle='--')
        plt.plot(df['fe'], label='FE Forecast', color='blue', alpha=0.7)
        plt.plot(combined_forecast, label='Combined Forecast', color='red', linestyle=':')
        plt.title(f'{ticker} Forecasts and Combined Forecast')
        plt.xlabel('Time Index')
        plt.ylabel('Trading Volume')
        plt.legend()
        plt.show()

    # Return a DataFrame with RMSE results for all tickers
    return pd.DataFrame(results)
    
print(get_results())
