import pandas as pd
import requests as rq
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def get_ticker_names():
    """
    Retrieve a list of stock ticker symbols with a capitalization greater than 1 trillion rubles.

    Returns:
        list: A list of stock ticker symbols with a capitalization greater than 1 trillion rubles.
    """
    # Make a request to the server
    url = "https://iss.moex.com/iss/engines/stock/markets/shares/securities.json"

    response = rq.get(url)

    if response.status_code == 200:
        data = response.json()
    else:
        print("Error: Unable to retrieve data")
        
    # Parse the response
    columns = response.json()["securities"]['columns']
    data = response.json()["securities"]["data"]

    df = pd.DataFrame(data, columns=columns)
    df = df[df["BOARDID"]=="TQBR"]
    df = df[df["ISSUESIZE"]*df["PREVWAPRICE"]>1e+12].reset_index().drop('index', axis=1)
    
    # Extract the ticker symbols
    ticker_names = df["SECID"].values.tolist()
    return ticker_names

def get_tickers(ticker_names):
    """
    Retrieve a list of tickers with data starting from June 1, 2023.

    Args:
        ticker_names (list): A list of stock ticker symbols.

    Returns:
        list: A list of tickers with data starting from June 1, 2023.
    """
    tickers = []
    date_today = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")

    for ticker in ticker_names:
        # Construct the URL for current data
        url = f"https://iss.moex.com/iss/engines/stock/markets/shares/securities/{ticker}/candles.json?from={date_today}&interval=60"

        # Send request to retrieve current data
        response = rq.get(url)

        if response.status_code == 200:
            data = response.json()

            columns = data["candles"]['columns']
            data = data["candles"]["data"]

            df = pd.DataFrame(data, columns=columns)

            if not df.empty:
                # Construct the URL for historical data
                url = f"https://iss.moex.com/iss/engines/stock/markets/shares/securities/{ticker}/candles.json?from={datetime(2023, 6, 1).strftime('%Y-%m-%d')}&interval=60"

                # Send request to retrieve historical data
                response = rq.get(url)

                if response.status_code == 200:
                    data = response.json()

                    columns = data["candles"]['columns']
                    data = data["candles"]["data"]

                    df = pd.DataFrame(data, columns=columns)

                    # Convert 'begin' column to datetime and check the earliest date
                    df['begin'] = pd.to_datetime(df['begin'])
                    min_begin_date = df['begin'].min()

                    if min_begin_date <= datetime(2023, 6, 1, 10, 0, 0):
                        tickers.extend([ticker])
        else:
            print(f"Error: Unable to retrieve data for {ticker}")

    return tickers

def get_array_date() -> list:
    """
    Generate a list of dates starting from June 1st, 2023 and ending on October 23rd, 2024
    with a step of 30 days.

    Returns:
        list: A list of strings representing dates in the format 'YYYY-MM-DD HH:MM:SS'.
    """
    start_date = datetime(2023, 6, 1, 0, 0, 0)
    end_date = datetime(2024, 10, 23, 0, 0, 0)

    current_date = start_date
    array_date = []
    while current_date <= end_date:
        # Append the current date to the list
        array_date.append(current_date.strftime("%Y-%m-%d %H:%M:%S"))
        # Increment the date by 30 days
        current_date += timedelta(days=30)

    # Append the end date to the list
    array_date.append(current_date.strftime("%Y-%m-%d %H:%M:%S"))

    return array_date


def add_new_day(date1: str, date2: str, ticker: str):
    """
    Fetches and returns data for a specific stock ticker within a given date range.

    Args:
        date1 (str): Start date in the format 'YYYY-MM-DD'.
        date2 (str): End date in the format 'YYYY-MM-DD'.
        ticker (str): Stock ticker symbol.

    Returns:
        pd.DataFrame: DataFrame containing the fetched data.

    This function attempts to retrieve stock market data for a given ticker between
    specified dates from a remote server. If the request fails, it will retry until
    successful. On successful retrieval, it returns the data as a DataFrame.
    """
    url = f"https://iss.moex.com/iss/engines/stock/markets/shares/securities/{ticker}/candles.json?from={date1}&till={date2}&interval=60"

    while True:
        try:
            # Make a request to the server
            response = rq.get(url)
        except Exception as e:
            # Handle exceptions that occur during the request
            print(f"An error occurred: {e}")
        else:
            break

    # Check the response status
    if response.status_code != 200:
        print("Error fetching data")
        # Retry fetching data if unsuccessful
        add_new_day(date1, date2, ticker)

    # Create a DataFrame from the fetched data
    df_for_add = pd.DataFrame(response.json()["candles"]["data"], columns=response.json()["candles"]['columns'])
    df_for_add.reset_index().drop('index', axis=1)
    return df_for_add
 
def process_ticker(ticker: str, array_date: list):
    """
    This function processes data for a given ticker symbol.
    It checks if the file 'data_1h_moex/{ticker}_volume.csv' exists.
    If it does, it reads the data from the file and adds new data
    for each day in the array_date list, starting from the next day
    after the last date in the file.
    If the file does not exist, it creates the file and adds data
    for each day in the array_date list.
    :param ticker: The ticker symbol for the stock.
    :param array_date: A list of dates in the format 'YYYY-MM-DD'.
    :return: None
    """
    # Check if the file 'data_1h_moex/{ticker}_volume.csv' exists
    if Path(f'data_1h_moex/{ticker}_volume.csv').exists():
        print(f"File 'data_1h_moex/{ticker}_volume.csv' exists.")

        # Read data from the file
        df = pd.read_csv(f'{ticker}_volume.csv')

        # Convert the column 'begin' to datetime format
        df['begin'] = pd.to_datetime(df['begin'])

        # Find the index of the last date with data
        last_date_str = df['begin'].iloc[-1].strftime('%Y-%m-%d')
        
        if last_date_str in array_date:
            # Add new data for each day in the array_date list,
            # starting from the next day after the last date in the file
            df_array = [df]
            for i, date in enumerate(array_date):
                if i < len(array_date) - 1:
                    next_date = array_date[i + 1]
                    df_array += [add_new_day(date, next_date, ticker)]

            # Concatenate all DataFrames in df_array and reset the index
            df = pd.concat(df_array).reset_index(drop=True)

            # Drop duplicates
            df = df.drop_duplicates()

            # Save the updated data to the file
            df.to_csv(f'data_1h_moex/{ticker}_volume.csv', sep=',', index=False, encoding='utf-8')
    else:
        print(f"File 'data_1h_moex/{ticker}_volume.csv' does not exist.")
        df_array = []

        # Add data for each day in the array_date list
        for i, date in enumerate(array_date):
            if i < len(array_date) - 1:
                next_date = array_date[i + 1]
                df_array += [add_new_day(date, next_date, ticker)]

        # Concatenate all DataFrames in df_array and reset the index
        df_array = [df for df in df_array if not df.empty and not df.isnull().all().all()]
        df = pd.concat(df_array).reset_index(drop=True)

        # Convert the column 'begin' to datetime format
        df['begin'] = pd.to_datetime(df['begin'], errors='coerce')  

        # Drop rows where the time in the column 'begin' is 09:00
        df = df[df['begin'].dt.time != pd.Timestamp("09:00").time()]  

        # Drop duplicates
        df = df.drop_duplicates(subset=['begin'])

        # Save the data to the file 'data_1h_moex/{ticker}_volume.csv'
        filename = f'data_1h_moex/{ticker}_volume.csv'
        df.to_csv(filename, sep=',', index=False, encoding='utf-8')


def get_data():
    """
    Get data for all tickers from MOEX and save it to separate files for each hour.

    :return: None
    """
    tickers = get_tickers(get_ticker_names())

    # Create a ThreadPoolExecutor to speed up the process
    with ThreadPoolExecutor() as executor: 
        executor.map(process_ticker, tickers, get_array_date())

    # Iterate over the tickers and save the data to separate files for each hour
    for ticker in tickers:
        # Load data from the CSV file
        df = pd.read_csv(f'data_1h_moex/{ticker}_volume.csv')

        # Convert the column 'begin' to datetime format
        df['begin'] = pd.to_datetime(df['begin'])

        # Iterate over the hours of the day
        for hour in range(10, 24):
            # Filter the data by the current hour
            filtered_df = df[(df['begin'].dt.hour == hour)]
            
            # Save the data for each hour to a separate file
            filename = f"data_moex/{ticker}_{hour}.csv"
            filtered_df[['begin', 'volume', 'close']].to_csv(filename, index=False)
            print(f"Data for ticker {ticker} and hour {hour}:00 saved in file {filename}")

get_data()