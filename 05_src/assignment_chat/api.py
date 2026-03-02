from langchain.tools import tool
import requests
import os
from dotenv import load_dotenv
import json
import pandas_ta as ta
import pandas as pd


load_dotenv('../.secrets')

eod_url = "https://api.marketstack.com/v2/eod"

@tool
def get_stock_history_trend(symbols:list[str], datefrom:str, dateto:str):
    """
    An API call to a company end of day service provides the end of day stock price for a list of comma separated tickers for a given date range. 
    The tool should return a summary of the stock price trend for each ticker in the given date range. 
    The trend can be "upward", "downward" or "stable". The tool should also return the percentage change in stock price for each ticker in the given date range.
    The date range is limited to one year at present.
    It should also present an image which plots the trends in the stock price for the given date range. The x-axis of the plot should represent the date and the y-axis should represent the stock price. Each ticker should be represented by a different colored line in the plot.
    Accepted values for ticket are: AAPL, MSFT, GOOGL, AMZN, TSLA
    Accepted values for date are: Date in format (YYYY-MM-DD) OR "TODAY" OR "TOMORROW" OR "YESTERDAY".
    Accepted values for date are: Date in format (YYYY-MM-DD) OR "TODAY" OR "TOMORROW" OR "YESTERDAY".
    """
    # _logs.debug(f'Getting company rating for ticker {ticker}')
    print(f'Finding stock history for {symbols[0]} from {datefrom} to {dateto}' )
    response = get_stock_history_trend_from_service(symbols, datefrom, dateto)
    rating = get_stock_history_trend_from_response(symbols, response)
    # _logs.debug(f'Company rating result: {rating}')
    rating_df = perform_technical_analysis(rating)

    return rating_df



def get_stock_history_trend_from_service(symbols:list[str], datefrom:str, dateto:str):
    params = {
        "access_key": os.environ.get('MARKET_STACK_KEY'),
        "symbols": ",".join(symbols),
        "date_from": datefrom,
        "date_to": dateto

    }
    response = requests.get(eod_url, params=params)
    return response



def get_stock_history_trend_from_response(symbols:list[str], response:requests.Response) -> str:
    resp_dict = json.loads(response.text)
    return resp_dict


def perform_technical_analysis(data):
    """
    This tool takes the data get historical stock data and presents a technical analysis summary based on common technical indicators such as SMA, EMA, RSI, MACD, etc.
     The tool should return a summary of the technical analysis for the given stock data, including any
        significant trends or patterns identified by the indicators.
    """
    # Implement your technical analysis logic here
    df = pd.DataFrame(data['data'])
    # 2. Define technical analysis strategy using pandas_ta
    # Example: SMA (50, 200), RSI(14), MACD
    df.ta.sma(length=50, append=True)
    df.ta.sma(length=200, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.macd(append=True)
    
    # 3. Clean up NaN values created by indicators
    df.dropna(inplace=True)
    return df
