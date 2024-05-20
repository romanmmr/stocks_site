# import os
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA

from loading_module.loading_module import LoadTickers


class Arima:
    def __init__(self, ticker: str, period: int, weeks: int, debug: bool):
        self.ticker = ticker
        self.period = period
        self.weeks = weeks
        self.debug = debug
        self.ticker_values = pd.DataFrame

    def load_values(self) -> None:
        values = LoadTickers(
            ticker=self.ticker,
            period=self.period,
            weeks=self.weeks,
            debug=self.debug
        )
        values.load_tickers()
        self.ticker_values = values.ticker_values
        # if self.debug:
        #     print(self.ticker_values)

if __name__ == '__main__':
    print('hello')

    ticker = '^IBEX'
    period = 'max'
    weeks = 12
    debug = True

    arima = Arima(ticker=ticker, period=period, weeks=weeks, debug=debug)
    arima.load_values()
    print(arima.ticker_values.head())

    # csv_test_path = 'C:\\Users\\rodri\\PycharmProjects\\stocks_website\\docs\\test_csv.csv'
    # ticker_values = pd.read_csv(csv_test_path)
    #
    # ticker_values.set_index('Date', inplace=True)
    #
    # print('done')