import pandas as pd
import yfinance as yf

class LoadTickers:

    def __init__(self, tickers='^IBEX', period='max', weeks=12):
        self.tickers = tickers
        self.period = period
        self.weeks = weeks
        self.ticker_values = pd.DataFrame()

    def load_tickers(self) -> pd.DataFrame:
        """
        Load historical values of the selected ticker and period
        """
        # if debug:
        #     self.ticker_values = pd.read_csv('ticker_values.csv')
        # else:
        self.ticker_values = yf.Tickers(self.tickers).tickers[self.tickers].history(period=self.period)

    def get_display_table(self):
        self.ticker_values = self.ticker_values.resample('W-FRI').first().reset_index().sort_values(by=['Date'])
        self.ticker_values = self.ticker_values.tail(self.weeks)[['Date', 'Close']].reset_index(drop=True).T

    def run_pipline(self):
        self.load_tickers()
        self.get_display_table()

if __name__ == '__main__':
    print('hello')

    ticker = '^IBEX'
    period = 'max'
    weeks = 12

    tick = LoadTickers(tickers=ticker, period=period, weeks=weeks)
    tick.run_pipline()

    print(tick.ticker_values)
    print('yay')
