import pandas as pd
import yfinance as yf


class LoadTickers:

    def __init__(self, ticker='^IBEX', period='max', weeks=12, debug=False):
        self.ticker = ticker
        self.period = period
        self.weeks = weeks
        self.debug = debug
        self.ticker_values = pd.DataFrame()
        self.display_values = pd.DataFrame()

    def load_tickers(self) -> pd.DataFrame:
        """
        Load historical values of the selected ticker and period
        """
        if self.debug:
            self.ticker_values = pd.read_csv('loading_module/ticker_values.csv')
            self.ticker_values.set_index('Date', inplace=True)
            self.ticker_values.index = pd.to_datetime(self.ticker_values.index, utc=True)
        else:
            self.ticker_values = yf.Tickers(self.ticker).tickers[self.ticker].history(period=self.period)

        self.ticker_values = self.ticker_values.resample('W-FRI').first().sort_values(by=['Date'])

    def get_display_table(self):
        self.display_values = self.ticker_values.tail(self.weeks)['Close'].to_frame().T

    def run_pipline(self):
        self.load_tickers()
        self.get_display_table()


if __name__ == '__main__':
    print('hello')

    ticker = '^IBEX'
    period = 'max'
    weeks = 12
    debug = True

    tick = LoadTickers(ticker=ticker, period=period, weeks=weeks, debug=debug)
    tick.run_pipline()

    print(tick.ticker_values)
    print('yay')
