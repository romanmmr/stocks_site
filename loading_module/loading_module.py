import pandas as pd
import yfinance as yf


class LoadTickers:

    def __init__(self, ticker='^IBEX', period='max', display_weeks=12, update_data=False):
        self.ticker = ticker
        self.period = period
        self.display_weeks = display_weeks
        self.update_data = update_data
        self.ticker_values = pd.DataFrame()
        self.display_table = pd.DataFrame()

    def load_tickers(self) -> pd.DataFrame:
        """
        Load historical values of the selected ticker and period
        """
        if self.update_data:
            self.ticker_values = yf.Tickers(self.ticker).tickers[self.ticker].history(period=self.period)
            self.ticker_values.to_csv('loading_module/ticker_values.csv')
        else:
            self.ticker_values = pd.read_csv('loading_module/ticker_values.csv')
            self.ticker_values.set_index('Date', inplace=True)
            self.ticker_values.index = pd.to_datetime(self.ticker_values.index, utc=True)

        self.ticker_values = self.ticker_values.resample('W-FRI').first().sort_values(by=['Date'])

    def get_display_table(self):
        self.display_table = self.ticker_values.tail(self.display_weeks)['Close'].to_frame().T

    def run_pipline(self):
        self.load_tickers()
        self.get_display_table()


if __name__ == '__main__':
    print('hello')

    ticker = '^IBEX'
    period = 'max'
    display_weeks = 12
    debug = True

    tick = LoadTickers(ticker=ticker, period=period, display_weeks=display_weeks, update_data=debug)
    tick.run_pipline()

    print(tick.ticker_values)
    print('yay')
