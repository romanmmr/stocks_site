import yfinance as yf

class LoadTickers:

    def __init__(self, tickers='^IBEX', period='max'):
        self.tickers = tickers
        self.period = period
        self.ticker_values = None

    def load_tickers(self):
        """
        Load historical values of the selected ticker and period
        """
        self.ticker_values = yf.Tickers(self.tickers).tickers[self.tickers].history(period=self.period)


if __name__ == '__main__':
    print('hello')

    ticker = '^IBEX'
    period = 'max'

    tick = LoadTickers(tickers=ticker, period=period)
    tick.load_tickers()

    print(tick.ticker_values.columns)
