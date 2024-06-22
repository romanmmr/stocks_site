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
            # self.ticker_values.index = pd.to_datetime(self.ticker_values.index, utc=True)
            self.ticker_values.index = pd.to_datetime(self.ticker_values.index)

        # Desired frequency
        freq = 'D'  # Daily frequency

        # Resample with missing value strategy ('ffill' for forward fill)
        self.ticker_values = self.ticker_values.resample(freq).fillna(method='ffill')

        # self.ticker_values['weekday'] = list(pd.Series(self.ticker_values.index).apply(lambda x: x.weekday()))
        # self.ticker_values = self.ticker_values[self.ticker_values['weekday'] == 4]
        # self.ticker_values = self.ticker_values[(self.ticker_values['weekday'] == 4) | (self.ticker_values['weekday'] == 3)]
        # self.ticker_values.drop(columns='weekday', inplace=True)

        # date_list = pd.Series(self.ticker_values.index).apply(lambda x: x.date())
        # date_list.diff()
        # date_list.diff().value_counts()

        self.ticker_values = self.ticker_values.resample('W-FRI').last().sort_values(by=['Date']).dropna()

        # self.modelling_data.freq = self.freq

    def get_display_table(self) -> None:
        self.display_table = self.ticker_values.tail(self.display_weeks)['Close'].to_frame().T

    def get_all_dates(self, df: pd.DataFrame, freq: str) -> None:

        # Define start and end dates
        start_date = df.index.min().date()
        end_date = df.index.max().date()

        # Create a list of days from start to end dates
        date_list = pd.date_range(start_date, end_date, inclusive='both')

        # Desired frequency
        freq = 'D'  # Daily frequency

        # Resample with missing value strategy ('ffill' for forward fill)
        df_resampled = df.resample(freq).fillna(method='ffill')

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
