# import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA

from loading_module.loading_module import LoadTickers


class Arima:
    def __init__(
            self,
            ticker: str,
            period: int,
            order: tuple,
            display_weeks: int,
            test_weeks: int,
            freq: str,
            column: str,
            years: float,
            update_data: bool,
    ):
        self.ticker = ticker
        self.period = period
        self.display_weeks = display_weeks
        self.test_weeks = test_weeks
        self.freq = freq
        self.column = column
        self.log_column = f"log_{column}"
        self.years = years
        self.update_data = update_data
        self.ticker_values = pd.DataFrame
        self.modelling_data = pd.DataFrame
        self.train = pd.DataFrame
        self.test = pd.DataFrame
        self.train_index = None
        self.test_index = None
        self.order = order
        self.arima_model = None
        self.arima_log_model = None

    def load_values(self) -> None:
        values = LoadTickers(
            ticker=self.ticker,
            period=self.period,
            display_weeks=self.display_weeks,
            update_data=self.update_data
        )
        values.load_tickers()
        self.ticker_values = values.ticker_values

    def get_modelling_data(self) -> None:
        self.modelling_data = self.ticker_values[self.column][-52*self.years:].copy(deep=True).to_frame()

    def get_log_column(self) -> None:
        self.modelling_data[self.log_column] = np.log(self.modelling_data[self.column])

    def define_freq(self) -> None:
        self.modelling_data.freq = self.freq

    def define_train_test(self) -> None:
        self.train = self.modelling_data.iloc[:-self.test_weeks]
        self.test = self.modelling_data.iloc[-self.test_weeks:]

        self.train_index = self.modelling_data.index <= self.train.index[-1]
        self.test_index = self.modelling_data.index > self.train.index[-1]

    def run_preprocess(self) -> None:
        self.load_values()
        self.get_modelling_data()
        self.get_log_column()
        self.define_freq()
        self.define_train_test()

    def define_models(self) -> None:
        self.arima_model = ARIMA(self.train[self.column], order=self.order)
        self.arima_log_model = ARIMA(self.train[self.log_column], order=self.order)

    def fit_models(self) -> None:
        self.arima_result = self.arima_model.fit()
        self.arima_log_result = self.arima_log_model.fit()

    def predict_arima_models(self) -> None:
        print('wait')
        # TODO: think if we want to show model (ARIMA) and preditions over test set
        # TODO: Or maybe we just want to train the model and simply get real-life predictions and plot that



if __name__ == '__main__':

    ticker = '^IBEX'
    period = 'max'
    display_weeks = 12
    test_weeks = 8
    column = 'Close'
    years = 4
    update_data = False
    freq = 'W-FRI'
    order = (26, 1, 1)

    arima = Arima(
        ticker=ticker,
        period=period,
        order=order,
        display_weeks=display_weeks,
        test_weeks=test_weeks,
        freq=freq,
        column=column,
        years=years,
        update_data=update_data
    )
    arima.run_preprocess()
    arima.define_models()
    arima.fit_models()
    print(arima.ticker_values.head())

    print('done')