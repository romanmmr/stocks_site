# Third-party library imports
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Local application imports
from settings.settings import load_config
from loading_module.loading_module import LoadTickers


class Arima:
    """
    This class performs ARIMA modelling on a given ticker's historical data.

    It allows for defining ARIMA model parameters, loading data, performing preprocessing
    steps, fitting models, testing on unseen data, and saving the results.
    """

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
        """
        Initializes the Arima object with configuration parameters.

        Args:
            ticker (str): Ticker symbol of the stock to analyze.
            period (int): Number of weeks of data to load.
            order (tuple): ARIMA model order (p, d, q).
            display_weeks (int): Number of weeks to display for visualization.
            test_weeks (int): Number of weeks to use for testing the model.
            freq (str): Frequency of the data (e.g., 'W-MON').
            column (str): Name of the column to use for modelling (e.g., 'Close').
            years (float): Number of years of data to consider for modelling.
            update_data (bool): Flag to update data if it already exists.
        """

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
        self.log_cols = None
        self.not_log_cols = None

    def load_values(self) -> None:
        """
        Loads historical data for the specified ticker using the LoadTickers class.

        Populates the `ticker_values` attribute with the loaded DataFrame.
        """

        values = LoadTickers(
            ticker=self.ticker,
            period=self.period,
            display_weeks=self.display_weeks,
            update_data=self.update_data
        )
        values.load_tickers()
        self.ticker_values = values.ticker_values

    def get_modelling_data(self) -> None:
        """
        Extracts the desired timeframe of data for modelling from `ticker_values`.

        Creates a copy of the last `years * 52` weeks of data for the specified column
        and assigns it to the `modelling_data` attribute.
        """

        self.modelling_data = self.ticker_values[self.column][-52*self.years:].copy(deep=True).to_frame()

    def get_log_column(self) -> None:
        """
        Creates a new column named 'log_{column}' containing the natural log of the data.

        Adds the log-transformed column to the `modelling_data` DataFrame.
        """

        self.modelling_data[self.log_column] = np.log(self.modelling_data[self.column])

    def define_freq(self) -> None:
        """
        Sets the frequency attribute of the `modelling_data` DataFrame to the specified value.

        This ensures proper handling of time-series data during modelling.
        """

        self.modelling_data.freq = self.freq

    def define_train_test(self) -> None:
        """
        Splits the `modelling_data` DataFrame into training and testing sets.

        Assigns the training and testing data to the `train` and `test` attributes, respectively.
        Also defines indices for training and testing periods.
        """

        self.train = self.modelling_data.iloc[:-self.test_weeks]
        self.test = self.modelling_data.iloc[-self.test_weeks:]

        self.train_index = self.modelling_data.index <= self.train.index[-1]
        self.test_index = self.modelling_data.index > self.train.index[-1]

    def run_preprocess(self) -> None:
        """
        Executes the pre-processing pipeline for ARIMA modelling.

        Calls functions in sequence to load data, select modelling data, create log-transformed data,
        set frequency, and define training/testing splits.
        """

        self.load_values()
        self.get_modelling_data()
        self.get_log_column()
        self.define_freq()
        self.define_train_test()

    def define_models(self) -> None:
        """
        Defines ARIMA models for both the original data and the log-transformed data.

        Initializes `arima_model` and `arima_log_model` attributes using the specified ARIMA order.
        """

        self.arima_model = ARIMA(self.train[self.column], order=self.order)
        self.arima_log_model = ARIMA(self.train[self.log_column], order=self.order)

    def fit_models(self) -> None:
        """
        Fits the ARIMA models to the training data.

        Calls the `fit` method on both `arima_model` and `arima_log_model` using their respective training sets.
        Stores the fitted models in `arima_result` and `arima_log_result` attributes.
        """

        self.arima_result = self.arima_model.fit()
        self.arima_log_result = self.arima_log_model.fit()

    def test_arima_models(self) -> None:
        """
        Tests the fitted ARIMA models on the hold-out test data.

        Performs the following steps for both the original and log-transformed ARIMA models:
            - Generates in-sample predictions for the training data and populates the 'arima_output' column in `modelling_data`.
            - Generates out-of-sample predictions (forecasts) for the test data and populates the 'arima_output' column in `modelling_data`.
            - Calculates confidence intervals for the forecasts on the test data and populates the 'arima_conf_int_lower' and 'arima_conf_int_upper' columns.
        """

        # Get predictions for training data for arima model and populate modelling_data dataframe
        self.modelling_data.loc[self.train_index, 'arima_output'] = self.arima_result.predict(
            start=self.train.index[0],
            end=self.train.index[-1]
        )
        # Get predictions for testing data for arima model and populate modelling_data dataframe
        prediction_result = self.arima_result.get_forecast(self.test_weeks)
        forecast = prediction_result.predicted_mean
        self.modelling_data.loc[self.test_index, 'arima_output'] = forecast
        # Get confident intervals for arima model and populate modelling_data dataframe
        conf_int = prediction_result.conf_int()
        lower, upper = conf_int[f"lower {self.column}"], conf_int[f"upper {self.column}"]
        self.modelling_data.loc[self.test_index, 'arima_conf_int_lower'] = lower
        self.modelling_data.loc[self.test_index, 'arima_conf_int_upper'] = upper


        # Get predictions for training data for arima log model and populate modelling_data dataframe
        self.modelling_data.loc[self.train_index, 'arima_log_output'] = self.arima_log_result.predict(
            start=self.train.index[0],
            end=self.train.index[-1]
        )
        # Get predictions for testing data for arima log model and populate modelling_data dataframe
        log_prediction_result = self.arima_log_result.get_forecast(self.test_weeks)
        log_forecast = log_prediction_result.predicted_mean
        self.modelling_data.loc[self.test_index, 'arima_log_output'] = log_forecast
        # Get confident intervals for arima log model and populate modelling_data dataframe
        log_conf_int = log_prediction_result.conf_int()
        log_lower, log_upper = log_conf_int[f"lower {self.log_column}"], log_conf_int[f"upper {self.log_column}"]
        self.modelling_data.loc[self.test_index, 'arima_log_conf_int_lower'] = log_lower
        self.modelling_data.loc[self.test_index, 'arima_log_conf_int_upper'] = log_upper

    def get_log_cols(self) -> None:
        """
        Identifies columns containing log-transformed data in the modelling_data DataFrame.

        - Creates a list named `log_cols` containing all column names with 'log' in them (indicating log-transformed data).
        - Creates a list named `not_log_cols` containing all remaining columns (original data).
        """

        self.log_cols = [col for col in self.modelling_data.columns if 'log' in col]
        self.not_log_cols = [col for col in self.modelling_data.columns if 'log' not in col]

    def reverse_logs(self) -> None:
        """
        Reverses the log transformation on the identified log columns in modelling_data.

        Iterates through the `log_cols` list and applies the exponential function (np.exp) to each column
        to convert the log-transformed values back to their original scale.
        """

        for column in self.log_cols:
            self.modelling_data[column] = np.exp(self.modelling_data[column])

    def save_data(self) -> None:
        """
        Saves the modelling_data DataFrame to a CSV file for further analysis or visualization.

        Saves the `modelling_data` DataFrame containing the original data, log-transformed data,
        predictions, and confidence intervals to a CSV file named 'docs/arima_fitted_data.csv' by default.
        """

        self.modelling_data.to_csv('docs/arima_fitted_data.csv')

    def run_modelling(self) -> None:
        """
        Executes the core modelling steps of the ARIMA analysis.

        Calls the following functions in sequence:
            - `define_models` to create ARIMA models for original and log-transformed data.
            - `fit_models` to fit the models to the training data.
            - `test_arima_models` to generate predictions and confidence intervals on the test data.
            - `get_log_cols` to identify log-transformed columns.
            - `reverse_logs` to convert log-transformed predictions back to the original scale.
        """

        self.define_models()
        self.fit_models()
        self.test_arima_models()
        self.get_log_cols()
        self.reverse_logs()

    def run_pipeline(self) -> None:
        """
        Executes the entire ARIMA modelling pipeline in sequence.

        Calls the following functions in sequence:
            - `run_preprocess` to load data, select modelling data, and define train/test splits.
            - `run_modelling` to perform model definition, fitting, testing, and data transformation.
            - `save_data` to save the final modelling data for later use.
        """

        self.run_preprocess()
        self.run_modelling()
        self.save_data()


if __name__ == '__main__':

    config = load_config("config/config.yaml")

    ticker = config['config']['ticker']
    period = config['config']['period']
    display_weeks = config['config']['display_weeks']
    test_weeks = config['config']['test_weeks']
    column = config['config']['column']
    years = config['config']['years']
    update_data = config['config']['update_data']
    freq = config['config']['freq']
    order = tuple(config['config']['order'])

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

    arima.run_pipeline()