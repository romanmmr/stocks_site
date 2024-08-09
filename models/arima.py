# Third-party library imports
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

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
            column_list: list,
            years: float,
            update_data: bool,
            train_arima: bool,
            train_nn: bool,
            weeks_nn: int,
            alpha_ci: float,
            order_gridsearch: list,
            neurons_nn: list,
            regularization_nn: float,
            regularization_crnn: float,
            train_val_partition: float,
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
        self.column_list = column_list
        self.years = years
        self.update_data = update_data
        self.train_arima = train_arima
        self.train_nn = train_nn
        self.weeks_nn = weeks_nn
        self.alpha_ci = alpha_ci
        self.order_gridsearch = order_gridsearch
        self.neurons_nn = neurons_nn
        self.regularization_nn = regularization_nn
        self.regularization_crnn = regularization_crnn
        self.train_val_partition = train_val_partition
        self.ticker_values = pd.DataFrame
        self.modelling_data = pd.DataFrame
        self.future_weeks = pd.DataFrame
        self.train_arima_df = pd.DataFrame
        self.test_arima_df = pd.DataFrame
        self.train_index = None
        self.test_index = None
        self.existing_dates = None
        self.future_dates = None
        self.order = order
        self.arima_training_model = None
        self.arima_training_log_model = None
        self.arima_training_result = None
        self.arima_training_log_result = None
        self.arima_predictive_model = None
        self.arima_predictive_log_model = None
        self.arima_predictive_result = None
        self.arima_predictive_log_result = None
        self.log_training_cols = None
        self.not_log_training_cols = None
        self.log_predictive_cols = None
        self.not_log_predictive_cols = None
        self.nn_df = None
        self.X_train_nn = None
        self.Y_train_nn = None
        self.X_val_nn = None
        self.Y_val_nn = None
        self.X_test_nn = None
        self.Y_test_nn = None
        self.nn_model = None
        self.crnn_model = None
        self.scaler = None
        self.nn_model_ci = None
        self.crnn_model_ci = None
        self.history_nn_model = None
        self.history_crnn_model = None
        self.nn_model_checkpoint_path = 'nn_checkpoint/best_model.weights.h5'
        self.crnn_model_checkpoint_path = 'crnn_checkpoint/best_model.weights.h5'
        self.nn_model_history_path = 'nn_checkpoint/history.csv'
        self.crnn_model_history_path = 'crnn_checkpoint/history.csv'
        self.nn_model_name = 'nn_model'
        self.crnn_model_name = 'crnn_model'
        self.arima_model_name = 'arima'
        self.model_names = [self.arima_model_name, self.nn_model_name, self.crnn_model_name]
        self.log_column = f"log_{column}_{self.arima_model_name}"
        self.diff_log_column = f"diff_log_{column}_{self.nn_model_name}"
        self.scaled_diff_log_column = f"scaled_diff_log_{column}_{self.nn_model_name}"
        self.relative_path_to_modelling_data = 'docs/modelled_fitted_data.csv'
        self.relative_path_to_predictive_data = 'docs/predictive_data.csv'

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

        # self.modelling_data = self.ticker_values[self.column][int(-52*self.years):].copy(deep=True).to_frame()
        self.modelling_data = self.ticker_values.iloc[int(-52 * self.years):][self.column_list].copy(deep=True)

    def copy_target_columns_for_modelling(self) -> None:
        for model in self.model_names:
            self.modelling_data[f"{self.column}_{model}"] = self.modelling_data[self.column].copy()

    def get_log_column(self) -> None:
        """
        Creates a new column named 'log_{column}' containing the natural log of the data.

        Adds the log-transformed column to the `modelling_data` DataFrame.
        """

        self.modelling_data[self.log_column] = np.log(self.modelling_data[f"{self.column}_{self.arima_model_name}"])

    def get_diff_log_column(self) -> None:
        """
        Creates a new column named 'diff_{log_column}' containing the differenced natural log of the data.

        Adds the diff-log-transformed column to the `modelling_data` DataFrame.
        """
        self.modelling_data[self.diff_log_column] = self.modelling_data[self.log_column].diff()

    def get_scaled_diff_log_column(self) -> None:
        self.scaler = StandardScaler()
        self.modelling_data[self.scaled_diff_log_column] = self.scaler.fit_transform(
            self.modelling_data[[self.diff_log_column]]
        ).flatten()

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

        self.train_arima_df = self.modelling_data.iloc[:-self.test_weeks]
        self.test_arima_df = self.modelling_data.iloc[-self.test_weeks:]

        self.train_index = self.modelling_data.index <= self.train_arima_df.index[-1]
        self.test_index = self.modelling_data.index > self.train_arima_df.index[-1]

    def create_nn_dataset(self):
        series = self.modelling_data[self.scaled_diff_log_column].dropna().to_numpy()

        T = self.weeks_nn
        X = []
        Y = []
        for t in range(len(series) - T):
            x = series[t:t + T]
            X.append(x)
            y = series[t + T]
            Y.append(y)

        X = np.array(X).reshape(-1, T)
        Y = np.array(Y)

        # Create dataframe with X and Y to later retrieve values
        self.nn_df = pd.DataFrame(X, columns=[f"feature_{feat}" for feat in range(X.shape[1])])
        self.nn_df['target'] = Y

        # Define test datasets as numpy objects
        # self.X_test_nn_1, self.Y_test_nn_1 = X[-self.test_weeks:], Y[-self.test_weeks:]

        # Define test datasets as pandas objects (same as previous but as pandas instead of numpy)
        self.X_test_nn = self.nn_df.tail(self.test_weeks)[
            [f"feature_{feat}" for feat in range(X.shape[1])]
        ].copy(deep=True)
        self.Y_test_nn = self.nn_df.tail(self.test_weeks)['target'].copy(deep=True)

        # Get randomized indexes for randomization of train and validation sets
        randomized_index = np.random.RandomState(0).choice(
            len(X[:-self.test_weeks]),
            size=len(X[:-self.test_weeks]),
            replace=False
        )

        # Get train and validation sizes
        train_size = int(len(X[:-self.test_weeks]) * self.train_val_partition)
        validation_size = len(X[:-self.test_weeks]) - train_size

        # Test datasets are created, let's create train and validation randomized datasets
        # First let's get the randomized train and validation indexes
        rand_train_index = randomized_index[:train_size]
        rand_val_index = randomized_index[train_size:]

        # Let's now get the train and validation datasets
        # self.X_train_nn, self.Y_train_nn = X[[rand_train_index]].reshape(-1, T), Y[[rand_train_index]].reshape(-1)
        # self.X_val_nn, self.Y_val_nn = X[[rand_val_index]].reshape(-1, T), Y[[rand_val_index]].reshape(-1)

        # Same as previous but as pandas dt instead of numpy object
        self.X_train_nn = self.nn_df.loc[
            list(rand_train_index),
            [f"feature_{feat}" for feat in range(X.shape[1])]
        ].copy(deep=True)
        self.Y_train_nn = self.nn_df.loc[list(rand_train_index), 'target'].copy(deep=True)
        self.X_val_nn = self.nn_df.loc[
            list(rand_val_index),
            [f"feature_{feat}" for feat in range(X.shape[1])]
        ].copy(deep=True)
        self.Y_val_nn = self.nn_df.loc[list(rand_val_index), 'target']


        print(f"X_train_nn shape: {self.X_train_nn.shape} --- Y_train_nn shape: {self.Y_train_nn.shape}")
        print(f"X_val_nn shape: {self.X_val_nn.shape} --- Y_val_nn shape: {self.Y_val_nn.shape}")
        print(f"X_test_nn shape: {self.X_test_nn.shape} --- Y_test_nn shape: {self.Y_test_nn.shape}")


    def get_future_weeks(self):
        self.future_weeks = self.modelling_data.copy(deep=True)

        # Generate a range of timestamps starting from the existing index with 7-day intervals
        self.future_dates = pd.Series(
            pd.date_range(
                start=self.future_weeks.index[-1],
                periods=self.test_weeks + 1,
                freq='7D'
            )[1:]
        )

        # Generate all index list including future weeks
        self.existing_dates = pd.Series(self.future_weeks.index)

        # Reindex the DataFrame with the new index (including existing and future timestamps)
        self.future_weeks = self.future_weeks.reindex(
            pd.concat([self.existing_dates, self.future_dates], ignore_index=True)
        ).reset_index().rename(columns={'index': 'Date'})

        self.future_weeks.set_index('Date', inplace=True)

    def run_preprocess(self) -> None:
        """
        Executes the pre-processing pipeline for ARIMA modelling.

        Calls functions in sequence to load data, select modelling data, create log-transformed data,
        set frequency, and define training/testing splits.
        """

        self.load_values()
        self.get_modelling_data()
        self.copy_target_columns_for_modelling()
        self.get_log_column()
        self.get_diff_log_column()
        self.get_scaled_diff_log_column()
        self.define_freq()
        self.define_train_test()
        self.create_nn_dataset()
        self.get_future_weeks()

    def train_arima_best_order(self) -> None:
        """
        Defines ARIMA models for both the original data and the log-transformed data.

        Initializes `arima_model` and `arima_log_model` attributes using the specified ARIMA order.
        """

        best_loss = None
        best_order = None

        for index, order in enumerate(product(
                range(self.order_gridsearch[0][0], self.order_gridsearch[0][1]),
                range(self.order_gridsearch[1][0], self.order_gridsearch[1][1]),
                range(self.order_gridsearch[2][0], self.order_gridsearch[2][1])
        )):
        # for order in product(
        #         range(self.order_gridsearch[0]), range(self.order_gridsearch[1]), range(self.order_gridsearch[2])
        # ):
        # for order in product(range(27), range(3), range(11)):
            print(order)
            arima_training_log_model = ARIMA(self.train_arima_df[self.log_column], order=order)
            arima_training_log_result = arima_training_log_model.fit()
            prediction_result = arima_training_log_result.get_forecast(self.test_weeks)
            forecast = prediction_result.predicted_mean
            loss = self.rmse(self.test_arima_df[self.log_column], forecast)
            # best_loss = None
            # best_order = None
            if index == 0:
                best_order = order
                best_loss = loss
            if loss < best_loss:
                best_order = order
                best_loss = loss

        print(f"Best order: {best_order}")
        self.order = best_order

    def define_arima_training_models(self) -> None:
        """
        Defines ARIMA models for both the original data and the log-transformed data.

        Initializes `arima_model` and `arima_log_model` attributes using the specified ARIMA order.
        """

        # self.arima_training_model = ARIMA(self.train[self.column], order=self.order)
        self.arima_training_log_model = ARIMA(self.train_arima_df[self.log_column], order=self.order)

    def rmse(self, y_true, y_pred):
        """
        Calculates the Root Mean Squared Error (RMSE) between two NumPy arrays.

        Args:
            y_true (np.ndarray): The ground truth values.
            y_pred (np.ndarray): The predicted values.

        Returns:
            float: The RMSE value.
        """

        # Calculate squared errors
        squared_errors = np.square(y_true - y_pred)

        # Mean of squared errors
        mean_squared_error = np.mean(squared_errors)

        # Take the square root
        rmse = np.sqrt(mean_squared_error)

        return rmse

    def define_arima_predictive_models(self) -> None:
        """
        Defines ARIMA models for both the original data and the log-transformed data.

        Initializes `arima_model` and `arima_log_model` attributes using the specified ARIMA order.
        """

        # self.arima_predictive_model = ARIMA(self.modelling_data[self.column], order=self.order)
        self.arima_predictive_log_model = ARIMA(self.modelling_data[self.log_column], order=self.order)

    def fit_arima_training_models(self) -> None:
        """
        Fits the ARIMA models to the training data.

        Calls the `fit` method on both `arima_model` and `arima_log_model` using their respective training sets.
        Stores the fitted models in `arima_result` and `arima_log_result` attributes.
        """

        # self.arima_training_result = self.arima_training_model.fit()
        self.arima_training_log_result = self.arima_training_log_model.fit()

    def fit_arima_predictive_models(self) -> None:
        """
        Fits the ARIMA models to the training data.

        Calls the `fit` method on both `arima_model` and `arima_log_model` using their respective training sets.
        Stores the fitted models in `arima_result` and `arima_log_result` attributes.
        """

        # self.arima_predictive_result = self.arima_predictive_model.fit()
        self.arima_predictive_log_result = self.arima_predictive_log_model.fit()

    def test_arima_models(self) -> None:
        """
        Tests the fitted ARIMA models on the hold-out test data.

        Performs the following steps for both the original and log-transformed ARIMA models:
            - Generates in-sample predictions for the training data and populates the 'arima_output' column in `modelling_data`.
            - Generates out-of-sample predictions (forecasts) for the test data and populates the 'arima_output' column in `modelling_data`.
            - Calculates confidence intervals for the forecasts on the test data and populates the 'arima_conf_int_lower' and 'arima_conf_int_upper' columns.
        """

        # # Get predictions for training data for arima model and populate modelling_data dataframe
        # self.modelling_data.loc[self.train_index, 'arima_output'] = self.arima_training_result.predict(
        #     start=self.train.index[0],
        #     end=self.train.index[-1]
        # )
        # # Get predictions for testing data for arima model and populate modelling_data dataframe
        # prediction_result = self.arima_training_result.get_forecast(self.test_weeks)
        # forecast = prediction_result.predicted_mean
        # self.modelling_data.loc[self.test_index, 'arima_output'] = forecast
        # # Get confident intervals for arima model and populate modelling_data dataframe
        # conf_int = prediction_result.conf_int()
        # lower, upper = conf_int[f"lower {self.column}"], conf_int[f"upper {self.column}"]
        # self.modelling_data.loc[self.test_index, 'arima_conf_int_lower'] = lower
        # self.modelling_data.loc[self.test_index, 'arima_conf_int_upper'] = upper


        # Get predictions for training data for arima log model and populate modelling_data dataframe
        self.modelling_data.loc[
            self.train_index, f"{self.arima_model_name}_output"
        ] = np.exp(self.arima_training_log_result.predict(start=self.train_arima_df.index[0], end=self.train_arima_df.index[-1]))
        # Get predictions for testing data for arima log model and populate modelling_data dataframe
        log_prediction_result = self.arima_training_log_result.get_forecast(self.test_weeks)
        log_forecast = log_prediction_result.predicted_mean
        self.modelling_data.loc[self.test_index, f"{self.arima_model_name}_output"] = np.exp(log_forecast)
        # Get confident intervals for arima log model and populate modelling_data dataframe
        log_conf_int = log_prediction_result.conf_int()
        lower = np.exp(log_conf_int[f"lower {self.log_column}"])
        upper = np.exp(log_conf_int[f"upper {self.log_column}"])
        self.modelling_data.loc[self.test_index, f"{self.arima_model_name}_conf_int_lower"] = lower
        self.modelling_data.loc[self.test_index, f"{self.arima_model_name}_conf_int_upper"] = upper

    def get_arima_real_predictions(self) -> None:
        """
        Tests the fitted ARIMA models on the hold-out test data.

        Performs the following steps for both the original and log-transformed ARIMA models:
            - Generates in-sample predictions for the training data and populates the 'arima_output' column in `modelling_data`.
            - Generates out-of-sample predictions (forecasts) for the test data and populates the 'arima_output' column in `modelling_data`.
            - Calculates confidence intervals for the forecasts on the test data and populates the 'arima_conf_int_lower' and 'arima_conf_int_upper' columns.
        """

        # # Get predictions for training data for arima model and populate modelling_data dataframe
        # self.future_weeks.loc[self.existing_dates, 'arima_output'] = self.arima_predictive_result.predict(
        #     start=self.modelling_data.index[0],
        #     end=self.modelling_data.index[-1]
        # )
        # # Get predictions for testing data for arima model and populate modelling_data dataframe
        # prediction_result = self.arima_predictive_result.get_forecast(self.test_weeks)
        # forecast = prediction_result.predicted_mean
        # self.future_weeks.loc[self.future_weeks.index[-self.test_weeks:], 'arima_output'] = forecast
        # # Get confident intervals for arima model and populate modelling_data dataframe
        # conf_int = prediction_result.conf_int()
        # lower, upper = conf_int[f"lower {self.column}"], conf_int[f"upper {self.column}"]
        # self.future_weeks.loc[self.future_weeks.index[-self.test_weeks:], 'arima_conf_int_lower'] = lower
        # self.future_weeks.loc[self.future_weeks.index[-self.test_weeks:], 'arima_conf_int_upper'] = upper


        # Get predictions for training data for arima log model and populate modelling_data dataframe
        self.future_weeks.loc[
            self.modelling_data.index, f"{self.arima_model_name}_output"
        ] = np.exp(
            self.arima_predictive_log_result.predict(
                start=self.modelling_data.index[0], end=self.modelling_data.index[-1]
            )
        )

        # Get predictions for testing data for arima log model and populate modelling_data dataframe
        log_prediction_result = self.arima_predictive_log_result.get_forecast(self.test_weeks)
        log_forecast = log_prediction_result.predicted_mean
        self.future_weeks.loc[
            self.future_weeks.index[-self.test_weeks:], f"{self.arima_model_name}_output"
        ] = np.exp(log_forecast)
        self.future_weeks.loc[
            self.future_weeks.index[-self.test_weeks:], f"{self.column}_{self.arima_model_name}"
        ] = np.exp(log_forecast)
        # Get confident intervals for arima log model and populate modelling_data dataframe
        log_conf_int = log_prediction_result.conf_int()
        lower = np.exp(log_conf_int[f"lower {self.log_column}"])
        upper = np.exp(log_conf_int[f"upper {self.log_column}"])

        self.future_weeks.loc[
            self.future_weeks.index[-self.test_weeks:], f"{self.arima_model_name}_conf_int_lower"
        ] = lower
        self.future_weeks.loc[
            self.future_weeks.index[-self.test_weeks:], f"{self.arima_model_name}_conf_int_upper"
        ] = upper

    def reset_random_seeds(self) -> None:
        os.environ['PYTHONHASHSEED'] = str(0)
        tf.random.set_seed(0)
        np.random.seed(0)
        random.seed(0)

    def define_nn_training_models(self) -> None:
        seed = 0
        # neurons_1 = 50
        # neurons_2 = 30
        # neurons_3 = 15
        # neurons_4 = 20
        # regularization = 0.03

        neurons_1 = self.neurons_nn[0]
        neurons_2 = self.neurons_nn[1]
        neurons_3 = self.neurons_nn[2]
        neurons_4 = self.neurons_nn[3]
        regularization = self.regularization_nn

        # Define the input layer
        inputs = tf.keras.Input(shape=(self.X_train_nn.shape[1],))

        # Add the first hidden layer with L2 regularization
        x = tf.keras.layers.Dense(
            neurons_1, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(regularization)
        )(inputs)

        # Add Batch Normalization
        x = tf.keras.layers.BatchNormalization()(x)

        # Add hidden layer with L2 regularization
        x = tf.keras.layers.Dense(
            neurons_2, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(regularization)
        )(x)

        # Add Batch Normalization
        x = tf.keras.layers.BatchNormalization()(x)

        # Add hidden layer with L2 regularization
        x = tf.keras.layers.Dense(
            neurons_3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(regularization)
        )(x)

        # Add Batch Normalization
        x = tf.keras.layers.BatchNormalization()(x)

        # Add hidden layer with L2 regularization
        x = tf.keras.layers.Dense(
            neurons_4, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(regularization)
        )(x)

        # Add Batch Normalization
        x = tf.keras.layers.BatchNormalization()(x)

        # Output layer
        outputs = tf.keras.layers.Dense(1)(x)

        # Create the model
        self.nn_model = tf.keras.Model(inputs=inputs, outputs=outputs)

    def compile_and_fit_nn_models(self) -> None:
        self.reset_random_seeds()
        tf.keras.backend.clear_session()
        batch_size = 64
        # epochs = 400
        epochs = 1000
        self.nn_model.compile(
            #   loss='mse',
            loss='mae',
            #   loss = [tf.keras.metrics.RootMeanSquaredError()],
            optimizer='adam',
        )

        checkpoint_filepath = os.path.join(os.getcwd(), self.nn_model_checkpoint_path)
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            # monitor='val_root_mean_squared_error',
            monitor='val_loss',
            mode='min',
            save_best_only=True
        )

        self.history_nn_model = self.nn_model.fit(
            self.X_train_nn,
            self.Y_train_nn,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(self.X_val_nn, self.Y_val_nn),
            callbacks=[model_checkpoint_callback],
            verbose=2
        )

        pd.DataFrame(self.history_nn_model.history).to_csv(self.nn_model_history_path)

        print(f"NN Model at Epoch 1000 - MAE: {self.nn_model.evaluate(self.X_test_nn, self.Y_test_nn)}")

        # plot_loss = self.history_nn_model.history['loss']
        # plot_val_loss = self.history_nn_model.history['val_loss']
        # plot_epochs = range(len(plot_loss))
        #
        # fig_mae = plt.figure(figsize=(10, 5))
        # plt.plot(plot_epochs, plot_loss, 'b', label='Training loss (mae)')
        # plt.plot(plot_epochs, plot_val_loss, 'r', label='Validation loss (mae)')
        # plt.title('Training and Validation loss (mae)')
        # plt.grid()
        # plt.legend()

    def load_nn_model_weights(self) -> None:
        checkpoint_filepath = os.path.join(os.getcwd(), self.nn_model_checkpoint_path)
        self.nn_model.load_weights(checkpoint_filepath)

        print(f"NN Best model - MAE: {self.nn_model.evaluate(self.X_test_nn, self.Y_test_nn)}")

    def test_nn_models(self) -> None:

        # first T+1 values are not predictable
        train_idx_nn = self.train_index.copy()
        train_idx_nn[:self.weeks_nn + 1] = False

        # Ptrain = self.scaler.inverse_transform(self.nn_model.predict(self.X_train_nn)).flatten()
        # Ptest = self.scaler.inverse_transform(self.nn_model.predict(self.X_test_nn)).flatten()
        Ptrain = self.scaler.inverse_transform(self.nn_model.predict(
            self.nn_df.loc[
                :len(self.nn_df) - self.test_weeks - 1,
                [f"feature_{feat}" for feat in range(self.nn_df.shape[1] - 1)]
            ]
        )).flatten()

        prev = self.modelling_data[self.log_column].shift(1)

        # Last-known train value
        last_train = self.train_arima_df.iloc[-1][self.log_column]

        # multi-step forecast
        multistep_predictions = []

        # first test input
        # last_x = self.X_test_nn_1[0]
        # last_x = self.X_test_nn.iloc[0].to_numpy()
        last_x = self.X_test_nn.iloc[[0]]

        while len(multistep_predictions) < self.test_weeks:
            # p = self.nn_model.predict(last_x.reshape(1, -1))[0]
            # p = self.nn_model.predict(last_x.reshape(1, self.weeks_nn))[0]
            p = self.nn_model.predict(last_x)[0]

            # update the predictions list
            multistep_predictions.append(p)

            # make the new input
            last_x = np.roll(last_x, -1)
            last_x[-1] = p

        # unscale
        multistep_predictions = np.array(multistep_predictions)
        multistep_predictions = self.scaler.inverse_transform(multistep_predictions.reshape(-1, 1)).flatten()

        # save multi-step forecast to dataframe
        self.modelling_data.loc[train_idx_nn, f"{self.nn_model_name}_output"] = np.exp(prev[train_idx_nn] + Ptrain)
        self.modelling_data.loc[
            self.test_index, f"{self.nn_model_name}_output"
        ] = np.exp(last_train + np.cumsum(multistep_predictions))

        residuals = (self.modelling_data.loc[train_idx_nn, f"{self.column}_{self.nn_model_name}"] -
                     self.modelling_data.loc[train_idx_nn, f"{self.nn_model_name}_output"])

        self.nn_model_ci = np.quantile(residuals, 1 - self.alpha_ci)

        ci_test_weeks = []

        for index, interval in enumerate(self.modelling_data.loc[self.test_index, f"{self.nn_model_name}_output"]):
            ci_test_weeks.append(self.nn_model_ci * (index + 1))

        self.modelling_data.loc[
            self.test_index, f"{self.nn_model_name}_conf_int_lower"
        ] = self.modelling_data.loc[self.test_index, f"{self.nn_model_name}_output"] - ci_test_weeks

        self.modelling_data.loc[
            self.test_index, f"{self.nn_model_name}_conf_int_upper"
        ] = self.modelling_data.loc[self.test_index, f"{self.nn_model_name}_output"] + ci_test_weeks

        print('nn models done')

        # plot 1-step and multi-step forecast
        # self.modelling_data.iloc[-16:][['log_Close', 'multistep', '1step_test']].plot(figsize=(15, 5))

    def get_nn_real_predictions(self) -> None:
        # first T+1 values are not predictable
        train_idx_nn = self.train_index.copy()
        train_idx_nn[:self.weeks_nn + 1] = False

        # Last-known train value
        last_test = self.test_arima_df.iloc[-1][self.log_column]

        # multi-step forecast
        multistep_predictions = []

        # first test input
        # last_x = self.X_test_nn[-1]
        last_x = self.X_test_nn.iloc[[-1]]

        while len(multistep_predictions) < self.test_weeks:
            # p = self.nn_model.predict(last_x.reshape(1, -1))[0]
            p = self.nn_model.predict(last_x)[0]

            # update the predictions list
            multistep_predictions.append(p)

            # make the new input
            last_x = np.roll(last_x, -1)
            last_x[-1] = p

        # unscale
        multistep_predictions = np.array(multistep_predictions)
        multistep_predictions = self.scaler.inverse_transform(multistep_predictions.reshape(-1, 1)).flatten()

        # save multi-step forecast to modelling dataframe
        # self.modelling_data.loc[train_idx_nn, f"{self.nn_model_name}_output"] = np.exp(prev[train_idx_nn] + Ptrain)
        # self.modelling_data.loc[
        #     self.test_index, f"{self.nn_model_name}_output"
        # ] = np.exp(last_train + np.cumsum(multistep_predictions))

        # save multi-step forecast to future weeks dataframe
        self.future_weeks.loc[
            self.future_weeks.index[-self.test_weeks:], f"{self.column}_{self.nn_model_name}"
        ] = np.exp(last_test + np.cumsum(multistep_predictions))

        ci_test_weeks = []

        for index, interval in enumerate(
                self.future_weeks.loc[
                    self.future_weeks.index[-self.test_weeks:], f"{self.column}_{self.nn_model_name}"
                ]
        ):
            ci_test_weeks.append(self.nn_model_ci * (index + 1))


        self.future_weeks.loc[
            self.future_weeks.index[-self.test_weeks:], f"{self.nn_model_name}_conf_int_lower"
        ] = self.future_weeks.loc[
            self.future_weeks.index[-self.test_weeks:], f"{self.column}_{self.nn_model_name}"
        ] - ci_test_weeks

        self.future_weeks.loc[
            self.future_weeks.index[-self.test_weeks:], f"{self.nn_model_name}_conf_int_upper"
        ] = self.future_weeks.loc[
            self.future_weeks.index[-self.test_weeks:], f"{self.column}_{self.nn_model_name}"
        ] + ci_test_weeks

        print('nn preds done')

    def define_crnn_training_models(self) -> None:
        seed = 0

        # neurons_1 = self.neurons_nn[0]
        # neurons_2 = self.neurons_nn[1]
        # neurons_3 = self.neurons_nn[2]
        # neurons_4 = self.neurons_nn[3]
        regularization = self.regularization_crnn

        # Define the input layer
        inputs = tf.keras.Input(shape=(self.X_train_nn.shape[1], 1))

        # Add the first hidden convolutional layer with L2 regularization
        x = tf.keras.layers.Conv1D(
            filters=128, kernel_size=3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(regularization)
        )(inputs)

        # Add LSTM layer
        # x = tf.keras.layers.LSTM(128, return_sequences=True)(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)

        # Add Batch Normalization
        x = tf.keras.layers.BatchNormalization()(x)

        # Add hidden convolutional layer with L2 regularization
        x = tf.keras.layers.Conv1D(
            filters=64, kernel_size=3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(regularization)
        )(x)

        # Add LSTM layer
        # x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)

        # Add Batch Normalization
        x = tf.keras.layers.BatchNormalization()(x)

        # Add hidden convolutional layer with L2 regularization
        x = tf.keras.layers.Conv1D(
            filters=32, kernel_size=3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(regularization)
        )(x)

        # Add LSTM layer
        # x = tf.keras.layers.LSTM(32, return_sequences=True)(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True))(x)

        # Add Batch Normalization
        x = tf.keras.layers.BatchNormalization()(x)

        # # Add LSTM layer
        # x = tf.keras.layers.LSTM(32)(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(x)

        # x = tf.keras.layers.GlobalMaxPooling1D()(x)

        # Add hidden layer with L2 regularization
        x = tf.keras.layers.Dense(
            60, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(regularization)
        )(x)

        # Add Batch Normalization
        x = tf.keras.layers.BatchNormalization()(x)

        # Add hidden layer with L2 regularization
        x = tf.keras.layers.Dense(
            50, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(regularization)
        )(x)

        # Add Batch Normalization
        x = tf.keras.layers.BatchNormalization()(x)

        # Add hidden layer with L2 regularization
        x = tf.keras.layers.Dense(
            40, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(regularization)
        )(x)

        # Add Batch Normalization
        x = tf.keras.layers.BatchNormalization()(x)

        # Add hidden layer with L2 regularization
        x = tf.keras.layers.Dense(
            30, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(regularization)
        )(x)

        # Add Batch Normalization
        x = tf.keras.layers.BatchNormalization()(x)

        # Output layer
        outputs = tf.keras.layers.Dense(1)(x)

        # Create the model
        self.crnn_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.crnn_model.summary()

    def compile_and_fit_crnn_models(self) -> None:
        self.reset_random_seeds()
        tf.keras.backend.clear_session()
        batch_size = 64
        # epochs = 400
        epochs = 1000
        self.crnn_model.compile(
            #   loss='mse',
            # loss='mae',
            loss=tf.keras.losses.Huber(),
            #   loss = [tf.keras.metrics.RootMeanSquaredError()],
            optimizer='adam',
        )

        checkpoint_filepath = os.path.join(os.getcwd(), self.crnn_model_checkpoint_path)
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            # monitor='val_root_mean_squared_error',
            monitor='val_loss',
            mode='min',
            save_best_only=True
        )

        self.history_crnn_model = self.crnn_model.fit(
            self.X_train_nn,
            self.Y_train_nn,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(self.X_val_nn, self.Y_val_nn),
            callbacks=[model_checkpoint_callback],
            verbose=2
        )

        pd.DataFrame(self.history_crnn_model.history).to_csv(self.crnn_model_history_path)

        print(f"CRNN Model at Epoch 1000 - MAE: {self.crnn_model.evaluate(self.X_test_nn, self.Y_test_nn)}")

        plot_loss = self.history_crnn_model.history['loss']
        plot_val_loss = self.history_crnn_model.history['val_loss']
        plot_epochs = range(len(plot_loss))

        fig_mae = plt.figure(figsize=(10, 5))
        plt.plot(plot_epochs, plot_loss, 'b', label='Training loss (mae)')
        plt.plot(plot_epochs, plot_val_loss, 'r', label='Validation loss (mae)')
        plt.title('Training and Validation loss (mae)')
        plt.grid()
        plt.legend()

    def load_crnn_model_weights(self) -> None:
        checkpoint_filepath = os.path.join(os.getcwd(), self.crnn_model_checkpoint_path)
        self.crnn_model.load_weights(checkpoint_filepath)

        print(f"CRNN Best model - MAE: {self.crnn_model.evaluate(self.X_test_nn, self.Y_test_nn)}")

    def test_crnn_models(self) -> None:

        # first T+1 values are not predictable
        train_idx_nn = self.train_index.copy()
        train_idx_nn[:self.weeks_nn + 1] = False

        # Ptrain = self.scaler.inverse_transform(self.nn_model.predict(self.X_train_nn)).flatten()
        # Ptest = self.scaler.inverse_transform(self.nn_model.predict(self.X_test_nn)).flatten()
        Ptrain = self.scaler.inverse_transform(self.crnn_model.predict(
            self.nn_df.loc[
                :len(self.nn_df) - self.test_weeks - 1,
                [f"feature_{feat}" for feat in range(self.nn_df.shape[1] - 1)]
            ]
        )).flatten()

        prev = self.modelling_data[self.log_column].shift(1)

        # Last-known train value
        last_train = self.train_arima_df.iloc[-1][self.log_column]

        # multi-step forecast
        multistep_predictions = []

        # first test input
        # last_x = self.X_test_nn_1[0]
        # last_x = self.X_test_nn.iloc[0].to_numpy()
        last_x = self.X_test_nn.iloc[[0]]

        while len(multistep_predictions) < self.test_weeks:
            # p = self.nn_model.predict(last_x.reshape(1, -1))[0]
            # p = self.nn_model.predict(last_x.reshape(1, self.weeks_nn))[0]
            p = self.crnn_model.predict(last_x)[0]

            # update the predictions list
            multistep_predictions.append(p)

            # make the new input
            last_x = np.roll(last_x, -1)
            last_x[-1] = p

        # unscale
        multistep_predictions = np.array(multistep_predictions)
        multistep_predictions = self.scaler.inverse_transform(multistep_predictions.reshape(-1, 1)).flatten()

        # save multi-step forecast to dataframe
        self.modelling_data.loc[train_idx_nn, f"{self.crnn_model_name}_output"] = np.exp(prev[train_idx_nn] + Ptrain)
        self.modelling_data.loc[
            self.test_index, f"{self.crnn_model_name}_output"
        ] = np.exp(last_train + np.cumsum(multistep_predictions))

        residuals = (self.modelling_data.loc[train_idx_nn, f"{self.column}_{self.crnn_model_name}"] -
                     self.modelling_data.loc[train_idx_nn, f"{self.crnn_model_name}_output"])

        self.crnn_model_ci = np.quantile(residuals, 1 - self.alpha_ci)

        ci_test_weeks = []

        for index, interval in enumerate(self.modelling_data.loc[self.test_index, f"{self.crnn_model_name}_output"]):
            ci_test_weeks.append(self.crnn_model_ci * (index + 1))

        self.modelling_data.loc[
            self.test_index, f"{self.crnn_model_name}_conf_int_lower"
        ] = self.modelling_data.loc[self.test_index, f"{self.crnn_model_name}_output"] - ci_test_weeks

        self.modelling_data.loc[
            self.test_index, f"{self.crnn_model_name}_conf_int_upper"
        ] = self.modelling_data.loc[self.test_index, f"{self.crnn_model_name}_output"] + ci_test_weeks

        print('crnn models done')

        # plot 1-step and multi-step forecast
        # self.modelling_data.iloc[-16:][['log_Close', 'multistep', '1step_test']].plot(figsize=(15, 5))

    def get_crnn_real_predictions(self) -> None:
        # first T+1 values are not predictable
        train_idx_nn = self.train_index.copy()
        train_idx_nn[:self.weeks_nn + 1] = False

        # Last-known train value
        last_test = self.test_arima_df.iloc[-1][self.log_column]

        # multi-step forecast
        multistep_predictions = []

        # first test input
        # last_x = self.X_test_nn[-1]
        last_x = self.X_test_nn.iloc[[-1]]

        while len(multistep_predictions) < self.test_weeks:
            # p = self.nn_model.predict(last_x.reshape(1, -1))[0]
            p = self.crnn_model.predict(last_x)[0]

            # update the predictions list
            multistep_predictions.append(p)

            # make the new input
            last_x = np.roll(last_x, -1)
            last_x[-1] = p

        # unscale
        multistep_predictions = np.array(multistep_predictions)
        multistep_predictions = self.scaler.inverse_transform(multistep_predictions.reshape(-1, 1)).flatten()

        # save multi-step forecast to modelling dataframe
        # self.modelling_data.loc[train_idx_nn, f"{self.nn_model_name}_output"] = np.exp(prev[train_idx_nn] + Ptrain)
        # self.modelling_data.loc[
        #     self.test_index, f"{self.nn_model_name}_output"
        # ] = np.exp(last_train + np.cumsum(multistep_predictions))

        # save multi-step forecast to future weeks dataframe
        self.future_weeks.loc[
            self.future_weeks.index[-self.test_weeks:], f"{self.column}_{self.crnn_model_name}"
        ] = np.exp(last_test + np.cumsum(multistep_predictions))

        ci_test_weeks = []

        for index, interval in enumerate(
                self.future_weeks.loc[
                    self.future_weeks.index[-self.test_weeks:], f"{self.column}_{self.crnn_model_name}"
                ]
        ):
            ci_test_weeks.append(self.crnn_model_ci * (index + 1))


        self.future_weeks.loc[
            self.future_weeks.index[-self.test_weeks:], f"{self.crnn_model_name}_conf_int_lower"
        ] = self.future_weeks.loc[
            self.future_weeks.index[-self.test_weeks:], f"{self.column}_{self.crnn_model_name}"
        ] - ci_test_weeks

        self.future_weeks.loc[
            self.future_weeks.index[-self.test_weeks:], f"{self.crnn_model_name}_conf_int_upper"
        ] = self.future_weeks.loc[
            self.future_weeks.index[-self.test_weeks:], f"{self.column}_{self.crnn_model_name}"
        ] + ci_test_weeks

        print('crnn preds done')

    def run_nn_model(self) -> None:
        self.reset_random_seeds()
        self.define_nn_training_models()
        if self.train_nn:
            self.compile_and_fit_nn_models()
        # self.compile_and_fit_nn_models()
        self.load_nn_model_weights()
        self.test_nn_models()
        self.get_nn_real_predictions()

    def run_crnn_model(self) -> None:
        self.reset_random_seeds()
        self.define_crnn_training_models()
        if self.train_nn:
            self.compile_and_fit_crnn_models()
        # self.compile_and_fit_nn_models()
        self.load_crnn_model_weights()
        self.test_crnn_models()
        self.get_crnn_real_predictions()

    def run_arima_model(self) -> None:

        if self.train_arima:
            self.train_arima_best_order()
        self.define_arima_training_models()
        self.define_arima_predictive_models()
        self.fit_arima_training_models()
        self.fit_arima_predictive_models()
        self.test_arima_models()
        self.get_arima_real_predictions()

    def get_training_log_cols(self) -> None:
        """
        Identifies columns containing log-transformed data in the modelling_data DataFrame.

        - Creates a list named `log_cols` containing all column names
            with 'log' in them (indicating log-transformed data).
        - Creates a list named `not_log_cols` containing all remaining columns (original data).
        """

        self.log_training_cols = [col for col in self.modelling_data.columns if 'log' in col]
        self.not_log_training_cols = [col for col in self.modelling_data.columns if 'log' not in col]

    def get_predictive_log_cols(self) -> None:
        """
        Identifies columns containing log-transformed data in the modelling_data DataFrame.

        - Creates a list named `log_cols` containing all column names
            with 'log' in them (indicating log-transformed data).
        - Creates a list named `not_log_cols` containing all remaining columns (original data).
        """

        self.log_predictive_cols = [col for col in self.future_weeks.columns if 'log' in col]
        self.not_log_predictive_cols = [col for col in self.future_weeks.columns if 'log' not in col]

    def reverse_training_logs(self) -> None:
        """
        Reverses the log transformation on the identified log columns in modelling_data.

        Iterates through the `log_cols` list and applies the exponential function (np.exp) to each column
        to convert the log-transformed values back to their original scale.
        """

        for column in self.log_training_cols:
            self.modelling_data[column] = np.exp(self.modelling_data[column])

    def reverse_predictive_logs(self) -> None:
        """
        Reverses the log transformation on the identified log columns in modelling_data.

        Iterates through the `log_cols` list and applies the exponential function (np.exp) to each column
        to convert the log-transformed values back to their original scale.
        """

        for column in self.log_predictive_cols:
            self.future_weeks[column] = np.exp(self.future_weeks[column])

    def reverse_logs(self) -> None:
        self.get_training_log_cols()
        self.get_predictive_log_cols()
        self.reverse_training_logs()
        self.reverse_predictive_logs()

    def save_data(self) -> None:
        """
        Saves the modelling_data DataFrame to a CSV file for further analysis or visualization.

        Saves the `modelling_data` DataFrame containing the original data, log-transformed data,
        predictions, and confidence intervals to a CSV file named 'docs/modelled_fitted_data.csv' by default.
        """

        self.modelling_data.to_csv(self.relative_path_to_modelling_data)
        self.future_weeks.to_csv(self.relative_path_to_predictive_data)

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

        self.run_arima_model()
        self.run_nn_model()
        self.run_crnn_model()
        print('yeah')

        # self.reverse_logs()

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
    column_list = config['config']['column_list']
    years = config['config']['years']
    update_data = config['config']['update_data']
    freq = config['config']['freq']
    order = tuple(config['config']['order'])
    train_arima = config['config']['train_arima']
    train_nn = config['config']['train_nn']
    weeks_nn = config['config']['weeks_nn']
    alpha_ci = config['config']['alpha_ci']
    order_gridsearch = config['config']['order_gridsearch']
    neurons_nn = config['config']['neurons_nn']
    regularization_nn = config['config']['regularization_nn']
    regularization_crnn = config['config']['regularization_crnn']
    train_val_partition = config['config']['train_val_partition']

    arima = Arima(
        ticker=ticker,
        period=period,
        order=order,
        display_weeks=display_weeks,
        test_weeks=test_weeks,
        freq=freq,
        column=column,
        column_list=column_list,
        years=years,
        update_data=update_data,
        train_arima=train_arima,
        train_nn=train_nn,
        weeks_nn=weeks_nn,
        alpha_ci=alpha_ci,
        order_gridsearch=order_gridsearch,
        neurons_nn=neurons_nn,
        regularization_nn=regularization_nn,
        regularization_crnn=regularization_crnn,
        train_val_partition=train_val_partition,
    )

    arima.run_pipeline()
