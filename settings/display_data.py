import matplotlib.pyplot as plt
import pandas as pd
import os


def plot_sample_data(path: str, column: str) -> None:
    modelling_data = pd.read_csv(path, index_col='Date')
    # modelling_data.index = pd.to_datetime(modelling_data.index).strftime('%Y-%m-%d')
    # modelling_data.index = pd.to_datetime(modelling_data.index, utc=True)
    modelling_data.index = pd.Series(modelling_data.index).apply(lambda x: pd.to_datetime(x).strftime('%Y-%m-%d'))
    modelling_data.index = modelling_data.index.astype(str).map(lambda x: x[:10])
    # data = round(modelling_data[column], 1).tail().to_frame().reset_index()

    return round(modelling_data[column], 1).tail().to_frame()


def table_horizontal(path: str, column: str, display_weeks: int) -> None:
    modelling_data = pd.read_csv(path, index_col='Date')
    # modelling_data.index = pd.to_datetime(modelling_data.index).strftime('%Y-%m-%d')
    # modelling_data.index = pd.to_datetime(modelling_data.index, utc=True)
    modelling_data.index = pd.Series(modelling_data.index).apply(lambda x: pd.to_datetime(x).strftime('%Y-%m-%d'))
    modelling_data.index = modelling_data.index.astype(str).map(lambda x: x[:10])
    modelling_data['if we knew the future'] = modelling_data[column]
    modelling_data['hypothetical present'] = modelling_data[column].apply(lambda x: str(round(x, 1)))
    index_to_change = modelling_data['hypothetical present'].tail(int(display_weeks / 2)).index
    modelling_data.loc[index_to_change, 'hypothetical present'] = '?'
    # data = round(modelling_data[['if we knew the future', 'hypothetical present']], 1).tail(display_weeks).T

    return round(modelling_data[['if we knew the future', 'hypothetical present']], 1).tail(display_weeks).T


def plot_fit_and_forecast(path: str, column: str, model: str, display_weeks: int) -> None:
    modelling_data = pd.read_csv(path, index_col='Date')
    # modelling_data.index = pd.to_datetime(modelling_data.index).strftime('%Y-%m-%d')
    # modelling_data.index = pd.to_datetime(modelling_data.index, utc=True)
    modelling_data.index = pd.Series(modelling_data.index).apply(lambda x: pd.to_datetime(x).strftime('%Y-%m-%d'))
    modelling_data.index = modelling_data.index.astype(str).map(lambda x: x[:10])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        modelling_data[column].tail(display_weeks),
        label='real data'
    )
    ax.plot(
        # modelling_data['arima_output'].tail(display_weeks),
        modelling_data[f"{model}_output"].tail(display_weeks),
        color='green',
        label=f"{model} model output - fitted"
    )
    ax.fill_between(
        modelling_data.tail(display_weeks).index,
        # modelling_data['arima_conf_int_lower'].tail(display_weeks),
        # modelling_data['arima_conf_int_upper'].tail(display_weeks),
        modelling_data[f"{model}_conf_int_lower"].tail(display_weeks),
        modelling_data[f"{model}_conf_int_upper"].tail(display_weeks),
        color='red',
        alpha=0.3,
        label='confidence interval'
    )
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()


def plot_predictions(path: str, column: str, model: str, display_weeks: int) -> None:
    predicted_data = pd.read_csv(path, index_col='Date')
    # modelling_data.index = pd.to_datetime(modelling_data.index).strftime('%Y-%m-%d')
    # predicted_data.index = pd.to_datetime(predicted_data.index, utc=True)
    predicted_data.index = pd.Series(predicted_data.index).apply(lambda x: pd.to_datetime(x).strftime('%Y-%m-%d'))
    predicted_data.index = predicted_data.index.astype(str).map(lambda x: x[:10])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        predicted_data[f"{column}_{model}"].tail(display_weeks),
        label=f"{model} real data + predictions"
    )

    ax.fill_between(
        predicted_data.tail(display_weeks).index,
        # modelling_data['arima_conf_int_lower'].tail(display_weeks),
        # modelling_data['arima_conf_int_upper'].tail(display_weeks),
        predicted_data[f"{model}_conf_int_lower"].tail(display_weeks),
        predicted_data[f"{model}_conf_int_upper"].tail(display_weeks),
        color='red',
        alpha=0.3,
        label='confidence interval'
    )
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()


def plot_nn_loss(path: str, y_min: int = 0, y_max: int = 5) -> None:
    my_path = ''
    for _ in os.getcwd().split('\\')[:-1]:
        my_path += _ + '\\'

    history = pd.read_csv(os.path.join(my_path, path))

    plot_loss = history['loss']
    plot_val_loss = history['val_loss']
    plot_epochs = range(len(plot_loss))

    fig_mae = plt.figure(figsize=(10, 5))
    plt.plot(plot_epochs, plot_loss, 'b', label='Training loss (mae)')
    plt.plot(plot_epochs, plot_val_loss, 'r', label='Validation loss (mae)')
    plt.title('Training and Validation loss (mae)')
    plt.grid()
    plt.legend()

    # Set fixed limits for the y-axis
    plt.ylim(y_min, y_max)  # Replace y_min and y_max with your desired values

    # plt.show()


# if __name__ == '__main__':
#     print('start')
#     plot_sample_data(path='modelled_fitted_data.csv', column='Close')
#     table_horizontal(path='modelled_fitted_data.csv', column='Close', display_weeks=16)
#     plot_fit_and_forecast(path='modelled_fitted_data.csv', column='Close', display_weeks=16)
#     print('done')