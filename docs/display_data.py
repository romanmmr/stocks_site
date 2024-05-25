import matplotlib.pyplot as plt
import pandas as pd


def plot_sample_data(path: str, column: str) -> None:
    modelling_data = pd.read_csv(path, index_col='Date')
    # modelling_data.index = pd.to_datetime(modelling_data.index).strftime('%Y-%m-%d')
    modelling_data.index = pd.to_datetime(modelling_data.index, utc=True)
    modelling_data.index = modelling_data.index.astype(str).map(lambda x: x[:10])
    # data = round(modelling_data[column], 1).tail().to_frame()

    return round(modelling_data[column], 1).tail().to_frame()


def table_horizontal(path: str, column: str, display_weeks: int) -> None:
    modelling_data = pd.read_csv(path, index_col='Date')
    # modelling_data.index = pd.to_datetime(modelling_data.index).strftime('%Y-%m-%d')
    modelling_data.index = pd.to_datetime(modelling_data.index, utc=True)
    modelling_data.index = modelling_data.index.astype(str).map(lambda x: x[:10])
    modelling_data['if we knew the future'] = modelling_data[column]
    modelling_data['hypothetical present'] = modelling_data[column].apply(lambda x: str(round(x, 1)))
    index_to_change = modelling_data['hypothetical present'].tail(int(display_weeks/2)).index
    modelling_data.loc[index_to_change, 'hypothetical present'] = '?'
    # data = round(modelling_data[['if we knew the future', 'hypothetical present']], 1).tail(display_weeks).T

    return round(modelling_data[['if we knew the future', 'hypothetical present']], 1).tail(display_weeks).T



def plot_fit_and_forecast(path: str, column: str, display_weeks: int) -> None:
    modelling_data = pd.read_csv(path, index_col='Date')
    # modelling_data.index = pd.to_datetime(modelling_data.index).strftime('%Y-%m-%d')
    modelling_data.index = pd.to_datetime(modelling_data.index, utc=True)
    modelling_data.index = modelling_data.index.astype(str).map(lambda x: x[:10])

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(
        modelling_data[column].tail(display_weeks),
        label='real data'
    )
    ax.plot(
        # modelling_data['arima_output'].tail(display_weeks),
        modelling_data['arima_log_output'].tail(display_weeks),
        color='green',
        label='model output - fitted'
    )
    ax.fill_between(
        modelling_data.tail(display_weeks).index,
        # modelling_data['arima_conf_int_lower'].tail(display_weeks),
        # modelling_data['arima_conf_int_upper'].tail(display_weeks),
        modelling_data['arima_log_conf_int_lower'].tail(display_weeks),
        modelling_data['arima_log_conf_int_upper'].tail(display_weeks),
        color='red',
        alpha=0.3,
        label='confidence interval'
    )
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()


# if __name__ == '__main__':
#     print('start')
#     plot_sample_data(path='arima_fitted_data.csv', column='Close')
#     table_horizontal(path='arima_fitted_data.csv', column='Close', display_weeks=16)
#     plot_fit_and_forecast(path='arima_fitted_data.csv', column='Close', display_weeks=16)
#     print('done')
