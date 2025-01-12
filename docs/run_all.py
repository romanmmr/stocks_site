import os
import shutil
import papermill as pm

# Local application imports
from models.arima import *


def copy_rename_file(source_path, destination_dir, new_filename):
    """
    Copies a file to a new location and renames it.

    Args:
    source_path: Path to the source file.
    destination_dir: Path to the destination directory.
    new_filename: New name for the file.

    Returns:
    True if the operation was successful, False otherwise.
    """
    try:
        # Create the full destination path
        destination_path = os.path.join(destination_dir, new_filename)

        # Copy the file to the destination
        shutil.copy2(source_path, destination_path)

        return True
    except Exception as e:
        print(f"Error copying and renaming file: {e}")
        return False


def run_notebooks(
        notebook_list: list = [
            'arima_model.ipynb',
            'arima_predictive_notebook.ipynb',
            'nn_model.ipynb',
            'nn_model_predictive_notebook.ipynb',
            'crnn_model.ipynb',
            'crnn_model_predictive_notebook.ipynb'
        ],
        input_directory: str = 'quarto_files',
        output_directory: str = 'docs',
) -> None:
    for notebook_name in notebook_list:
        input_path = os.path.join(input_directory, notebook_name)
        output_path = os.path.join(output_directory, notebook_name)
        pm.execute_notebook(
            input_path=input_path,
            output_path=output_path
        )

    return


def render_files() -> None:
    # Copy quarto yml file to render index:
    source_file = "quarto_files/_quarto_index.yml"
    # destination_directory = "path/to/destination/directory"
    destination_directory = ""
    new_file_name = "_quarto.yml"

    if copy_rename_file(source_file, destination_directory, new_file_name):
        print(f"File copied and renamed successfully to {destination_directory}/{new_file_name}")

    os.system('''
        quarto render index.qmd
    ''')

    # Copy quarto yml file to render index:
    source_file = "quarto_files/_quarto_projects.yml"
    # destination_directory = "path/to/destination/directory"
    destination_directory = ""
    new_file_name = "_quarto.yml"

    if copy_rename_file(source_file, destination_directory, new_file_name):
        print(f"File copied and renamed successfully to {destination_directory}/{new_file_name}")

    os.system('''
        quarto render projects.qmd
        quarto render docs/arima_model.ipynb
        quarto render docs/arima_model_predictive_notebook.ipynb
        quarto render docs/nn_model.ipynb
        quarto render docs/nn_model_predictive_notebook.ipynb
        quarto render docs/crnn_model.ipynb
        quarto render docs/crnn_model_predictive_notebook.ipynb
    ''')
    return

def run_notebooks_and_render():
    run_notebooks()
    render_files()


if __name__ == "__main__":

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

    run_notebooks_and_render()

