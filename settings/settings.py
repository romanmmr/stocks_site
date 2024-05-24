import yaml

def load_config(config_path):
  """Loads configuration settings from a YAML file.

  Args:
      config_path (str): Path to the YAML configuration file.

  Returns:
      dict: Dictionary containing the loaded configuration settings.
  """
  with open(config_path, 'r') as file:
    config_data = yaml.safe_load(file)
  return config_data

# Example usage (assuming this is in settings.py)
# You can call this function from other parts of your code
# config = load_config("config.yaml")
