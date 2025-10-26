# --------- Standard library imports ---------#
import yaml
from pathlib import Path

# --------- Config Manager class ---------#
class ConfigManager:
    def __init__(self, config_dir='config'):
        self.config_dir = Path(config_dir)
        self.configs = {}
        self.load_all_configs()

    def load_yaml(self, filename):
        """
        Loads YAML files

        :param filename: Name of the  YAML file
        :return: YAML config file
        """

        filepath = self.config_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Config file {filepath} does not exist")

        with open(filepath, "r") as f:
            config = yaml.safe_load(f)
            if config is None:
                raise ValueError(f'Config file "{filename}" is empty.')
            return config

    def load_all_configs(self):
        """Loads all config files"""
        if not self.config_dir.exists():
            print(f'Config directory "{self.config_dir}" does not exist.')
            return

        config_files = list(self.config_dir.glob("*_config.yaml"))

        for yaml_file in config_files:
            try:
                config_name = yaml_file.stem
                self.configs[config_name] = self.load_yaml(yaml_file.name)
            except Exception as e:
                print(f"Error loading {yaml_file}: {e}")

    def get(self,
            config_name: str,
            algorithm_name: str | None = None,
            validate: bool = True) -> dict:
        """
        Retrieves and extracts algorithm specific parameters from config file

        :param config_name: Environment configuration file
        :param algorithm_name: Name of the algorithm
        :param validate: Determine whether to validate config file or not
        :return: Config file with only algorithm specific parameters
        """

        config = self.configs.get(config_name)
        if config is None:
            available = list(self.configs.keys())
            raise KeyError(f'Config "{config_name}" not found. Available configs: {available}')

        if algorithm_name and 'algorithms' in config:
            if algorithm_name not in config['algorithms']:
                available_algorithms = list(config['algorithms'].keys())
                raise KeyError(f'Algorithms "{algorithm_name}" not found in config. Available: {available_algorithms}')

            result = {'environment': config.get('environment'),
                      'demo': config.get('demo'),
                      'training': config['algorithms'][algorithm_name]}
            config = result

        if validate:
            self.validate_config(config)

        return config

    @staticmethod
    def validate_config(config: dict) -> bool:
        """
        Checks if the training section has the required keys

        :param config: Configuration dictionary to validate
        :return: True if valid
        :raises ValueError: When required keys are missing
        """

        training_config = config.get('training')
        if not training_config:
            return True

        required_keys = ['target_score', 'max_timesteps', 'learning_rate', 'policy_net', 'eval_freq']
        missing_keys = [key for key in required_keys if key not in training_config]
        if missing_keys:
            raise ValueError(f'Configuration missing required keys: {missing_keys}')

        return True