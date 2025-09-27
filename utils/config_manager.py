import yaml
from pathlib import Path
from utils.logger import get_logger

class ConfigManager:
    def __init__(self, config_dir='configs'):
        self.logger = get_logger(__name__)
        self.config_dir = Path(config_dir)
        self.configs = {}
        self.load_all_configs()

    def load_yaml(self, filename):
        with open(self.config_dir / filename, "r") as f:
            config = yaml.safe_load(f)
            if config in None:
                raise ValueError(f'Config file "{filename}" is empty.')
            return config

    def load_all_configs(self):
        if not self.config_dir.exists():
            self.logger.warnring(f'Config directory "{self.config_dir}" does not exist.')
            return

        for yaml_file in self.config_dir.glob("*_config.yaml"):
            env_name = yaml_file.stem.replace('_config', '')
            self.configs[env_name] = self.load_yaml(yaml_file)
            self.logger.info(f'Loaded configuration for {env_name}')

    def get(self, env_name, validate=True):
        config = self.configs.get(env_name)
        if config is None:
            available = list(self.configs.keys())
            raise KeyError(f'Config "{env_name}" not found. Available configs: {available}')
        if validate:
            self.validate_config(config)

        return config

    def get_available_configs(self):
        return list(self.configs.keys())

    def validate_config(self, config: dict):
        training_config = config.get('training')
        if not training_config:
            return True

        required_keys = ['target_score', 'max_timesteps', 'learning_rate', 'policy_net', 'eval_freq']
        missing_keys = [key for key in required_keys if key not in training_config]
        if missing_keys:
            raise ValueError (f'Configuration missing required keys: {missing_keys}')
        self.logger.info('Configuration validation passed')
        return True

config_manager = ConfigManager()