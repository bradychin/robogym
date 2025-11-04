"""Manages run directories and organization"""

# --------- Standard library imports ---------#
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
# --------- Third-party imports ---------#
import yaml

# --------- Config imports ---------#
from utils.config_manager import ConfigManager
config_manager = ConfigManager()
paths_config = config_manager.get('paths_config', validate=False)
utilities_config = config_manager.get('utilities_config')

# --------- RunManger Class ---------#
class RunManager:
    """Manages logs and evaluations for each run"""

    def __init__(self, env_name: str, agent_name: str, seed: int = None) -> None:
        self.env_name = env_name
        self.agent_name = agent_name
        self.seed = seed
        self.timestamp = datetime.now().strftime(utilities_config['date_time'])
        # Create parent directory
        run_name = f'{self.timestamp}_{env_name}_{agent_name}'
        self.run_dir = Path(paths_config['run_path']) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        print('\nRun directories created')

    def save_config(self, config: Dict[str, Any]) -> None:
        """
        Save config file for this run

        :param config: Configuration file to save
        :return: n/a (generates a YAML config file)
        """

        config_path = self.run_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f'Config saved to: {config_path}')

    def get_log_path(self) -> str:
        """Get log path for log files"""

        return str(self.run_dir / 'robogym.log')

    def get_directory(self, item_type: str, extension: str | None = None) -> str:
        """
        Get the directory for a given item

        :param item_type: Type of item
        :param extension: Extension of item
        :return: Directory with file name
        """

        filename = f'{item_type}'
        if extension:
            filename += f'.{extension}'

        return str(self.run_dir / filename)

    def get_run_dir(self) -> str:
        """Get the run directory path"""

        return str(self.run_dir)

    def create_summary(self):
        """Create a summary file for this run"""

        summary_path = self.run_dir / 'summary.txt'

        eval_files = list(self.run_dir.glob('evaluation.json'))
        model_files = list(self.run_dir.glob('*.zip'))
        log_file = self.run_dir / 'robogym.log'

        with open(summary_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write(f"RUN SUMMARY\n")
            f.write("=" * 60 + "\n")
            f.write(f"Environment: {self.env_name}\n")
            f.write(f"Agent: {self.agent_name}\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Run Directory: {self.run_dir.name}\n")
            f.write(f"\n")
            f.write(f"Files:\n")
            f.write(f"  - Log: {log_file.name if log_file.exists() else 'N/A'}\n")
            f.write(f"  - Config: config.yaml\n")
            f.write(f"  - Seed: {self.seed if self.seed else 'Not set (random)'}\n")
            f.write(f"  - TensorBoard: {self.env_name}_{self.agent_name}_tb_{self.timestamp}/\n")
            f.write(f"  - Evaluations: {len(eval_files)} file(s)\n")
            f.write(f"  - Models: {len(model_files)} file(s)\n")
            f.write("=" * 60 + "\n")

        print(f'Summary created: {summary_path}')