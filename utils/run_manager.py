"""Manages run directories and organization"""

# --------- Standard library imports ---------#
import os
from pathlib import Path
from datetime import datetime

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

    def __init__(self, env_name, agent_name):
        self.env_name = env_name
        self.agent_name = agent_name
        self.timestamp = datetime.now().strftime(utilities_config['date_time'])
        # Create parent directory
        run_name = f'{self.timestamp}_{env_name}_{agent_name}'
        self.run_dir = Path(paths_config['run_path']) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        print('\nRun directories created')

    def save_config(self, config):
        """Save config file for this run"""
        config_path = self.run_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f'Config saved to: {config_path}')

    def get_log_path(self):
        """Get log path"""
        return str(self.run_dir / 'robogym.log')

    def get_tensorboard_path(self):
        """Get the tensorboard directory path"""
        tb_dir = f"{self.env_name}_{self.agent_name}_tb_{self.timestamp}"
        return str(self.run_dir / tb_dir)

    def get_model_path(self):
        """Get the models directory path"""
        model_filename = f"{self.env_name}_{self.agent_name}_model_{self.timestamp}.zip"
        return str(self.run_dir / model_filename)

    def get_evaluation_path(self, eval_timestamp=None):
        """Get the evaluations directory path"""
        if eval_timestamp is None:
            eval_timestamp = datetime.now().strftime(utilities_config['date_time'])

        eval_filename = f"{self.env_name}_{self.agent_name}_eval_{eval_timestamp}.json"
        return str(self.run_dir / eval_filename)

    def get_run_dir(self):
        """Get the run directory path"""
        return str(self.run_dir)

    @staticmethod
    def list_runs(base_runs_dir=paths_config['run_path'], limit=10):
        """List recent runs"""
        runs_dir = Path(base_runs_dir)
        if not runs_dir.exists():
            print('No runs directory found')
            return []

        runs = sorted(
            [d for d in runs_dir.iterdir() if d.is_dir()],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )[:limit]

        if not runs:
            print('No runs found')
            return []

        print(f'\nRecent runs (Last {len(runs)}')
        print('-'*80)
        for i, run in enumerate(runs, 1):
            # Extract info from directory
            parts = run.name.split('_')
            if len(parts) >= 3:
                timestamp = f'{parts[0]}_{parts[1]}'
                env_agent = '_'.join(parts[2:])
                print(f'{i}. {timestamp} - {env_agent}')
            else:
                print(f'{i}. {run.name}')
        print('-'*80 + '\n')
        return [str(r) for r in runs]

    def create_summary(self):
        """Create a summary file for this run"""
        summary_path = self.run_dir / 'summary.txt'

        eval_files = list(self.run_dir.glob('*_eval_*.json'))
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
            f.write(f"  - TensorBoard: {self.env_name}_{self.agent_name}_tb_{self.timestamp}/\n")
            f.write(f"  - Evaluations: {len(eval_files)} file(s)\n")
            f.write(f"  - Models: {len(model_files)} file(s)\n")
            f.write("=" * 60 + "\n")

        print(f'Summary created: {summary_path}')