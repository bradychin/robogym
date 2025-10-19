# --------- Local Imports ---------#
from environments.base_environment import BaseEnvironment
from utils.config_manager import ConfigManager
config_manager = ConfigManager()
acrobot_config = config_manager.get('acrobot_config')

# --------- Bipedal Walker environment---------#
class AcrobotEnv(BaseEnvironment):
    def __init__(self, render_mode='rgb_array', run_manager=None):
        super().__init__(acrobot_config['environment']['env_id'],
                         render_mode,
                         run_manager=run_manager)