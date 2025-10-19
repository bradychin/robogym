# --------- Local Imports ---------#
from environments.base_environment import BaseEnvironment
from utils.config_manager import ConfigManager
config_manager = ConfigManager()
cartpole_config = config_manager.get('cartpole_config')

# --------- Bipedal Walker environment---------#
class CartPoleEnv(BaseEnvironment):
    def __init__(self, render_mode='rgb_array', run_manager=None):
        super().__init__(cartpole_config['environment']['env_id'],
                         render_mode,
                         run_manager=run_manager)