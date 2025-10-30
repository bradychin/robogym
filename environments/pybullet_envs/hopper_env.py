# --------- Third-party Imports ---------#
import pybullet_envs_gymnasium

# --------- Local Imports ---------#
from environments.base_environment import BaseEnvironment
from utils.config_manager import ConfigManager
config_manager = ConfigManager()
hopper_config = config_manager.get('hopper_config')

# --------- Hopper Pybullet environment ---------#
class HopperEnv(BaseEnvironment):
    def __init__(self, render_mode='rgb_array', run_manager=None):
        super().__init__(hopper_config['environment']['env_id'],
                         render_mode,
                         run_manager=run_manager)