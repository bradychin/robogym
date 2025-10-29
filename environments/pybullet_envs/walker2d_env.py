# --------- Third-party Imports ---------#
import pybullet_envs_gymnasium

# --------- Local Imports ---------#
from environments.base_environment import BaseEnvironment
from utils.config_manager import ConfigManager
config_manager = ConfigManager()
walker2d_config = config_manager.get('walker2d_config')

# --------- Walker2D Pybullet environment ---------#
class Walker2DEnv(BaseEnvironment):
    def __init__(self, render_mode='rgb_array', run_manager=None):
        super().__init__(walker2d_config['environment']['env_id'],
                         render_mode,
                         run_manager=run_manager)