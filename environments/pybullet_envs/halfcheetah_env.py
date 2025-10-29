# --------- Third-party Imports ---------#
import pybullet_envs_gymnasium

# --------- Local Imports ---------#
from environments.base_environment import BaseEnvironment
from utils.config_manager import ConfigManager
config_manager = ConfigManager()
halfcheetah_config = config_manager.get('halfcheetah_config')

# --------- Half Cheetah Pybullet environment ---------#
class HalfCheetahEnv(BaseEnvironment):
    def __init__(self, render_mode='rgb_array', run_manager=None):
        super().__init__(halfcheetah_config['environment']['env_id'],
                         render_mode,
                         run_manager=run_manager)