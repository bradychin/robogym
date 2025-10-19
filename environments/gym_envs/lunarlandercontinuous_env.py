# --------- Local Imports ---------#
from environments.base_environment import BaseEnvironment
from utils.config_manager import ConfigManager
config_manager = ConfigManager()
lunarlandercontinuous_config = config_manager.get('lunarlandercontinuous_config')

# --------- Bipedal Walker environment---------#
class LunarLanderContinuousEnv(BaseEnvironment):
    def __init__(self, render_mode='rgb_array', run_manager=None):
        super().__init__(lunarlandercontinuous_config['environment']['env_id'],
                         render_mode,
                         run_manager=run_manager)