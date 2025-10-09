# --------- Environment imports ---------#
from environments.gym_envs.bipedalwalker_env import BipedalWalkerEnv

# --------- Local imports ---------#
from utils.logger import logger
logger = logger(__name__)

# --------- Environment factory class ---------#
class EnvironmentFactory:
    """Factory class for creating environment instances"""

    ENVIRONMENTS = {
        'bipedalwalker': BipedalWalkerEnv
    }

    @classmethod
    def create(cls, env_name: str, env_type: str, render_mode: str = 'rgb_array', run_manager=None):
        """
        Create environment instance

        :param env_name: Name of environment to create
        :param env_type: Type of environment to create
        :param render_mode: Rendering mode for environment
        :param run_manager: RunManager instance
        :return: BaseEnvironment instance
        """

        env_name = env_name.lower()
        if env_name not in cls.ENVIRONMENTS:
            available = ': '.join(cls.ENVIRONMENTS.keys())
            raise ValueError(f'Environment "{env_name}" not available. Available environments: {available}')

        logger.info(f'Creating {env_type} environment: {env_name}')
        return cls.ENVIRONMENTS[env_name](render_mode=render_mode, run_manager=run_manager)

    @classmethod
    def get_available_environments(cls):
        """Get list of available environments"""
        return list(cls.ENVIRONMENTS.keys())