# --------- Environment imports ---------#
from environments.gym_envs.bipedalwalker_env import BipedalWalkerEnv
from environments.gym_envs.cartpole_env import CartPoleEnv
from environments.gym_envs.pendulum_env import PendulumEnv
from environments.gym_envs.lunarlander_env import LunarLanderEnv
from environments.gym_envs.lunarlandercontinuous_env import LunarLanderContinuousEnv
from environments.gym_envs.acrobot_env import AcrobotEnv
from environments.gym_envs.mountaincar_env import MountainCarEnv
from environments.gym_envs.mountaincarcontinuous_env import MountainCarContinuousEnv

# --------- Local imports ---------#
from utils.logger import logger
logger = logger(__name__)

# --------- Environment factory class ---------#
class EnvironmentFactory:
    """Factory class for creating environment instances"""

    ENVIRONMENTS = {
        'bipedalwalker': BipedalWalkerEnv,
        'cartpole': CartPoleEnv,
        'pendulum': PendulumEnv,
        'lunarlander': LunarLanderEnv,
        'lunarlandercontinuous': LunarLanderContinuousEnv,
        'acrobot': AcrobotEnv,
        'mountaincar': MountainCarEnv,
        'mountaincarcontinuous': MountainCarContinuousEnv,
    }

    # Define action space types for environments
    ENVIRONMENT_ACTION_SPACES = {
        'bipedalwalker': 'continuous',
        'cartpole': 'discrete',
        'pendulum': 'continuous',
        'lunarlander': 'discrete',
        'lunarlandercontinuous': 'continuous',
        'acrobot': 'discrete',
        'mountaincar': 'discrete',
        'mountaincarcontinuous': 'continuous'
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

    @classmethod
    def get_action_space_type(cls, env_name: str):
        """
        Get action space type for a given environment

        :param env_name: Name of the environment
        :return: Action space type (discrete or continuous)
        """

        env_name = env_name.lower()
        if env_name not in cls.ENVIRONMENT_ACTION_SPACES:
            raise ValueError(f'Environment "{env_name}" not found')
        return cls.ENVIRONMENT_ACTION_SPACES[env_name]