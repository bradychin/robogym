# --------- Third-party imports ---------#
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env as sb3_make_vec_env

# --------- Local imports ---------#
from utils.logger import get_logger

# --------- Base environment class ---------#
class BaseEnvironment:
    def __init__(self, environment_id, render_mode='rgb_array'):
        self.logger = get_logger(__name__)
        self.environment_id = environment_id
        self.render_mode = render_mode
        self.env = None

    def create_env(self):
        self.logger.info(f'Using {self.environment_id}')
        self.env = gym.make(self.environment_id, render_mode=self.render_mode)
        return self.env

    def create_vec_env(self, n_envs=4):
        return sb3_make_vec_env(self.environment_id,
                                n_envs=n_envs,
                                env_kwargs={'render_mode': self.render_mode})

    def demo(self, agent, max_steps=2000):
        """Demonstrate trained agent"""
        self.logger.info(f'Demonstrating trained agent on {self.environment_id}...')
        demo_env = gym.make(self.environment_id, render_mode='human')

        try:
            obs, _ = demo_env.reset()
            total_reward = 0

            for step in range(max_steps):
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = demo_env.step(action)
                total_reward += reward
                demo_env.render()

                if terminated or truncated:
                    self.logger.info(f'Episode finished after {step} steps with total reward: {total_reward:.2f}')
                    obs, _ = demo_env.reset()
                    total_reward = 0
        except:
            self.logger.error(f'Could not run demonstration...')
        finally:
            demo_env.close()

    def close(self):
        if self.env:
            self.env.close()