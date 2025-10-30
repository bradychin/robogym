# --------- Standard library imports ---------#
from datetime import datetime
from pathlib import Path
import json

# --------- Third-party imports ---------#
import gymnasium as gym
import pybullet_envs_gymnasium
from stable_baselines3.common.env_util import make_vec_env as sb3_make_vec_env
from stable_baselines3.common.monitor import Monitor

# --------- Local imports ---------#
from utils.config_manager import ConfigManager
config_manager = ConfigManager()
utilities_config = config_manager.get('utilities_config')
from utils.logger import logger
logger = logger(__name__)

# --------- Base environment class ---------#
class BaseEnvironment:
    def __init__(self, environment_id, render_mode='rgb_array', run_manager=None, env_kwargs=None):
        self.environment_id = environment_id
        self.render_mode = render_mode
        self.run_manager = run_manager
        self.env_kwargs = env_kwargs if env_kwargs is not None else {}
        self.env = None

    def create_env(self):
        """Create a single environment wrapped with Monitor"""

        env = gym.make(self.environment_id, render_mode=self.render_mode, **self.env_kwargs)
        self.env = Monitor(env)
        return self.env

    def create_vec_env(self, n_envs=4):
        """
        Create vectorized environment

        :param n_envs: Number of environments
        :return: Vectorized environment
        """
        env_kwargs = {'render_mode': self.render_mode, **self.env_kwargs}
        return sb3_make_vec_env(self.environment_id,
                                n_envs=n_envs,
                                env_kwargs=env_kwargs)

    def demo(self, agent, max_steps=2000):
        """
        Demonstrate trained agent

        :param agent: Agent to demo
        :param max_steps: maximum steps per episode
        :return: n/a
        """

        logger.info(f'Demonstrating trained agent on {self.environment_id}...')
        try:
            demo_env = gym.make(self.environment_id, render_mode='human', **self.env_kwargs)
        except Exception as e:
            logger.warning(f'Cound not create environement with human render mode: {e}')
            logger.info('Trying without render mode...')
            try:
                demo_env = gym.make(self.environment_id, **self.env_kwargs)
            except Exception as e2:
                logger.error(f'Failed to create demo environment: {e}')
                return

        demo_episodes = []
        episode_count = 0

        try:
            obs, _ = demo_env.reset()
            total_reward = 0
            episode_steps = 0
            episode_count = 1

            for step in range(max_steps):
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = demo_env.step(action)
                total_reward += reward
                episode_steps += 1

                try:
                    demo_env.render()
                except Exception:
                    pass

                if terminated or truncated:
                    logger.info(f'Episode {episode_count} finished after {step} steps with total reward: {total_reward:.2f}')

                    demo_episodes.append({'episode': episode_count,
                                          'steps': episode_steps,
                                          'reward': float(total_reward)})

                    episode_count += 1
                    obs, _ = demo_env.reset()
                    total_reward = 0
                    episode_steps = 0

            if episode_steps > 0 and not demo_episodes:
                demo_episodes.append({'episode': 1,
                                      'steps': episode_steps,
                                      'reward': float(total_reward)})
            elif episode_steps > 0:
                demo_episodes.append({'episode': episode_count,
                                      'steps': episode_steps,
                                      'reward': float(total_reward)})

            if demo_episodes:
                rewards = [ep['reward'] for ep in demo_episodes]
                mean_reward = sum(rewards) / len(rewards)
                max_reward = max(rewards)
                min_reward = min(rewards)
                total_episodes = len(demo_episodes)
            else:
                mean_reward = max_reward = min_reward = 0.0
                total_episodes = 0

            demo_results = {'timestamp': datetime.now().strftime(utilities_config['date_time']),
                            'environment': self.environment_id,
                            'total_episodes': total_episodes,
                            'max_steps': max_steps,
                            'mean_reward': mean_reward,
                            'max_reward': max_reward,
                            'min_reward': min_reward,
                            'episodes': demo_episodes}

            print("\n" + "=" * 60)
            print("🎮 DEMO SUMMARY")
            print("=" * 60)
            print(f"  Episodes Completed: {total_episodes}")
            print(f"  Mean Reward:        {mean_reward:>10.2f}")
            print(f"  Max Reward:         {max_reward:>10.2f}")
            print(f"  Min Reward:         {min_reward:>10.2f}")
            print("=" * 60 + "\n")

            if self.run_manager:
                filepath = self.run_manager.get_directory('demo', extension='json')
            else:
                demo_dir = Path('./demos')
                demo_dir.mkdir(parents=True, exist_ok=True)
                filename = f'demo.json'
                filepath = demo_dir / filename
            with open(filepath, 'w') as f:
                json.dump(demo_results, f, indent=2)
            print(f'Demo results saved to: {filepath}')
            logger.info(f'Demo results saved to: {filepath}')

        except Exception as e:
            logger.error(f'Could not run demonstration: {e}')
            import traceback
            logger.error(traceback.format_exc())
        except KeyboardInterrupt:
            logger.warning('Demo interrupted by user')
        finally:
            demo_env.close()

    def close(self):
        if self.env:
            self.env.close()