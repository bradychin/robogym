# --------- Standard library imports ---------#
import os
from abc import ABC, abstractmethod

# --------- Third-party imports ---------#
from stable_baselines3.common.evaluation import evaluate_policy

# --------- Local imports ---------#
from utils.evaluation_manager import EvaluationManager
from utils.logger import logger
logger = logger(__name__)

# --------- Config imports ---------#
from utils.config_manager import ConfigManager
config_manager = ConfigManager()
paths_config = config_manager.get('paths_config')
utilities_config = config_manager.get('utilities_config')

# --------- Base agent class ---------#
class BaseAgent(ABC):
    """Base class for all RL agents"""

    def __init__(self, vec_env, eval_env, env_name=None, agent_name=None, run_manager=None):
        self.vec_env = vec_env
        self.eval_env = eval_env
        self.env_name = env_name
        self.agent_name = agent_name
        self.run_manager = run_manager
        if run_manager:
            self.tensorboard_log = run_manager.get_directory('tensorboard')
            self.eval_manager = EvaluationManager(env_name, agent_name, run_manager)
        else:
            self.tensorboard_log = None
            self.eval_manager = EvaluationManager(env_name, agent_name)

        self.model = None

    @abstractmethod
    def _create_model(self, config):
        pass

    @abstractmethod
    def _create_training_callbacks(self, config):
        pass

    @abstractmethod
    def get_algorithm_class(self):
        pass

    def train(self, config):
        """
        Method to train agent

        :param config: Configuration settings
        """

        training_config = config['training']
        self.model = self._create_model(training_config)
        callbacks = self._create_training_callbacks(training_config)

        try:
            logger.info('Starting training...')
            print("\n🚀 Training started...")
            print(f"   Environment: {self.env_name}")
            print(f"   Agent: {self.agent_name}")
            print(f"   Max timesteps: {training_config['max_timesteps']}")
            print(f"   Target score: {training_config['target_score']}")
            if self.run_manager:
                print(f'   Run directory: {self.run_manager.get_run_dir()}')
            print(f"   TensorBoard logs: {self.tensorboard_log}\n")

            self.model.learn(total_timesteps=training_config['max_timesteps'],
                             callback=callbacks)

            print('Training completed!\n')
            logger.info('Training completed!')

        except KeyboardInterrupt:
            print('Training interrupted by user.')
            logger.warning('Training interrupted by user.')
            logger.info('Saving interrupted model...')
            if self.run_manager:
                model_path = self.run_manager.get_directory('model', extension='zip').replace('.zip', '_interrupted.zip')
                self.model.save(model_path)
                print(f"💾 Interrupted model saved to: {model_path}")
            else:
                self.model.save('./interrupted_model.zip')
        except Exception as e:
            print(f'\nTraining failed: {str(e)}')
            logger.error(f'Training failed: {str(e)}')
            return

        self._load_best_model()

    def evaluate(self, n_episodes=10):
        """
        Evaluate model with reporting

        :param n_episodes: Number of episodes to evaluate
        :param show_history: Show evaluation history
        :return: Dictionary or results
        """

        if self.model is None:
            logger.error('No model to evaluate. Train first or load a model.')
            return None

        results = self.eval_manager.evaluate_model(self.model,
                                                   self.eval_env,
                                                   n_episodes=n_episodes)

        return results

    def _load_best_model(self):
        """Chooses the loads the best model between the final model or the best model throughout training"""

        if not self.run_manager:
            logger.warning('No run manager available, skipping best model loading')
            return
        temp_best_path = os.path.join(self.run_manager.get_run_dir(), 'best_model.zip')

        if os.path.exists(temp_best_path):
            logger.info('Loading best model...')
            # Load the saved best model temporarily
            algorithm_class = self.get_algorithm_class()
            loaded_best_model = algorithm_class.load(temp_best_path)

            # Evaluate both models
            best_mean_reward, _ = evaluate_policy(loaded_best_model, self.eval_env, n_eval_episodes=5)
            final_mean_reward, _ = evaluate_policy(self.model, self.eval_env, n_eval_episodes=5)

            if best_mean_reward >= final_mean_reward:
                self.model = loaded_best_model
                print("   Using best checkpoint\n")
                logger.info(f'Using best checkpoint (reward: {best_mean_reward:.2f})')
            else:
                print("   Using final model\n")
                logger.info(f'Using final model (reward: {final_mean_reward:.2f})')

            final_model_path = self.run_manager.get_directory('model', extension='zip')
            self.model.save(final_model_path)
            logger.info(f'Model saved to: {final_model_path}')
            os.remove(temp_best_path)

        else:
            logger.warning('Best model not found. Using final training model')
            if self.run_manager:
                final_model_path = self.run_manager.get_directory('model', extension='zip')
                self.model.save(final_model_path)
                print(f'Final model saved to: {final_model_path}')