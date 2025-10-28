# --------- Third-party imports ---------#
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3 import A2C
import torch

# --------- Local imports ---------#
from agents.base_agent import BaseAgent

# --------- A2C Agent Class ---------#
class A2CAgent(BaseAgent):
    def _create_model(self, config):
        """
        Creates A2C agent

        :param config: Agent specific configurations
        :return: A2C agent
        """

        # Neural network architecture for the policy
        policy_kwargs = dict(
            activation_fn=torch.nn.ReLU,
            net_arch=config['policy_net']
        )

        return A2C(
            'MlpPolicy',
            self.vec_env,
            learning_rate=config.get('learning_rate', 0.0007),
            n_steps=config.get('n_steps', 5),
            gamma=config.get('gamma', 0.99),
            gae_lambda=config.get('gae_lambda', 1.0),
            ent_coef=config.get('ent_coef', 0.0),
            vf_coef=config.get('vf_coef', 0.5),
            max_grad_norm=config.get('max_grad_norm', 0.5),
            rms_prop_eps=config.get('rms_prop_eps', 1e-5),
            use_rms_prop=config.get('use_rms_prop', True),
            use_sde=config.get('use_sde', False),
            normalize_advantage=config.get('normalize_advantage', False),
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=self.tensorboard_log
        )

    def _create_training_callbacks(self, config):
        """
        Training callbacks

        :param config: Agent specific configurations
        :return: evaluation callback
        """

        # Callback to stop training when target reward is reached
        stop_callback = StopTrainingOnRewardThreshold(
            reward_threshold=config['target_score'],
            verbose=1
        )

        if self.run_manager:
            best_model_path = self.run_manager.get_run_dir()
        else:
            best_model_path = './models/best_models'

        # Evaluation callback
        eval_callback = EvalCallback(
            self.eval_env,
            callback_on_new_best=stop_callback,
            eval_freq=config['eval_freq'],
            deterministic=True,
            render=False,
            verbose=1,
            best_model_save_path=best_model_path
        )

        return eval_callback

    def get_algorithm_class(self):
        """Return the algorithm class"""
        return A2C

    def predict(self, obs, deterministic):
        return self.model.predict(obs, deterministic=deterministic)