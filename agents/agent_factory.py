# --------- Environment imports ---------#
from agents.ppo_agent import PPOAgent

# --------- Local imports ---------#
from utils.logger import logger
logger = logger(__name__)

# --------- Agent factory class ---------#
class AgentFactory:
    """Factory class for creating agent instances"""

    AGENTS = {
        'ppo': PPOAgent
    }

    @classmethod
    def create(cls, agent_name: str, vec_env, eval_env, env_name=None, run_manager=None):
        """
        Create agent instance

        :param agent_name: Name of the agent to create
        :param vec_env: Vectorized training environment
        :param eval_env: Evaluation environment
        :param env_name: Name of the environment
        :param run_manager: RunManager instance
        :return: BaseAgent instance
        """

        agent_name = agent_name.lower()
        if agent_name not in cls.AGENTS:
            available = ': '.join(cls.AGENTS.keys())
            raise ValueError (f'Agent "{agent_name}" not available. Available agents: {available}')

        logger.info(f'Creating agent: {agent_name}')
        return cls.AGENTS[agent_name](vec_env=vec_env,
                                      eval_env=eval_env,
                                      env_name=env_name,
                                      agent_name=agent_name,
                                      run_manager=run_manager)

    @classmethod
    def get_available_agents(cls):
        """Get list of available agents"""
        return list(cls.AGENTS.keys())