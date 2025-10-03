# --------- Environment imports ---------#
from agents.ppo_agent import PPOAgent

# --------- Local imports ---------#
from utils.logger import get_logger
logger = get_logger(__name__)

# --------- Agent factory class ---------#
class AgentFactory:
    """Factory class for creating agent instances"""

    AGENTS = {
        'ppo': PPOAgent
    }

    @classmethod
    def create(cls, agent_name: str, vec_env, eval_env, tensorboard_log=None, env_name=None):
        """
        Create agent instance

        :param agent_name: Name of the agent to create
        :param vec_env: Vectorized training environment
        :param eval_env: Evaluation environment
        :param tensorboard_log: Path for tensorboard logs
        :param env_name: Name of the environment
        :return: BaseAgent instance
        """

        agent_name = agent_name.lower()
        if agent_name not in cls.AGENTS:
            available = ', '.join(cls.AGENTS.keys())
            raise ValueError (f'Agent "{agent_name}" not available. Available agents: {available}')

        logger.info(f'Creating agent {agent_name}')
        return cls.AGENTS[agent_name](vec_env, eval_env, env_name, agent_name)

    @classmethod
    def get_available_agents(cls):
        """Get list of available agents"""
        return list(cls.AGENTS.keys())

    @classmethod
    def register_agents(cls, name: str, agent_class):
        """
        Register a new agent

        :param name: Agent name to register
        :param agent_class: Agent class to register
        """
        cls.AGENTS[name.lower()] = agent_class