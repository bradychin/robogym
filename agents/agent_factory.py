# --------- Environment imports ---------#
from agents.ppo_agent import PPOAgent
from agents.sac_agent import SACAgent

# --------- Local imports ---------#
from utils.logger import logger
logger = logger(__name__)

# --------- Agent factory class ---------#
class AgentFactory:
    """Factory class for creating agent instances"""

    AGENTS = {
        'ppo': PPOAgent,
        'sac': SACAgent
    }

    # Define environment/agent compatibility
    AGENT_COMPATIBILITY = {
        'ppo': ['discrete', 'continuous'],
        'sac': ['continuous'],
        'dqn': ['discrete'],
        'a2c': ['discrete', 'continuous'],
        'td3': ['continuous']
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

    @classmethod
    def get_compatible_agents(cls, action_space_type: str):
        """
        Get list of agents compatible with action space type

        :param action_space_type: Type of action space (discrete or continuous)
        :return: List of compatible agents
        """

        compatible = []
        for agent_name, supported_spaces in cls.AGENT_COMPATIBILITY.items():
            if action_space_type in supported_spaces and agent_name in cls.AGENTS:
                compatible.append(agent_name)
        return compatible

    @classmethod
    def is_compatible(cls, agent_name: str, action_space_type: str):
        """
        Check agent/action space compatibility

        :param agent_name: Name of the agent
        :param action_space_type: Type of action space (discrete or continuous)
        :return: True of compatible, False otherwise
        """

        agent_name = agent_name.lower()
        if agent_name not in cls.AGENT_COMPATIBILITY:
            return False
        return action_space_type in cls.AGENT_COMPATIBILITY[agent_name]