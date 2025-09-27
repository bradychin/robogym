# --------- Standard library imports ---------#

# --------- Third-party imports ---------#

# --------- Local imports ---------#
from .ppo_agent import PPOAgent

AVAILABLE_AGENTS = {
    'PPO': PPOAgent
}