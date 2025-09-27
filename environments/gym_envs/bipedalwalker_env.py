# --------- Standard library imports ---------#

# --------- Third-party imports ---------#
from environments.base_environment import BaseEnvironment

# --------- Local imports ---------#

class BipedalWalkerEnv(BaseEnvironment):
    def __init__(self, render_mode='rgb_array'):
        super().__init__('BipedalWalker-v3', render_mode)