# --------- Third-party imports ---------#
from environments.base_environment import BaseEnvironment

# --------- Bipedal Walker environment---------#
class BipedalWalkerEnv(BaseEnvironment):
    def __init__(self, render_mode='rgb_array'):
        super().__init__('BipedalWalker-v3', render_mode)