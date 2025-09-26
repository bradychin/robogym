# RoboGym

## Core Goals:
1. Multiple Environments: BipedalWalker, Ant, CartPole, Humanoid, etc.
2. Multiple Algorithms: PPO, SAC, A2C, DQN, TD3, etc.
3. User Choice Interface: Pick any environment + algorithm combination
4. Demo Existing Models: Check if trained model exists and demo without retraining
5. Add custom environments

## User Workflow:

1. Choose Environment: "BipedalWalker, Ant, CartPole...?"
2. Choose Algorithm: "PPO, SAC, A2C, DQN...?"  
3. Choose Action: "Train, Evaluate, Demo?"
4. Execute with selected Agent class

## Development Approach:
1. Start with PPOAgent (BipedalWalker)
3. Add BaseAgent abstraction
4. Gradually add new algorithms
5. Build user interface for choices
6. Create custom environments

/robogym/
│
├── agents/           # BaseAgent and all algorithm classes
│   ├── base_agent.py
│   ├── ppo_agent.py
│   └── sac_agent.py
│
├── environments/     # Environment wrappers
│   ├── gym_envs/     # CartPole, BipedalWalker, etc.
│   └── robots/       # Bravo, 3-link arm, etc.
│
├── scripts/          # train.py, evaluate.py, demo.py
└── configs/          # JSON/YAML configs for experiments
