# RoboGym - Reinforcement Learning Training Pipeline

A modular framework for training RL agents on Gymnasium and PyBullet environments.

## Features

- **6 RL Algorithms**: PPO, SAC, DQN, TD3, A2C, DDPG
- **10 Environments**: CartPole, Acrobot, Pendulum, LunarLander, BipedalWalker, Ant, HalfCheetah, Hopper, Walker2D
- **Factory Pattern**: Easy to extend with new environments and agents
- **TensorBoard Integration**: Real-time training visualization
- **Hyperparameter Tuning**: Optuna optimization support

## Installation

### Prerequisites
- Python 3.11+
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (installation guide)

### Setup

```bash
# Clone repository
git clone https://github.com/bradychin/robogym.git
cd robogym

# Create environment
conda env create -f environment.yml # The environment includes Python 3.11 and all required ML and simulation dependencies.
conda activate robogym

# Verify installation
python test_installation.py
```

## Usage

```bash
# Train a model
python main.py

# Monitor training
tensorboard --logdir=./runs
```

Follow the interactive prompts to:
1. Select environment
2. Select algorithm  
3. Choose action (train/evaluate/demo) # Will need to retrain a model to run evaluations and demos

## Project Structure

```
robogym/
├── agents/           # RL algorithm implementations
├── environments/     # Environment wrappers
├── config/          # YAML configurations
├── utils/           # Helper functions
└── main.py          # Entry point
```

## Configuration

Edit YAML files in `config/` to adjust hyperparameters:

```yaml
# Example: config/walker2d_config.yaml
ppo:
  learning_rate: 0.0003
  n_steps: 2048
  batch_size: 64
```

## Requiremen
- stable-baselines3
- gymnasium
- pybullet
- torch (with MPS support on Apple Silicon)
- optuna
- tensorboard

See `environment.yml` for full dependencies.
