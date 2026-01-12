# RoboGym - Reinforcement Learning Training Pipeline

A modular, extensible framework for training and evaluating reinforcement learning agents across multiple OpenAI Gymnasium and PyBullet environments.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Features

- **6 RL Algorithms**: PPO, SAC, DQN, TD3, A2C, DDPG
- **10 Environments**: From simple control (CartPole) to complex locomotion (Walker2D, Humanoid)
- **Factory Pattern Architecture**: Easy extension of environments and agents
- **Hyperparameter Optimization**: Optuna integration for automated tuning
- **Comprehensive Logging**: TensorBoard integration with automatic model checkpointing
- **Interactive Demos**: Visual demonstrations of trained agents
- **Apple Silicon Optimized**: Native MPS acceleration for M-series Macs

## 🚀 Quick Start

### Prerequisites

- **Python 3.10, 3.11, or 3.12**
- **macOS** (Apple Silicon optimized) or **Linux**
- **Git**

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/bradychin/robogym.git
cd robogym

# 2. Run the automated setup script
chmod +x setup.sh
./setup.sh

# The script will:
# - Check Python version
# - Create virtual environment
# - Install all dependencies
# - Verify installation
```

### Manual Installation (Alternative)

If you prefer manual installation:

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Upgrade pip
pip install --upgrade pip setuptools wheel

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install in editable mode
pip install -e .

# 5. Verify installation
python test_installation.py
```

## 📖 Usage

### Training a Model

```bash
# Interactive mode (recommended)
python main.py

# Follow the prompts to:
# 1. Select environment (e.g., Walker2D, BipedalWalker)
# 2. Select algorithm (e.g., PPO, SAC)
# 3. Choose action (train, evaluate, demo)
```

### Monitoring Training

```bash
# Start TensorBoard (in a separate terminal)
tensorboard --logdir=./runs

# Open browser to http://localhost:6006
```

### Project Structure

```
robogym/
├── agents/              # RL algorithm implementations
├── environments/        # Environment wrappers and factory
├── config/             # YAML configuration files
├── utils/              # Utilities (logging, model I/O, etc.)
├── hyperparameter_tuning/  # Optuna optimization scripts
├── runs/               # Training logs and models (auto-generated)
├── log/                # Application logs (auto-generated)
├── main.py             # Main training pipeline
├── requirements.txt    # Python dependencies
├── pyproject.toml      # Package configuration
└── setup.sh            # Automated installation script
```

## 🎮 Supported Environments

### Classic Control
- **Acrobot-v1**: Swing up the two-link robot
- **CartPole-v1**: Balance a pole on a moving cart
- **Pendulum-v1**: Swing up and balance an inverted pendulum

### Box2D (Continuous Control)
- **LunarLander-v3**: Land a spacecraft (discrete actions)
- **LunarLanderContinuous-v3**: Land a spacecraft (continuous actions)
- **BipedalWalker-v3**: Walk a 2D bipedal robot

### PyBullet 3D Robotics
- **AntBulletEnv-v0**: Quadruped locomotion
- **HalfCheetahBulletEnv-v0**: Fast forward locomotion
- **HopperBulletEnv-v0**: Single-leg hopping robot
- **Walker2DBulletEnv-v0**: 2D bipedal walking

## 🤖 Supported Algorithms

| Algorithm | Type | Action Space | Best For |
|-----------|------|--------------|----------|
| **PPO** | On-policy | Discrete/Continuous | General purpose, stable |
| **SAC** | Off-policy | Continuous | Sample efficiency, robustness |
| **TD3** | Off-policy | Continuous | Continuous control |
| **A2C** | On-policy | Discrete/Continuous | Fast training |
| **DQN** | Off-policy | Discrete | Discrete action spaces |
| **DDPG** | Off-policy | Continuous | Continuous control |

## ⚙️ Configuration

All environment and algorithm configurations are in `config/`:

```yaml
# Example: config/walker2d_config.yaml
training:
  total_timesteps: 2_000_000
  n_eval_episodes: 10
  eval_freq: 50_000
  
ppo:
  learning_rate: 0.0003
  n_steps: 2048
  batch_size: 64
  # ... more hyperparameters
```

## 🔧 Hyperparameter Tuning

```bash
cd hyperparameter_tuning
python optimize_hyperparameters.py

# Edit the script to configure:
# - Environment
# - Algorithm
# - Number of trials
# - Search space
```

## 🐛 Troubleshooting

### Import Errors

```bash
# Verify installation
python test_installation.py

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### PyBullet GUI Issues on macOS

If you see GUI warnings:
```bash
# PyBullet will fall back to CPU rendering automatically
# No action needed - training will work normally
```

### MPS (Apple Silicon) Not Available

```bash
# Check PyTorch installation
python -c "import torch; print(torch.backends.mps.is_available())"

# If False, reinstall PyTorch:
pip uninstall torch torchvision
pip install torch torchvision
```

## 📊 Example Results

| Environment | Algorithm | Steps | Final Reward |
|-------------|-----------|-------|--------------|
| Walker2D | PPO | 2M | 2403 |
| BipedalWalker | SAC | 1M | 290+ |
| LunarLander | DQN | 500K | 200+ |

## 🤝 Contributing

Contributions welcome! This is a portfolio project, but improvements and extensions are appreciated.

## 📝 License

MIT License - see LICENSE file for details

## 🎓 Learning Resources

This project implements concepts from:
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [OpenAI Spinning Up in Deep RL](https://spinningup.openai.com/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

## 📬 Contact

Brady Chin - [GitHub](https://github.com/bradychin)

---

**Note**: This is a portfolio project demonstrating clean architecture, production practices, and deep RL understanding. Designed for easy extension to custom robotics environments.
