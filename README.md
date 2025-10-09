# RoboGym - Reinforcement Learning Training Pipeline

A modular, extensible framework for training and evaluating reinforcement learning agents on OpenAI Gymnasium environments.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Features

- **Modular Architecture**: Factory pattern for easy extension of environments and agents
- **Comprehensive Logging**: Automatic logging of training runs, evaluations, and demos
- **TensorBoard Integration**: Real-time training visualization
- **Smart Model Management**: Automatic best model selection and checkpointing
- **Interactive Demos**: Visual demonstrations of trained agents
- **Evaluation Framework**: Detailed performance metrics and reporting
- **Flexible Configuration**: YAML-based configuration system

## Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/bradychin/robogym.git
cd robogym

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Interactive mode (recommended for beginners)
python main.py

# Command-line mode (coming soon)
python main.py --env bipedalwalker --agent ppo --action train