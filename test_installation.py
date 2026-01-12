#!/usr/bin/env python3
"""
Installation verification script for RoboGym
Tests that all required packages and environments are working correctly
"""

import sys
from typing import Tuple

# Import to register PyBullet environments
import pybullet_envs_gymnasium

def test_import(module_name: str, package_name: str = None) -> Tuple[bool, str]:
    """Test if a module can be imported"""
    try:
        __import__(module_name)
        return True, f"✓ {package_name or module_name}"
    except ImportError as e:
        return False, f"✗ {package_name or module_name}: {str(e)}"

def test_environment(env_id: str) -> Tuple[bool, str]:
    """Test if an environment can be created"""
    try:
        import gymnasium as gym
        env = gym.make(env_id)
        env.close()
        return True, f"✓ {env_id}"
    except Exception as e:
        return False, f"✗ {env_id}: {str(e)}"

def main():
    print("=" * 60)
    print("RoboGym Installation Verification")
    print("=" * 60)
    print()
    
    all_passed = True
    
    # Test core packages
    print("Testing Core Packages:")
    print("-" * 60)
    core_packages = [
        ("stable_baselines3", "stable-baselines3"),
        ("gymnasium", "gymnasium"),
        ("torch", "torch"),
        ("pybullet", "pybullet"),
        ("optuna", "optuna"),
        ("tensorboard", "tensorboard"),
        ("yaml", "pyyaml"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
    ]
    
    for module, package in core_packages:
        passed, msg = test_import(module, package)
        print(msg)
        all_passed = all_passed and passed
    
    print()
    
    # Test PyTorch MPS (Apple Silicon acceleration)
    print("Testing PyTorch Configuration:")
    print("-" * 60)
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ MPS available: {torch.backends.mps.is_available()}")
        print(f"✓ MPS built: {torch.backends.mps.is_built()}")
        if torch.backends.mps.is_available():
            print("  → Apple Silicon GPU acceleration enabled!")
    except Exception as e:
        print(f"✗ PyTorch check failed: {e}")
        all_passed = False
    
    print()
    
    # Test environments
    print("Testing Gymnasium Environments:")
    print("-" * 60)
    
    test_envs = [
        "CartPole-v1",
        "Acrobot-v1", 
        "Pendulum-v1",
        "LunarLander-v3",
        "LunarLanderContinuous-v3",
        "BipedalWalker-v3",
    ]
    
    for env_id in test_envs:
        passed, msg = test_environment(env_id)
        print(msg)
        all_passed = all_passed and passed
    
    print()
    
    # Test PyBullet environments
    print("Testing PyBullet Environments:")
    print("-" * 60)
    
    pybullet_envs = [
        "AntBulletEnv-v0",
        "HalfCheetahBulletEnv-v0",
        "HopperBulletEnv-v0",
        "Walker2DBulletEnv-v0",
    ]
    
    for env_id in pybullet_envs:
        passed, msg = test_environment(env_id)
        print(msg)
        all_passed = all_passed and passed
    
    print()
    print("=" * 60)
    
    if all_passed:
        print("✓ All tests passed! Installation successful.")
        print("=" * 60)
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
