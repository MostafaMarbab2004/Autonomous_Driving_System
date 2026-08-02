import sys
import os

# Append project dir so imports work safely
sys.path.append(r'E:\Carla_PPO_project_the.py')

import numpy as np
import torch
from parameters import PPO_LATENT_DIM, PPO_MINIBATCH_SIZE
from ppo_agent import PPOAgent

def main():
    try:
        print("1. Initializing PPOAgent (includes PointNet Extractor)...")
        agent = PPOAgent()

        print("\n2. Creating mock environment observations...")
        z = torch.randn(1, PPO_LATENT_DIM)
        state = np.array([0.5, -0.1], dtype=np.float32)
        lidar = np.random.rand(1024, 3).astype(np.float32)

        print("\n3. Testing PPOAgent.make_obs(z, state, lidar)...")
        obs = PPOAgent.make_obs(z, state, lidar)
        print(f"   -> Flattened obs tensor shape: {obs.shape}")

        print("\n4. Testing Actor/Critic forward pass (select_action)...")
        action, raw, log_prob, value = agent.select_action(obs)
        print(f"   -> Selected Action: {action}")
        
        print(f"\n5. Populating RolloutBuffer with {PPO_MINIBATCH_SIZE} dummy steps to test backprop...")
        for i in range(PPO_MINIBATCH_SIZE):
            agent.buffer.store(
                obs      = obs.squeeze(0),
                action   = raw,
                log_prob = log_prob,
                reward   = 1.0,
                value    = value,
                done     = 0.0,
            )
            
        last_obs = obs
        print("\n6. Testing PPO update() backward pass (loss calculation, PointNet gradients)...")
        agent.update(last_obs)
        print("   -> Update successful! PointNet and PPO handled gradients correctly.")

        print("\n====== SUCCESS: ENTIRE PIPELINE IS TENSOR-ERROR FREE! ======")
    except Exception as e:
        print("\n====== PIPELINE FAILED! ======")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
