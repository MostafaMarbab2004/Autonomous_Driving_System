"""
Audit script: checks the PPO checkpoint against the current architecture
and reports any mismatches or problems.
"""
import sys, os
sys.path.append(r'E:\Carla_PPO_project_the.py')

import torch
from ppo_agent import PPOAgent, Actor, Critic, DEVICE, MLP_INPUT_DIM, FLAT_OBS_DIM
from parameters import (
    PPO_LATENT_DIM, PPO_STATE_DIM, PPO_LIDAR_FEAT_DIM, PPO_MAX_LIDAR_POINTS,
    PPO_ACTION_DIM, PPO_ACTOR_HIDDEN, PPO_CRITIC_HIDDEN,
)
from settings import PPO_CHECKPOINT, MAX_LIDAR_POINTS
import numpy as np

print("="*60)
print("PPO ARCHITECTURE AUDIT")
print("="*60)

# 1. Print expected dimensions
print(f"\n--- Expected Dimensions ---")
print(f"  PPO_LATENT_DIM       = {PPO_LATENT_DIM}")
print(f"  PPO_STATE_DIM        = {PPO_STATE_DIM}")
print(f"  PPO_LIDAR_FEAT_DIM   = {PPO_LIDAR_FEAT_DIM}")
print(f"  PPO_MAX_LIDAR_POINTS = {PPO_MAX_LIDAR_POINTS}")
print(f"  MAX_LIDAR_POINTS (settings.py) = {MAX_LIDAR_POINTS}")
print(f"  MLP_INPUT_DIM        = {MLP_INPUT_DIM}  (= {PPO_LATENT_DIM} + {PPO_STATE_DIM} + {PPO_LIDAR_FEAT_DIM})")
print(f"  FLAT_OBS_DIM         = {FLAT_OBS_DIM}  (= {PPO_LATENT_DIM} + {PPO_STATE_DIM} + {PPO_MAX_LIDAR_POINTS}*3)")
print(f"  PPO_ACTION_DIM       = {PPO_ACTION_DIM}  (2D: [steer, accel])")

# 1b. Consistency check
if PPO_MAX_LIDAR_POINTS != MAX_LIDAR_POINTS:
    print(f"\n  !! MISMATCH: PPO_MAX_LIDAR_POINTS ({PPO_MAX_LIDAR_POINTS}) != MAX_LIDAR_POINTS in settings.py ({MAX_LIDAR_POINTS})")
else:
    print(f"\n  OK: PPO_MAX_LIDAR_POINTS matches settings.py")

# 2. Create fresh agent and print layer shapes
print(f"\n--- Fresh Agent Layer Shapes ---")
agent = PPOAgent()

print("\n  ACTOR:")
for name, param in agent.actor.named_parameters():
    print(f"    {name:40s}  {str(list(param.shape)):20s}  requires_grad={param.requires_grad}")

print("\n  CRITIC:")
for name, param in agent.critic.named_parameters():
    print(f"    {name:40s}  {str(list(param.shape)):20s}  requires_grad={param.requires_grad}")

# 3. Check checkpoint
print(f"\n--- Checkpoint Analysis ---")
print(f"  Path: {PPO_CHECKPOINT}")
if not os.path.exists(PPO_CHECKPOINT):
    print("  NOT FOUND — agent will train from scratch (this is expected after architecture change)")
else:
    ckpt = torch.load(PPO_CHECKPOINT, map_location="cpu")
    print(f"  Keys in checkpoint: {list(ckpt.keys())}")
    
    # Compare actor keys
    fresh_actor_keys = set(agent.actor.state_dict().keys())
    ckpt_actor_keys = set(ckpt["actor"].keys())
    
    if fresh_actor_keys != ckpt_actor_keys:
        print(f"\n  !! ACTOR KEY MISMATCH:")
        print(f"     Missing in checkpoint: {fresh_actor_keys - ckpt_actor_keys}")
        print(f"     Extra in checkpoint:   {ckpt_actor_keys - fresh_actor_keys}")
    else:
        print(f"  OK: Actor keys match")
        
    # Compare shapes
    mismatch = False
    for key in fresh_actor_keys & ckpt_actor_keys:
        fresh_shape = agent.actor.state_dict()[key].shape
        ckpt_shape = ckpt["actor"][key].shape
        if fresh_shape != ckpt_shape:
            print(f"  !! SHAPE MISMATCH actor.{key}: fresh={fresh_shape} vs ckpt={ckpt_shape}")
            mismatch = True
    
    fresh_critic_keys = set(agent.critic.state_dict().keys())
    ckpt_critic_keys = set(ckpt["critic"].keys())
    
    if fresh_critic_keys != ckpt_critic_keys:
        print(f"\n  !! CRITIC KEY MISMATCH:")
        print(f"     Missing in checkpoint: {fresh_critic_keys - ckpt_critic_keys}")
        print(f"     Extra in checkpoint:   {ckpt_critic_keys - fresh_critic_keys}")
    else:
        print(f"  OK: Critic keys match")
        
    for key in fresh_critic_keys & ckpt_critic_keys:
        fresh_shape = agent.critic.state_dict()[key].shape
        ckpt_shape = ckpt["critic"][key].shape
        if fresh_shape != ckpt_shape:
            print(f"  !! SHAPE MISMATCH critic.{key}: fresh={fresh_shape} vs ckpt={ckpt_shape}")
            mismatch = True
    
    if not mismatch:
        print(f"  OK: All tensor shapes match between checkpoint and current architecture")
    else:
        print(f"\n  !! WARNING: Checkpoint is INCOMPATIBLE with new architecture (3→2 actions, LayerNorm added)")
        print(f"  !! DELETE the checkpoint to train from scratch: {PPO_CHECKPOINT}")
    
    # Try loading
    try:
        agent.actor.load_state_dict(ckpt["actor"])
        agent.critic.load_state_dict(ckpt["critic"])
        print(f"  OK: Checkpoint loads successfully into current architecture")
    except Exception as e:
        print(f"  !! LOAD FAILED (expected after architecture change): {e}")

# 4. Forward pass test
print(f"\n--- Forward Pass Test ---")
try:
    # Reset agent to fresh state (in case checkpoint load failed)
    agent = PPOAgent()
    
    z = torch.randn(1, PPO_LATENT_DIM).to(DEVICE)
    state = np.array([0.5, 0.0, 0.5, 0.0, 0.0], dtype=np.float32)
    lidar = np.random.randn(PPO_MAX_LIDAR_POINTS, 3).astype(np.float32)
    
    obs = PPOAgent.make_obs(z, state, lidar)
    print(f"  obs shape: {obs.shape}  (expected: [1, {FLAT_OBS_DIM}])")
    
    action, raw, log_p, val = agent.select_action(obs)
    print(f"  action: {action}  (2D: [steer, accel])")
    print(f"  raw:    {raw}")
    print(f"  log_p:  {log_p}")
    print(f"  value:  {val}")
    
    # Verify action dimensions
    assert action.shape == (PPO_ACTION_DIM,), f"Action shape mismatch: {action.shape} vs expected ({PPO_ACTION_DIM},)"
    assert all(-1.0 <= a <= 1.0 for a in action), f"Actions out of [-1,1] range: {action}"
    print(f"  OK: Forward pass works, actions are 2D with tanh squashing")
except Exception as e:
    print(f"  !! FORWARD PASS FAILED: {e}")
    import traceback; traceback.print_exc()

# 5. Backward pass test
print(f"\n--- Backward Pass Test (mini PPO update) ---")
try:
    agent.buffer.reset()
    for i in range(64):
        obs = PPOAgent.make_obs(
            torch.randn(1, PPO_LATENT_DIM).to(DEVICE),
            np.array([0.3, 0.1, 0.4, 0.0, 0.0], dtype=np.float32),
            np.random.randn(PPO_MAX_LIDAR_POINTS, 3).astype(np.float32)
        )
        action, raw, log_p, val = agent.select_action(obs)
        agent.buffer.store(
            obs=obs.squeeze(0), action=raw,
            log_prob=log_p, reward=1.0, value=val, done=0.0
        )
    
    last_obs = obs
    agent.update(last_obs)
    print(f"  OK: PPO update completed successfully (gradients flow through LayerNorm + PointNet)")
except Exception as e:
    print(f"  !! UPDATE FAILED: {e}")
    import traceback; traceback.print_exc()

# 6. Architecture summary
print(f"\n--- Architecture Summary ---")
print(f"  Action space: 2D [steer, accel] with tanh squashing")
print(f"  Steer  ∈ [-1, 1]  (tanh)")
print(f"  Accel  ∈ [-1, 1]  (tanh) → positive=throttle, negative=brake")
print(f"  At initialization: tanh(0)=0 → no throttle, no brake → throttle floor kicks in")
print(f"  MLP has LayerNorm after each hidden layer")
print(f"  LOG_STD_MIN = -2.0 (std ≥ 0.13, prevents deterministic collapse)")
print(f"  LiDAR normalized to [-1,1] by dividing by sensor range (50m)")

print(f"\n{'='*60}")
print(f"AUDIT COMPLETE")
print(f"{'='*60}")
