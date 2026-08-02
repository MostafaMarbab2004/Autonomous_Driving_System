import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import glob
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from encoder import VAE
from ppo_agent import PPOAgent
from parameters import VAE_LATENT_DIM, VAE_BETA
from settings import VAE_CHECKPOINT, PPO_CHECKPOINT, IMG_WIDTH, IMG_HEIGHT

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("GPU detected! Using CUDA.")
else:
    DEVICE = torch.device("cpu")
    print("WARNING: GPU not found or Torch not compiled with CUDA. Using CPU.")

class RawDemoDataset(Dataset):
    """ Loads all pkl files and applies basic transforms """
    def __init__(self, data_list):
        self.rgbs = []
        self.lidars = []
        self.states = []
        self.actions = []
        
        for d in data_list:
            # rgb is (128, 128, 3) float32 in [0, 1]
            # PyTorch expects (C, H, W) so we permute the axes
            img_t = torch.tensor(d['rgb'], dtype=torch.float32).permute(2, 0, 1)
            self.rgbs.append(img_t)
            self.lidars.append(torch.tensor(d['lidar'].flatten(), dtype=torch.float32))
            self.states.append(torch.tensor(d['state'], dtype=torch.float32))
            self.actions.append(torch.tensor(d['action'], dtype=torch.float32))
            
    def __len__(self):
        return len(self.rgbs)
        
    def __getitem__(self, idx):
        return self.rgbs[idx], self.states[idx], self.lidars[idx], self.actions[idx]

class BCDataset(Dataset):
    """ Final dataset holding the exact observation vectors PPO expects """
    def __init__(self, latents, states, lidars, actions):
        self.latents = latents
        self.states = states
        self.lidars = lidars
        self.actions = actions
        
    def __len__(self):
        return len(self.latents)
        
    def __getitem__(self, idx):
        # Concat: z (latent_dim), state (6), lidar (flattened)
        # This perfectly mimics `PPOAgent.make_obs`
        obs = torch.cat([self.latents[idx], self.states[idx], self.lidars[idx]], dim=0)
        return obs, self.actions[idx]


def load_data(max_episodes=100):
    files = glob.glob("demonstrations/*.pkl")
    
    # Limit to the most recent episodes to prevent Out of Memory (RAM) errors
    if len(files) > max_episodes:
        print(f"Found {len(files)} episodes. Limiting to latest {max_episodes} to save memory.")
        files = sorted(files)[-max_episodes:]
        
    all_data = []
    for f in files:
        with open(f, 'rb') as file:
            data = pickle.load(file)
            all_data.extend(data)
    print(f"Loaded {len(all_data)} transitions from {len(files)} episodes.")
    return all_data

def precompute_latents(vae, dataloader):
    print("Pre-computing VAE latents for all frames (this makes training much faster)...")
    latents = []
    vae.eval()
    with torch.no_grad():
        for rgbs, _, _, _ in dataloader:
            rgbs = rgbs.to(DEVICE)
            z = vae.encode(rgbs)
            latents.append(z.cpu())
    return torch.cat(latents, dim=0)

def main():
    print("="*50)
    print("Starting Behavioral Cloning Pre-training")
    print("="*50)
    
    # 1. Load raw data
    raw_data = load_data()
    if len(raw_data) == 0:
        print("No data found in demonstrations/ directory.")
        return
        
    raw_dataset = RawDemoDataset(raw_data)
    raw_loader = DataLoader(raw_dataset, batch_size=128, shuffle=False)
    
    # 2. Load VAE
    if not os.path.exists(VAE_CHECKPOINT):
        print(f" VAE Checkpoint missing at {VAE_CHECKPOINT}")
        return
        
    print(f"Loading VAE from {VAE_CHECKPOINT}")
    vae = VAE(latent_dim=VAE_LATENT_DIM, beta=VAE_BETA).to(DEVICE)
    ckpt = torch.load(VAE_CHECKPOINT, map_location=DEVICE)
    vae.load_state_dict(ckpt["state_dict"])
    
    # 3. Precompute Latents
    latents = precompute_latents(vae, raw_loader)
    
    # Extract tensors
    states = torch.stack(raw_dataset.states)
    lidars = torch.stack(raw_dataset.lidars)
    actions = torch.stack(raw_dataset.actions)
    
    # Create final BC dataloader
    bc_dataset = BCDataset(latents, states, lidars, actions)
    bc_loader = DataLoader(bc_dataset, batch_size=128, shuffle=True)
    
    # 4. Load PPO Agent
    print("\nInitializing PPO Agent...")
    agent = PPOAgent()
    
    # We only train the Actor. We want its output to match your actions!
    optimizer = optim.Adam(agent.actor.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    EPOCHS = 100
    print(f"Training Actor for {EPOCHS} epochs on {DEVICE}...")
    agent.actor.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for obs, action in bc_loader:
            obs = obs.to(DEVICE)
            action = action.to(DEVICE)
            
            # Forward pass through the Actor network
            dist = agent.actor(obs)
            
            # The Actor outputs a Normal distribution, and during driving we use tanh(mean)
            # So we compare the squashed mean prediction against your actual action
            predicted_action = torch.tanh(dist.mean)
            
            loss = loss_fn(predicted_action, action)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(bc_loader)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{EPOCHS} | Mean Squared Error: {avg_loss:.4f}")
        
    print("\n Training Complete!")
    agent.save(PPO_CHECKPOINT)
    print(f" Saved pre-trained weights to {PPO_CHECKPOINT}")
    print("\nYou can now run 'main_runner.py' to let the agent drive and fine-tune itself with PPO!")

if __name__ == '__main__':
    main()
