import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import glob

def main():
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # Find the most recent ppo_rewards csv
        files = glob.glob("ppo_rewards*.csv")
        if not files:
            print("No ppo_rewards CSV files found.")
            return
        csv_path = max(files, key=os.path.getmtime)
        print(f"Plotting most recent file: {csv_path}")
        
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found.")
        return
    
    df = pd.read_csv(csv_path)
    
    if df.empty:
        print("The CSV file is empty. Waiting for more data...")
        return
        
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot 1: Episode Reward and Average Reward
    axs[0].plot(df['episode'], df['reward'], alpha=0.3, label='Episode Reward', color='blue')
    if 'avg_reward' in df.columns:
        axs[0].plot(df['episode'], df['avg_reward'], label='Avg Reward (last 20)', color='red', linewidth=2)
    axs[0].set_title('PPO Episode Rewards')
    axs[0].set_ylabel('Reward')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot 2: Speed
    axs[1].plot(df['episode'], df['speed'], color='green')
    axs[1].set_title('Terminal Speed per Episode')
    axs[1].set_ylabel('Speed (km/h)')
    axs[1].grid(True)
    
    # Plot 3: Waypoint Progress
    if 'waypoint' in df.columns and 'total_waypoints' in df.columns:
        # Avoid division by zero
        progress = df.apply(lambda row: (row['waypoint'] / row['total_waypoints'] * 100) if row['total_waypoints'] > 0 else 0, axis=1)
        axs[2].plot(df['episode'], progress, color='purple')
        axs[2].set_title('Waypoint Progress (%)')
        axs[2].set_xlabel('Episode')
        axs[2].set_ylabel('Progress %')
        axs[2].grid(True)
    
    plt.tight_layout()
    plt.savefig("ppo_performance.png")
    print("Plot saved as ppo_performance.png")
    plt.show()

if __name__ == "__main__":
    main()
