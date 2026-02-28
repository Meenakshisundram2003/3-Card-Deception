import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO


def simulate_3card_physics(user_choice):
    cards = [0, 1, 2]
    others = [i for i in range(3) if i != user_choice]
    
    val_a = cards[others[0]]
    val_b = cards[others[1]]
    cards[others[0]] = val_b
    cards[others[1]] = val_a
    
    return cards


def run_feasibility_check():
    obs_seen = []
    for choice in range(3):
        cards = simulate_3card_physics(choice)
        obs_seen.append(cards[0])
    
    if len(set(obs_seen)) == 3:
        print(f"Single-Flip Strategy (Index 0) SUCCESS.")
        print(f"Observations {obs_seen} are unique.")
        print("Proceeding with Single-Flip.\n")
    else:
        print("Ambiguity Detected. Switching strategy.")


class ScoutEnv(gym.Env):
    def __init__(self):
        super(ScoutEnv, self).__init__()
        self.n_cards = 3
        # ACTION: Choose 1 Index (0, 1, or 2)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        chosen_index = int(action)
        possible_observations = []
        
        # Simulate Multiverse
        for choice in range(3):
            cards = simulate_3card_physics(choice)
            obs = cards[chosen_index]
            possible_observations.append(obs)
            
        if len(set(possible_observations)) == 3:
            reward = 10.0
        else:
            reward = -10.0
        return np.array([0.0], dtype=np.float32), reward, True, False, {}


class DetectiveEnv(gym.Env):
    def __init__(self, sensor_index):
        super(DetectiveEnv, self).__init__()
        self.sensor_index = sensor_index
        # Action: Guess User Choice (0, 1, or 2)
        self.action_space = spaces.Discrete(3)
        # Observation: We see 1 card value (0, 1, or 2)
        self.observation_space = spaces.Box(low=0, high=2, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.user_choice = np.random.randint(0, 3)
        cards = simulate_3card_physics(self.user_choice)
        obs = [cards[self.sensor_index]]
        return np.array(obs, dtype=np.float32), {}

    def step(self, action):
        if action == self.user_choice:
            reward = 1.0
        else:
            reward = -1.0
        return np.array([0], dtype=np.float32), reward, True, False, {}


if __name__ == "__main__":
    
    run_feasibility_check()

    # Train Brain 1
    env1 = ScoutEnv()
    brain1 = PPO("MlpPolicy", env1, verbose=0)
    brain1.learn(total_timesteps=5000)
    
    brain1.save("brain_1_3card")
    print("Brain 1 Saved as 'brain_1_3card.zip'.")
    
    # Get Decision
    obs, _ = env1.reset()
    action, _ = brain1.predict(obs)
    OPTIMAL_INDEX = int(action)
    print(f"Optimal Index is {OPTIMAL_INDEX}\n")

    # Train Brain 2
    env2 = DetectiveEnv(sensor_index=OPTIMAL_INDEX)
    brain2 = PPO("MlpPolicy", env2, verbose=0)
    brain2.learn(total_timesteps=10000)
    
    brain2.save("brain_2_3card")
    print("Brain 2 Saved as 'brain_2_3card.zip'.\n")