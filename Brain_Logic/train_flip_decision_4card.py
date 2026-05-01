import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO


def simulate_4card_physics(user_choice):
    cards = [0, 1, 2, 3]
    others = [i for i in range(4) if i != user_choice]
    vals = [cards[i] for i in others]
    shifted = [vals[-1]] + vals[:-1] # Cyclic Shift
    for i, idx in enumerate(others):
        cards[idx] = shifted[i]
    return cards

def run_feasibility_check():

    obs_seen = []
    for choice in range(4):
        cards = simulate_4card_physics(choice)
        obs_seen.append(cards[0])
    
    if len(set(obs_seen)) < 4:
        print(f"Single-Flip Strategy (Index 0) FAILED.")
        print(f"Ambiguity Detected in observations {obs_seen}.")
        print("Switching to Multi-Flip Strategy.\n")
    else:
        print("Single Flip is enough.")


class ScoutEnv(gym.Env):
    def __init__(self):
        super(ScoutEnv, self).__init__()
        self.n_cards = 4
        self.action_space = spaces.Discrete(6)
        self.pair_map = {
            0: [0, 1], 1: [1, 2], 2: [2, 3],
            3: [3, 0], 4: [0, 2], 5: [1, 3]
        }
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        chosen_indices = self.pair_map[int(action)]
        possible_observations = []
        for choice in range(4):
            cards = simulate_4card_physics(choice)
            obs = tuple([cards[i] for i in chosen_indices])
            possible_observations.append(obs)
        if len(set(possible_observations)) == 4:
            reward = 10.0
        else:
            reward = -10.0
        return np.array([0.0], dtype=np.float32), reward, True, False, {}


class DetectiveEnv(gym.Env):
    def __init__(self, sensor_indices):
        super(DetectiveEnv, self).__init__()
        self.sensor_indices = sensor_indices
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=3, shape=(2,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.user_choice = np.random.randint(0, 4)
        cards = simulate_4card_physics(self.user_choice)
        obs = [cards[i] for i in self.sensor_indices]
        return np.array(obs, dtype=np.float32), {}

    def step(self, action):
        if action == self.user_choice:
            reward = 1.0
        else:
            reward = -1.0
        return np.array([0,0], dtype=np.float32), reward, True, False, {}


if __name__ == "__main__":
    
    run_feasibility_check()

    # Train Brain 1
    env1 = ScoutEnv()
    brain1 = PPO("MlpPolicy", env1, verbose=0)
    brain1.learn(total_timesteps=5000)
    
    brain1.save("brain_1_4card")
    print("Brain 1 Saved as 'brain_1_4card.zip'.")
    
    # Get Decision
    obs, _ = env1.reset()
    action, _ = brain1.predict(obs, deterministic=True)
    OPTIMAL_INDICES = env1.pair_map[int(action)]
    print(f"Optimal Indices are {OPTIMAL_INDICES}\n")

    # Train Brain 2
    env2 = DetectiveEnv(sensor_indices=OPTIMAL_INDICES)
    brain2 = PPO("MlpPolicy", env2, verbose=0)
    brain2.learn(total_timesteps=300000)
    
    brain2.save("brain_2_4Card")
    print("Brain 2 Saved as 'brain_2_4Card.zip'.\n")

    print("--- Testing Brain 2 Accuracy ---")
    correct_count = 0
    
    for true_choice in range(4):
        # 1. Simulate the physical table after the human shifts the cards
        cards = simulate_4card_physics(true_choice)
        
        # 2. Extract exactly what the robot's camera will see at the chosen spots
        obs = np.array([cards[i] for i in OPTIMAL_INDICES], dtype=np.float32)
        
        # 3. Ask Brain 2 to solve the trick (deterministic=True means no random guessing)
        predicted_choice, _ = brain2.predict(obs, deterministic=True)
        
        # 4. Check if Brain 2's prediction matches the true choice
        is_correct = (int(predicted_choice) == true_choice)
        if is_correct:
            correct_count += 1
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
            
        print(f"True Choice: {true_choice} | Camera Sees: {obs.tolist()} | Brain Predicts: {int(predicted_choice)} | {status}")

    print(f"\nTotal Accuracy: {correct_count}/4")
    if correct_count == 4:
        print("Brain 2 is 100% accurate! The logic is mathematically sound.")
    else:
        print("Brain 2 made a mistake. You need to increase total_timesteps further.")

    