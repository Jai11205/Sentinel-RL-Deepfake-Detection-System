#for reinforcement learning agent and envronment
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import torch

class DeepfakeEnv(gym.Env):
    """
    Custom Environment that follows gymnasium interface.
    The RL agent acts as an adaptive frame selector.
    """
    def __init__(self, features_dict, labels_dict, pretrained_transformer, device, max_frames=32):
        super(DeepfakeEnv, self).__init__()

        self.features_dict = features_dict
        self.video_ids = list(features_dict.keys())
        self.labels_dict = labels_dict
        self.model = pretrained_transformer
        self.model.eval() # MUST be in eval mode
        self.device = device
        self.max_frames = max_frames

        # Hyperparameters for the Reward Function
        self.reward_correct = 10.0
        self.reward_wrong = -10.0
        self.penalty_per_frame = -0.2 # Cost of processing a frame

        # Action Space: 0 (Skip Frame), 1 (Select Frame)
        self.action_space = spaces.Discrete(2)

        # Observation Space: The 768-dimensional feature of the current frame
        # (Gym requires numpy arrays, so we use Box)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(768,), dtype=np.float32)

        # Episode tracking variables
        self.current_video_id = None
        self.current_video_features = None
        self.current_true_label = None
        self.current_frame_idx = 0

        self.selected_indices = []
        self.selected_features = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 1. Pick a random video for the new episode
        self.current_video_id = random.choice(self.video_ids)
        self.current_video_features = self.features_dict[self.current_video_id]
        self.current_true_label = self.labels_dict[self.current_video_id]

        # 2. Reset tracking variables
        self.current_frame_idx = 0
        self.selected_indices = []
        self.selected_features = []

        # 3. Get the first frame's feature to show the agent
        first_frame = self.current_video_features[self.current_frame_idx].numpy()

        info = {}
        return first_frame, info

    def step(self, action):
        # 1. Process the agent's action for the current frame
        if action == 1:
            self.selected_indices.append(self.current_frame_idx)
            self.selected_features.append(self.current_video_features[self.current_frame_idx])

        # 2. Move to the next frame
        self.current_frame_idx += 1

        # 3. Check if the episode is done (reached the end of the video)
        done = self.current_frame_idx >= len(self.current_video_features) or self.current_frame_idx >= self.max_frames

        reward = 0.0
        info = {}

        if done:
            # --- EPISODE END: CALCULATE THE FINAL REWARD ---

            # Edge Case: The agent skipped EVERY frame. Punish heavily.
            if len(self.selected_features) == 0:
                reward = self.reward_wrong - 5.0
            else:
                # Prepare data for the Transformer
                x_input = torch.stack(self.selected_features).unsqueeze(0).to(self.device) # Shape: (1, T_selected, 768)
                indices_input = torch.tensor(self.selected_indices).unsqueeze(0).to(self.device) # Shape: (1, T_selected)

                # Get the Transformer's prediction
                with torch.no_grad(): # No gradients needed for RL reward generation
                    logits = self.model(x_input, frame_indices=indices_input)
                    predicted_class = torch.argmax(logits, dim=1).item()

                # Calculate the core reward
                if predicted_class == self.current_true_label:
                    reward += self.reward_correct
                else:
                    reward += self.reward_wrong

                # Apply the computational penalty for the number of frames used
                compute_penalty = len(self.selected_indices) * self.penalty_per_frame
                reward += compute_penalty

                info['accuracy'] = 1 if predicted_class == self.current_true_label else 0
                info['frames_used'] = len(self.selected_indices)

            # Create a dummy next state since the episode is over
            next_state = np.zeros(768, dtype=np.float32)

        else:
            # Episode is still ongoing, provide the next frame's features
            next_state = self.current_video_features[self.current_frame_idx].numpy()

        # Gymnasium returns: observation, reward, terminated, truncated, info
        return next_state, reward, done, False, info

