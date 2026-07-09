from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from RL_env import DeepfakeEnv
import torch
from sklearn.model_selection import train_test_split
from TemporalTransformerEncoder import TemporalTransformerEncoder
from evaluator import evaluate_rl_agent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TemporalTransformerEncoder(
    feature_dim=768,
    num_frames=32,
    num_heads=8,
    num_layers=4,
    num_classes=2
).to(device)

# Load the state dictionary into the instantiated model
model.load_state_dict(torch.load("load/trained_transformer_model.pt"))

# Set the model to evaluation mode (important for inference, especially with Dropout/BatchNorm)
model.eval()

print("Model loaded successfully and set to evaluation mode.")
# ==========================================
# TRAINING THE PPO AGENT
# ==========================================
if __name__ == "__main__":
    
    features = torch.load("load/your/merged_features.pt") 
    video_ids = list(features.keys())
    train_video_ids, val_video_ids = train_test_split(video_ids, test_size=0.2, random_state=42)
    
    train_features_dict = {vid_id: features[vid_id] for vid_id in train_video_ids}
    val_features_dict = {vid_id: features[vid_id] for vid_id in val_video_ids}

    # Create label dictionaries
    train_labels_dict = {}
    for vid_id in train_video_ids:
        path_lower = vid_id.lower()
        # CELEB-DF LOGIC
        if "celeb-synthesis" in path_lower:
                train_labels_dict[vid_id] = 1 # Fake
        elif "celeb-real" in path_lower or "youtube-real" in path_lower:
                train_labels_dict[vid_id] = 0 # Real
       # FACEFORENSICS++ LOGIC
        elif "original" in path_lower or "youtube" in path_lower or "real" in path_lower:
                train_labels_dict[vid_id] = 0 # Real
        elif "manipulated" in path_lower or "fake" in path_lower or "swap" in path_lower or "face2face" in path_lower or "neuraltextures" in path_lower:
                train_labels_dict[vid_id] = 1 # Fake
        else:
            train_labels_dict[vid_id] = 1 # Default to fake if unidentifiable

    val_labels_dict = {}
    for vid_id in val_video_ids:
        path_lower = vid_id.lower()
        # CELEB-DF LOGIC
        if "celeb-synthesis" in path_lower:
            val_labels_dict[vid_id] = 1 # Fake
        elif "celeb-real" in path_lower or "youtube-real" in path_lower:
            val_labels_dict[vid_id] = 0 # Real

        # FACEFORENSICS++ LOGIC
        elif "original" in path_lower or "youtube" in path_lower or "real" in path_lower:
                val_labels_dict[vid_id] = 0 # Real
        elif "manipulated" in path_lower or "fake" in path_lower or "swap" in path_lower or "face2face" in path_lower or "neuraltextures" in path_lower:
                val_labels_dict[vid_id] = 1 # Fake
        else:
            val_labels_dict[vid_id] = 1 # Default to fake if unidentifiable

    print("Initializing Deepfake RL Environment...")
    env = DeepfakeEnv(train_features_dict, train_labels_dict, model, device)

    # Optional: Check if the environment follows Gym rules correctly
    check_env(env)

    print("Starting PPO Training...")
    # Initialize PPO model
    # MlpPolicy is used because state is a 1D feature vector (768,)
    ppo_agent = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-5, n_steps=2048)

    # Train the agent
    ppo_agent.learn(total_timesteps=100000)

    print("Training Complete!")
    # Save the trained agent
    ppo_agent.save("/content/drive/MyDrive/final year project/ppo_deepfake_frame_selector_2de_02")
    # To run this:
    # 1. Create a validation environment using your val_features_dict and val_labels_dict
    val_env = DeepfakeEnv(val_features_dict, val_labels_dict, model, device)

    # 2. Pass trained agent and the validation environment
    evaluate_rl_agent(ppo_agent, val_env, num_episodes=len(val_features_dict))