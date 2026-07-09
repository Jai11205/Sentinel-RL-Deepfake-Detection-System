from TemporalTransformerEncoder import TemporalTransformerEncoder
from Encoder_Trainer import VideoFeatureDataset, custom_collate_fn, train_transformer, evaluate_model
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TemporalTransformerEncoder(
    feature_dim=768,
    num_frames=32,
    num_heads=8,
    num_layers=4
)
model.to(device)


# Dummy data test
batch_size = 2
seq_len = 5 # Let's say the RL agent selected 5 frames
feat_dim = 768

dummy_x = torch.randn(batch_size, seq_len, feat_dim).to(device)

# Simulate RL agent picking different frames for different videos in the batch
dummy_indices = torch.tensor([
        [0, 5, 10, 15, 20],  # Video 1 frame choices
        [2, 4, 8,  16, 31]   # Video 2 frame choices
    ]).to(device)

print("Testing forward pass with frame indices...")
logits = model(dummy_x, frame_indices=dummy_indices)
print("Logits shape:", logits.shape) # Should be (2, 2)
print("Success! The encoder is ready for training.")
    
# CrossEntropyLoss is standard for multi-class/binary classification with logits
criterion = nn.CrossEntropyLoss()

# AdamW is highly recommended for Transformers over standard Adam
optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-2) # Use model2's parameters

# Learning rate scheduler (warmup is crucial for Transformers)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

features = torch.load("load/your/merged_features.pt")  # Load the merged features dictionary
# Split features into training and validation sets
video_ids = list(features.keys())
train_video_ids, val_video_ids = train_test_split(video_ids, test_size=0.2, random_state=42)

train_features_dict = {vid_id: features[vid_id] for vid_id in train_video_ids}
val_features_dict = {vid_id: features[vid_id] for vid_id in val_video_ids}

# Split features into training and validation sets
video_ids = list(features.keys())
train_video_ids = video_ids

train_features_dict = {vid_id: features[vid_id] for vid_id in train_video_ids}
train_video_dataset = VideoFeatureDataset(train_features_dict)
train_dataloader_ft = DataLoader(train_video_dataset, batch_size=1 ,shuffle=True, collate_fn=custom_collate_fn)

# Run this once on your training features
import torch
all_feats = torch.cat([v for v in features.values()], dim=0) # Shape: (N, 768)

mean = all_feats.mean(dim=0)
std = all_feats.std(dim=0)

torch.save({'mean': mean, 'std': std}, "dataset_stats.pt")

# Create instances dataset
train_video_dataset = VideoFeatureDataset(train_features_dict)
val_video_dataset = VideoFeatureDataset(val_features_dict)

# 1. First, count how many Real and Fake videos you actually have
num_real = sum(1 for label in train_video_dataset._labels.values() if label == 0)
num_fake = sum(1 for label in train_video_dataset._labels.values() if label == 1)
total_videos = num_real + num_fake

print(f"Dataset Distribution - REAL: {num_real}, FAKE: {num_fake}")

# 2. Calculate the weights (Inverse Frequency)
# The rarer class (Real) gets a higher weight.
weight_for_real = total_videos / (2.0 * num_real)
weight_for_fake = total_videos / (2.0 * num_fake)

# 3. Create the weight tensor and send it to your GPU/CPU
class_weights = torch.tensor([weight_for_real, weight_for_fake], dtype=torch.float).to(device)

print(f"Class Weights - Real(0): {weight_for_real:.2f}, Fake(1): {weight_for_fake:.2f}")

# 4. Pass the weights to the CrossEntropyLoss
criterion = nn.CrossEntropyLoss(weight=class_weights)

# ---------------------------------------------------------

# Create DataLoaders, applying the custom collate_fn
batch_size = 40
train_dataloader = DataLoader(train_video_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
val_dataloader = DataLoader(val_video_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

# Train the model
print("Starting Training...")
train_transformer(model, optimizer, criterion, scheduler, train_dataloader, epochs=30)

# Evaluate the model on the validation set
print("\nStarting Validation...")
evaluate_model(model, val_dataloader)
