import torch
from torch.optim import optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Custom Dataset for your features
class VideoFeatureDataset(Dataset):
    def __init__(self, features_dict, num_classes=2):
        self.video_ids = list(features_dict.keys())
        self.features = features_dict
        self._labels = {}

        for vid_id in self.video_ids:
            path_lower = vid_id.lower()
            # CELEB-DF LOGIC
            if "celeb-synthesis" in path_lower:
                self._labels[vid_id] = 1 # Fake
            elif "celeb-real" in path_lower or "youtube-real" in path_lower:
                self._labels[vid_id] = 0 # Real

            # FACEFORENSICS++ LOGIC
            elif "original" in path_lower or "youtube" in path_lower or "real" in path_lower:
                self._labels[vid_id] = 0 # Real
            elif "manipulated" in path_lower or "fake" in path_lower or "swap" in path_lower or "face2face" in path_lower or "neuraltextures" in path_lower:
                self._labels[vid_id] = 1 # Fake
            else:
                # Fallback just in case, but print a warning so you know
                print(f"Warning: Could not determine label for {vid_id}. Defaulting to 1.")
                self._labels[vid_id] = 1


    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        feature_tensor = self.features[video_id]
        label = self._labels[video_id]
        return feature_tensor, torch.tensor(label, dtype=torch.long)

# Custom collate_fn for DataLoader to handle varying sequence lengths
def custom_collate_fn(batch):
    # batch is a list of (feature_tensor, label) tuples
    features_list = [item[0] for item in batch]
    labels_list = [item[1] for item in batch]

    # Pad the feature sequences to the maximum length in the current batch
    # pad_sequence expects a list of Tensors, each of shape (L, *), where L is the length of a sequence
    # Here, features_list contains tensors of shape (num_frames, feature_dim)
    padded_features = pad_sequence(features_list, batch_first=True, padding_value=0)

    # Stack the labels
    labels = torch.stack(labels_list)

    return padded_features, labels

def train_transformer(model,optimizer,criterion,scheduler, dataloader, epochs=10):
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        correct_preds = 0
        total_samples = 0

        for batch_features, batch_labels in dataloader:
            # Move batch to device
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            # batch_features shape: (Batch, Max_Total_Video_Frames_in_Batch, Feature_Dim)
            B, total_frames, D = batch_features.shape

            # --- SIMULATING THE RL AGENT FOR PRE-TRAINING ---
            # Randomly select between 4 and 16 frames to train robustness
            # Ensure num_selected_frames is not greater than total_frames
            max_frames_to_select = min(total_frames, 16)
            if max_frames_to_select < 4:
                # If total_frames is less than 4, select all available frames
                num_selected_frames = total_frames
                # If total_frames is 0, this will be 0, which is handled below
            else:
                num_selected_frames = torch.randint(16,31, (1,)).item()

            # Handle cases where num_selected_frames might be 0 due to very short videos
            if num_selected_frames == 0:
                # Skip this batch if no frames can be selected (e.g., all videos were empty or too short)
                continue

            # Generate random sorted indices for the frames we are "keeping"
            # Shape: (Batch, num_selected_frames)
            frame_indices_list = []
            selected_features_list = []
            for i in range(B):
                # Ensure we don't try to select more frames than available for this specific video
                current_video_frames = batch_features[i].sum(dim=-1).nonzero(as_tuple=True)[0].shape[0] # Count non-padded frames
                actual_frames_to_select = min(num_selected_frames, current_video_frames)

                if actual_frames_to_select == 0:
                    # If no actual frames, add dummy tensors to maintain batch size for stacking
                    frame_indices_list.append(torch.zeros(num_selected_frames, dtype=torch.long, device=device))
                    selected_features_list.append(torch.zeros(num_selected_frames, D, device=device))
                    continue

                perm = torch.randperm(current_video_frames, device=device)[:actual_frames_to_select]
                indices = torch.sort(perm)[0]

                # Pad indices to num_selected_frames if actual_frames_to_select is smaller
                if indices.shape[0] < num_selected_frames:
                    padding_indices = torch.full((num_selected_frames - indices.shape[0],), 0, dtype=torch.long, device=device) # Pad with 0s
                    indices = torch.cat((indices, padding_indices))

                frame_indices_list.append(indices.to(device))
                selected_features_list.append(batch_features[i, indices].to(device))

            frame_indices = torch.stack(frame_indices_list)
            selected_features = torch.stack(selected_features_list)

            # --- FORWARD PASS ---
            optimizer.zero_grad()

            # Pass only selected features for TemporalTransformerEncoder
            logits = model(selected_features, frame_indices = frame_indices) # TemporalTransformerEncoder does NOT use frame_indices

            # Calculate Loss
            loss = criterion(logits, batch_labels)

            # --- BACKWARD PASS ---
            loss.backward()

            # Gradient clipping (prevents exploding gradients in Transformers)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Metrics tracking
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct_preds += (predictions == batch_labels).sum().item()
            total_samples += B

        scheduler.step()

        epoch_acc = (correct_preds / total_samples) * 100
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f} | Accuracy: {epoch_acc:.2f}%")

def evaluate_model(model, dataloader):
    model.eval() # Set model to evaluation mode
    total_loss = 0
    correct_preds = 0
    total_samples = 0

    with torch.no_grad(): # Disable gradient calculation for evaluation
        for batch_features, batch_labels in dataloader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            logits = model(batch_features) # Pass all features for evaluation

            loss = criterion(logits, batch_labels)
            total_loss += loss.item()

            predictions = torch.argmax(logits, dim=1)
            correct_preds += (predictions == batch_labels).sum().item()
            total_samples += batch_labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = (correct_preds / total_samples) * 100 if total_samples > 0 else 0
    print(f"\nValidation | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")



