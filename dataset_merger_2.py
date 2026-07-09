import torch
import os

#Load the features of the two datasets
features1 = torch.load(
    "path/to/your/features1.pt"
)
features2 = torch.load(
    "path/to/your/features2.pt"
)

# if wanted add new arduments to the merge_datasets function to specify the paths of the two datasets and the output path.
import torch
import os

def merge_datasets(ff_plus_path, custom_reals_path, output_path):
    """
    Merges the massive FaceForensics++ feature dictionary with your new
    custom real videos to anchor the domain.
    """
    print("Loading FaceForensics++ Features...")
    ff_features = ff_plus_path

    print("Loading Custom Real Videos...")
    custom_features = custom_reals_path

    # Check how many we have
    print(f"Initial FF++ Videos: {len(ff_features)}")
    print(f"New Custom Real Videos: {len(custom_features)}")

    # Merge them into a new dictionary
    merged_features = ff_features.copy()

    # Add a unique prefix so the dictionary keys don't collide
    # We will assume all custom videos are REAL (label 0)
    for key, tensor_data in custom_features.items():
        new_key = f"celebdf{key}"
        merged_features[new_key] = tensor_data

    print(f"Total Merged Videos: {len(merged_features)}")

    # Save the new master dataset
    torch.save(merged_features, output_path)
    print(f"Successfully saved master dataset to: {output_path}")

# =========================================================
# HOW TO UPDATE YOUR LABEL ASSIGNMENT (in your training script)
# =========================================================
"""
When you initialize your VideoFeatureDataset, you need to update
the labeling logic to ensure the new custom videos are labeled as REAL (0).

Add this to your __init__ label assignment loop:

if "custom_real" in path_lower:
    self._labels[vid_id] = 0  # Anchor these as REAL
elif "original" in path_lower or "youtube" in path_lower:
    self._labels[vid_id] = 0
elif "manipulated" in path_lower or "fake" in path_lower or "swap" in path_lower:
    self._labels[vid_id] = 1
"""

if __name__ == "__main__":
    # Example Usage:
    merge_datasets(
         ff_plus_path=features1,
         custom_reals_path=features2,
         output_path="master_training_features.pt"
     )
    pass
