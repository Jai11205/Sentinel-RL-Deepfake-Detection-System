from timm import create_model
from facenet_pytorch import MTCNN
import os
import torch
import torch.nn as nn
from torchvision import transforms
import cv2
from PIL import Image
from tqdm import tqdm
from stable_baselines3 import PPO
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

swin_tiny_model = create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes = 0)
swin_tiny_model.eval()
swin_tiny_model.to(device)

mtcnn = MTCNN(keep_all = False, device=device, margin = 30)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

class TemporalTransformerEncoder2(nn.Module):
    def __init__(self,
                 feature_dim=768,      # Swin output dim
                 num_frames=32,
                 num_heads=8,
                 num_layers=4,         # usually 2-6 is enough
                 mlp_ratio=4,
                 num_classes=2,
                 dropout=0.1):
        super().__init__()

        self.num_frames = num_frames
        self.feature_dim = feature_dim

        # Learnable temporal positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_frames, feature_dim))

        # Optional CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, feature_dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * mlp_ratio,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(feature_dim)
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x, frame_indices = None):
        # x shape: (B, T, D)   where T = num_frames, D = feature_dim
        B, T, D = x.shape

        if frame_indices is not None:
            # frame_indices shape must be (B, T)
            # We need to gather the correct positional embeddings for each item in the batch.

            # Expand pos_embed to match batch size: (B, max_frames, D)
            expanded_pos_embed = self.pos_embed.expand(B, -1, -1)

            # Expand indices to match feature dimension: (B, T, D)
            # This allows us to use torch.gather to pluck out the correct 768-dim vectors
            gather_indices = frame_indices.unsqueeze(-1).expand(-1, -1, D)

            # Gather the selected embeddings
            selected_pos_embed = torch.gather(expanded_pos_embed, 1, gather_indices)

            x = x + selected_pos_embed
        else:
            x = x + self.pos_embed[:, :T]

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)   # (B, T+1, D)

        # Pass through Transformer
        x = self.transformer(x)
        x = self.norm(x)

        # Use CLS token for classification
        cls_output = x[:, 0]
        logits = self.head(cls_output)

        return logits

model = TemporalTransformerEncoder2(
    feature_dim=768,
    num_frames=32,
    num_heads=8,
    num_layers=4,
    num_classes=2
).to(device)

# Load the state dictionary into the instantiated model
model.load_state_dict(torch.load("/content/drive/MyDrive/Sentinel-RL: AI-Powered Deepfake Detection/trained_transformer_model_2ds_02.pt", map_location=device))

# Set the model to evaluation mode (important for inference, especially with Dropout/BatchNorm)
model.eval()

print("Model loaded successfully and set to evaluation mode.")

ppo_agent = PPO.load("/content/drive/MyDrive/Sentinel-RL: AI-Powered Deepfake Detection/ppo_deepfake_frame_selector_2de_02.zip", device=device)

def predict_single_video(video_path, swin_model, rl_agent, transformer_model,mtcnn, device, max_frames=32):

    print(f"\nProcessing Video: {video_path}")

    # Initialize the Face Detector
    # margin=30 ensures to capture the whole head (chin, hair) not just the eyes/nose

    cap = cv2.VideoCapture(video_path)
    cropped_faces = []

    while len(cropped_faces) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)

        # Detect and crop the face
        face = mtcnn(pil_frame)

        # If a face is found, MTCNN returns a tensor scaled [-1, 1].
        # Need to convert it back to a standard PIL image for the existing transforms,
        # or just apply  normalization directly
        # To keep it compatible with training pipeline, get the bounding box instead
        # and crop it manually using PIL.

        boxes, _ = mtcnn.detect(pil_frame)
        if boxes is not None:
            box = boxes[0] # Take the primary face
            # Expand box slightly based on margin
            x1, y1, x2, y2 = [int(b) for b in box]

            # Ensure coordinates are within image bounds
            x1 = max(0, x1 - 30)
            y1 = max(0, y1 - 30)
            x2 = min(pil_frame.width, x2 + 30)
            y2 = min(pil_frame.height, y2 + 30)

            face_img = pil_frame.crop((x1, y1, x2, y2))
            cropped_faces.append(face_img)

    cap.release()

    if len(cropped_faces) == 0:
        return "Error: Could not detect any faces in the video frames."

    print(f"Successfully extracted and cropped {len(cropped_faces)} faces.")

    #Apply transforms to the cropped faces, not the whole TV frame
    batch = torch.stack([transform(f) for f in cropped_faces]).to(device)

    # Extract features using Swin
    swin_model.eval()
    with torch.no_grad():
        video_features = swin_model(batch).cpu() # Shape: (T, 768)

        # Inside predict_video.py, after getting video_features:
    stats = torch.load("dataset_stats.pt")
    mean = stats['mean'].to(device).cpu() # Move to CPU
    std = stats['std'].to(device).cpu()   # Move to CPU

    # Standardize: (X - mean) / std
    video_features = (video_features - mean) / (std + 1e-6)

    # -----------------------------------------
    # STEP 2: RL Agent Frame Selection
    # -----------------------------------------
    selected_features = []
    selected_indices = []

    for i in range(len(video_features)):
        obs = video_features[i].numpy()

        action, _ = rl_agent.predict(obs, deterministic=True)

        if action == 1:
            selected_features.append(video_features[i])
            selected_indices.append(i)

    # IMPROVED FALLBACK: If the RL agent skips everything, uniformly sample frames
    # This prevents the model from just looking at the first 4 (often static) frames.
    if len(selected_features) == 0:
        print("Agent attempted to skip all frames. Applying uniform sampling fallback.")
        num_frames = len(video_features)
        fallback_len = min(6, num_frames)
        # Generate evenly spaced indices (e.g., [0, 6, 12, 18, 24, 30])
        selected_indices = np.linspace(0, num_frames - 1, fallback_len, dtype=int).tolist()
        selected_features = [video_features[i] for i in selected_indices]

    # -----------------------------------------
    # STEP 3: Final Classification
    # -----------------------------------------
    x_input = torch.stack(selected_features).unsqueeze(0).to(device)
    indices_input = torch.tensor(selected_indices).unsqueeze(0).to(device)

    transformer_model.eval()
    with torch.no_grad():
        logits = transformer_model(x_input, frame_indices=indices_input)

        probabilities = torch.softmax(logits, dim=1)[0]

        real_prob = probabilities[0].item() * 100
        fake_prob = probabilities[1].item() * 100

        prediction = torch.argmax(logits, dim=1).item()

    # -----------------------------------------
    # STEP 4: Output Results
    # -----------------------------------------
    # In predict_video.py, replace your prediction logic with this:
    probabilities = torch.softmax(logits, dim=1)[0]
    real_prob = probabilities[0].item()
    fake_prob = probabilities[1].item()

    if abs(real_prob - fake_prob) < 0.20: # If confidence difference is < 20%
        result_label = "UNCERTAIN / INCONCLUSIVE"
        print("The video quality is too different from my training data to determine.")
    else:
        result_label = "FAKE" if fake_prob > real_prob else "REAL"
    #result_label = "FAKE (Deepfake)" if prediction == 1 else "REAL"
    confidence = fake_prob if prediction == 1 else real_prob

    print("====================================")
    print(f"Prediction:    {result_label}")
    print(f"Confidence:    {confidence:.2f}%")
    print(f"Frames Used:   {len(selected_indices)} out of {len(video_features)} original frames")
    print("====================================")

    return prediction, confidence, len(selected_indices)

# ==========================================
# HOW TO RUN IT
# ==========================================
if __name__ == "__main__":
    # Ensure all your models are loaded and sent to the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vid_path = input()
   
    test_video_path = vid_path
    # Load the trained PPO agent
    # Make sure 'ppo_deepfake_frame_selector' is the correct path where you saved it.


    # Run the inference
    predict_single_video(
        video_path=test_video_path,
         swin_model=swin_tiny_model,
         mtcnn = mtcnn,
         rl_agent=ppo_agent, # Pass the loaded PPO agent object
         transformer_model=model,
         device=device
      )
