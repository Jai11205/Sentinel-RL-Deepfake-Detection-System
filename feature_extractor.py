from timm import create_model
from facenet_pytorch import MTCNN
import torch
from torchvision import transforms
import cv2
from PIL import Image
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

swin_tiny_model = create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes = 0)
swin_tiny_model.to(device)
swin_tiny_model.eval()

mtcnn = MTCNN(keep_all = False, device=device, margin = 30)

def get_video_features_mtcnn(video_path, max_frames=32):
    """
    Reads a video, detects faces, crops them, and extracts Swin features.
    """
    cap = cv2.VideoCapture(video_path)
    cropped_faces = []

    # We keep reading until we get `max_frames` valid faces OR the video ends
    while len(cropped_faces) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)

        # Detect face
        boxes, _ = mtcnn.detect(pil_frame)

        if boxes is not None:
            box = boxes[0] # Take the primary face
            x1, y1, x2, y2 = [int(b) for b in box]

            # Ensure coordinates are within image bounds
            x1 = max(0, x1 - 30)
            y1 = max(0, y1 - 30)
            x2 = min(pil_frame.width, x2 + 30)
            y2 = min(pil_frame.height, y2 + 30)

            # Crop and append
            face_img = pil_frame.crop((x1, y1, x2, y2))
            cropped_faces.append(face_img)

    cap.release()

    # If no faces were found at all, return None
    if len(cropped_faces) == 0:
        return None

    # Process the cropped faces through Swin

    batch = torch.stack([transform(f) for f in cropped_faces]).to(device)


    with torch.no_grad():
        features = swin_tiny_model(batch).cpu() # Shape: (Num_Valid_Faces, 768)

    return features

# ==========================================
# MAIN EXTRACTION LOOP
# ==========================================
if __name__ == "__main__":
    # Update this to your actual FaceForensics++ root directory!
    dataset_path = "path/to/your/dataset"

    all_features = {}

    # Walk through the dataset
    print(f"Scanning directory: {dataset_path}")

    # Get all video files first for a cleaner tqdm progress bar
    video_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith((".mp4", ".avi", ".mov")):
                video_files.append(os.path.join(root, file))

    print(f"Found {len(video_files)} videos. Starting extraction...")

    for video_path in tqdm(video_files):
        feats = get_video_features_mtcnn(video_path)

        if feats is not None:
            # Use relative path as the unique dictionary key
            unique_key = os.path.relpath(video_path, dataset_path)
            all_features[unique_key] = feats

    # Save the new robust features!
    output_filename = "ffplus_swin_features_mtcnn.pt"
    torch.save(all_features, output_filename)
    print(f"\nExtraction complete! Saved to {output_filename}")

    