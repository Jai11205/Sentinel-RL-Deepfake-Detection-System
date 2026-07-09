# Sentinel-RL: AI-Powered Deepfake Detection
Sentinel-RL is a high-performance deepfake detection system designed for real-time video moderation. Unlike standard "black-box" models that process every frame, Sentinel-RL leverages Reinforcement Learning (PPO) to intelligently select only the most critical frames, reducing computational overhead by over 60% without sacrificing detection accuracy.
## Key Features
RL-Driven Efficiency: An intelligent agent that dynamically chooses frames, optimizing GPU usage for high-throughput moderation environments.
Robust Architecture: Employs a Swin Transformer for spatial feature extraction paired with a Temporal Transformer to analyze manipulation artifacts.
Adaptive Inference: Scales compute usage based on video complexity—processing fewer frames for obvious fakes and more for sophisticated deepfakes.
Production-Ready: Calibrated for real-world deployment, handling both internet-sourced deepfakes and high-definition smartphone recordings.
## Architecture:

<img width="2486" height="5223" alt="Video Ingestion and-2026-07-08-154626" src="https://github.com/user-attachments/assets/c9cd6e6f-9473-41a7-90f6-ae50a5bf18b5" />


The system operates in three phases:
Spatial Feature Extraction: MTCNN detects faces, and Swin Transformer (Tiny) encodes spatial features into a 768-dimensional vector.
RL-Based Frame Selection: A PPO agent acts as a filter, deciding which frames warrant deep analysis, significantly lowering the inference cost.
Temporal Classification: A Multi-Head Temporal Transformer analyzes the selected sequence to detect temporal inconsistencies.
## Performance Metrics
Detection Accuracy: ~87%+
Computational Efficiency: ~60-70% frame reduction.
Inference Latency: Optimized for near real-time performance.
## getting Started
Prerequisites
Python 3.10+
PyTorch & Torchvision
Stable-Baselines3 (for the RL agent)
facenet-pytorch
## Installation
    git clone https://github.com/Jai11205/Sentinal-RL-Deepfake-Detectio_system.git
    
    pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 timm stable-baselines3==2.1.0 facenet-pytorch


## Inference
To run detection on a new video:
    
    python Sentinal-RL_Runner.py --video_path path/to/your/video.mp4

## Sentinel-RL Technical Handbook

1. **Pipeline Architecture**

The system is optimized for real-time video moderation, decoupled into three specialized phases to ensure high throughput and low latency.

Phase A: Spatial Feature Extraction

Input: Raw video stream (MP4/MOV).

Processing:

MTCNN: Isolates facial regions to filter out irrelevant background noise (e.g., news tickers, scene changes).

Swin Transformer (Tiny): Extracts 768-dimensional visual embeddings. This backbone is optimized for dense feature representation, capturing fine-grained texture artifacts.

Normalization: Inputs are normalized using ImageNet statistics to ensure compatibility with pre-trained weights.

Phase B: Adaptive Frame Selection (RL)

Agent: PPO (Proximal Policy Optimization) via StableBaselines3.

Policy: The agent functions as a high-speed inference gatekeeper. It evaluates frame-level features and outputs a binary decision: {0: Skip, 1: Keep}.

Reward: A custom terminal reward function balances classification accuracy (+10.0) against per-frame computational cost (-0.2), incentivizing the agent to maximize efficiency.

Phase C: Temporal Classification

Encoder: Custom Multi-Head Temporal Transformer.

Mechanism: Aggregates the sparse sequence of RL-selected frames to model temporal manipulation artifacts, specifically identifying jitter, inconsistent lighting, and lip-sync anomalies.

Positional Embeddings: Utilizes dynamic index-based embedding mapping to compensate for the temporal gaps introduced by the RL agent’s "skip" actions.

2. **Training Workflow**

Phase 1 (Supervised Pre-training): The Temporal Encoder is pre-trained on a balanced 1:1 dataset (FaceForensics++ / Celeb-DF) to ensure the model does not collapse into majority-class bias.

Phase 2 (RL Agent Training): The PPO agent is trained using the frozen Temporal Encoder as the reward brain.

Data Balancing: We utilize strict undersampling (maintaining a 1:1 Real-to-Fake ratio) to prevent the "Mode Collapse" issue common in imbalanced deepfake datasets.

3. **Deployment & Calibration**

To ensure consistent performance across diverse input sources (webcam, social media clips):

Standardization: All input frames must be processed through the exact MTCNN and Swin Transformer pre-processing pipeline used during training.

Inference Confidence: The system outputs raw logits. For enterprise deployment, we recommend an "Inconclusive" threshold: if |Real% - Fake%| < 20%, the system returns a manual review flag rather than an automated prediction.

Computational Savings: In production, the model consistently demonstrates a 60-70% reduction in total frame processing, significantly lowering cloud infrastructure expenditure.
## Contributing
This project is part of my final year engineering work. Contributions, suggestions, and feedback on architectural optimizations are welcome!

**License**
MIT
