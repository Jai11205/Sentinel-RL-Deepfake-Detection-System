#Sentinel-RL: AI-Powered Deepfake Detection
**Sentinel-RL is a high-performance deepfake detection system designed for real-time video moderation. Unlike standard "black-box" models that process every frame, Sentinel-RL leverages Reinforcement Learning (PPO) to intelligently select only the most critical frames, reducing computational overhead by over 60% without sacrificing detection accuracy.
**Key Features
RL-Driven Efficiency: An intelligent agent that dynamically chooses frames, optimizing GPU usage for high-throughput moderation environments.
Robust Architecture: Employs a Swin Transformer for spatial feature extraction paired with a Temporal Transformer to analyze manipulation artifacts.
Adaptive Inference: Scales compute usage based on video complexity—processing fewer frames for obvious fakes and more for sophisticated deepfakes.
Production-Ready: Calibrated for real-world deployment, handling both internet-sourced deepfakes and high-definition smartphone recordings.
**Architecture
The system operates in three phases:
Spatial Feature Extraction: MTCNN detects faces, and Swin Transformer (Tiny) encodes spatial features into a 768-dimensional vector.
RL-Based Frame Selection: A PPO agent acts as a filter, deciding which frames warrant deep analysis, significantly lowering the inference cost.
Temporal Classification: A Multi-Head Temporal Transformer analyzes the selected sequence to detect temporal inconsistencies.
**Performance Metrics
Detection Accuracy: ~87%+
Computational Efficiency: ~60-70% frame reduction.
Inference Latency: Optimized for near real-time performance.
**Getting Started
Prerequisites
Python 3.10+
PyTorch & Torchvision
Stable-Baselines3 (for the RL agent)
facenet-pytorch
**Installation
git clone https://github.com/yourusername/sentinel-rl.git
pip install -r requirements.txt


**Inference
To run detection on a new video:
python predict_video.py --video_path path/to/your/video.mp4


**Contributing
This project is part of my final year engineering work. Contributions, suggestions, and feedback on architectural optimizations are welcome!
License
MIT
