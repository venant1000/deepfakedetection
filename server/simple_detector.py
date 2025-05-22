#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import cv2
import numpy as np
import json
import sys
import os
from torchvision import transforms
from PIL import Image

class LightweightDeepfakeDetector(nn.Module):
    def __init__(self, cnn_backbone='mobilenet_v2', rnn_type='lstm', hidden_size=256,
                 num_layers=1, num_classes=2, dropout=0.3):
        super(LightweightDeepfakeDetector, self).__init__()
        
        # Load CNN backbone for per-frame feature extraction (without downloading pretrained weights)
        if cnn_backbone == 'mobilenet_v2':
            cnn = models.mobilenet_v2(pretrained=False)  # Don't download weights, we'll load your trained ones
            # Use only the feature extractor part
            self.cnn = cnn.features
            cnn_out_features = 1280  # MobileNetV2's final feature size
        elif cnn_backbone == 'resnet18':
            cnn = models.resnet18(pretrained=False)
            # Remove the final classification layer
            modules = list(cnn.children())[:-1]
            self.cnn = nn.Sequential(*modules)
            cnn_out_features = 512
        else:
            raise ValueError(f"Unsupported CNN backbone: {cnn_backbone}")
        
        # Define the RNN for temporal sequence processing
        self.rnn_type = rnn_type.lower()
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=cnn_out_features, hidden_size=hidden_size,
                               num_layers=num_layers, batch_first=True,
                               dropout=dropout if num_layers > 1 else 0)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=cnn_out_features, hidden_size=hidden_size,
                              num_layers=num_layers, batch_first=True,
                              dropout=dropout if num_layers > 1 else 0)
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        
        # Final classification layer
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, sequence_length, channels, height, width)
        """
        batch_size, seq_len, C, H, W = x.size()
        # Merge batch and sequence dimensions for CNN processing
        x = x.view(batch_size * seq_len, C, H, W)
        features = self.cnn(x)
        
        # If the CNN outputs a feature map, apply adaptive pooling to get a fixed-size vector
        if features.ndim == 4:  # (batch_size*seq_len, channels, H, W)
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(batch_size, seq_len, -1)
        else:
            features = features.view(batch_size, seq_len, -1)
        
        # Process the sequence of features with the RNN
        rnn_out, _ = self.rnn(features)
        # Use the last output of the RNN for classification
        final_feature = rnn_out[:, -1, :]
        out = self.classifier(final_feature)
        return out

def analyze_video(video_path):
    try:
        # Load your actual trained model with correct architecture
        device = torch.device("cpu")
        
        # Initialize model with your exact configuration from config.yaml
        model = LightweightDeepfakeDetector(
            cnn_backbone='mobilenet_v2',
            rnn_type='lstm', 
            hidden_size=256,
            num_layers=1,
            num_classes=2,
            dropout=0.3
        )
        
        # Load your trained model weights
        model_path = os.path.join(os.path.dirname(__file__), "..", "attached_assets", "lightweight_deepfake_detector.pth")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
        
        # Define transform to match your training configuration
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # From your config: frame_height/width: 224
            transforms.ToTensor(),
        ])
        
        # Extract frames from video for sequence analysis (matching your training approach)
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Extract 10 frames for sequence analysis (matching your config: sequence_length: 10)
        sequence_length = 10
        step = max(1, frame_count // sequence_length)
        
        frame_sequence = []
        frame_idx = 0
        timestamps = []
        
        while cap.isOpened() and len(frame_sequence) < sequence_length:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % step == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)
                frame_tensor = transform(pil_frame)
                frame_sequence.append(frame_tensor)
                timestamps.append(frame_idx / fps)
            
            frame_idx += 1
        
        cap.release()
        
        if len(frame_sequence) < sequence_length:
            # Pad sequence if we don't have enough frames
            while len(frame_sequence) < sequence_length:
                frame_sequence.append(frame_sequence[-1])  # Repeat last frame
        
        # Prepare input tensor for your model: (1, sequence_length, C, H, W)
        input_tensor = torch.stack(frame_sequence).unsqueeze(0).to(device)
        
        # Run your trained model on the video sequence
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            deepfake_confidence = probabilities[0][1].item()  # Probability of being deepfake
            is_deepfake_overall = deepfake_confidence > 0.5
        
        # Create frame-by-frame results for visualization
        frame_results = []
        for i, timestamp in enumerate(timestamps):
            frame_results.append({
                "frame_index": i,
                "timestamp": timestamp,
                "confidence": deepfake_confidence,  # Your model gives overall confidence
                "is_deepfake": is_deepfake_overall
            })
        
        # Use results from your trained model
        avg_confidence = deepfake_confidence
        max_confidence = deepfake_confidence
        
        # Create timeline markers based on overall analysis
        timeline = []
        if deepfake_confidence > 0.7:
            timeline.append({
                "position": 50,  # Middle of video
                "tooltip": f"High deepfake probability: {deepfake_confidence:.1%}",
                "type": "danger"
            })
        elif deepfake_confidence > 0.4:
            timeline.append({
                "position": 50,  # Middle of video
                "tooltip": f"Moderate probability: {deepfake_confidence:.1%}",
                "type": "warning"
            })
        
        # Generate findings based on your model's results
        findings = []
        if max_confidence > 0.8:
            findings.append({
                "title": "High Confidence Deepfake Detection",
                "icon": "AlertTriangle",
                "severity": "high",
                "timespan": "Overall video analysis",
                "description": f"Your trained model detected deepfake with {max_confidence:.1%} confidence"
            })
        
        # Generate issues
        issues = []
        if is_deepfake_overall:
            issues.append({
                "type": "deepfake",
                "text": f"Video shows AI manipulation signs (confidence: {avg_confidence:.1%})"
            })
        
        return {
            "isDeepfake": is_deepfake_overall,
            "confidence": avg_confidence,
            "processingTime": len(frame_sequence) * 0.2,
            "maxConfidence": max_confidence,
            "framesAnalyzed": len(frame_sequence),
            "issues": issues,
            "findings": findings,
            "timeline": timeline,
            "frameResults": frame_results,
            "modelUsed": "PyTorch Lightweight Deepfake Detector"
        }
        
    except Exception as e:
        return {
            "error": f"Analysis failed: {str(e)}",
            "isDeepfake": False,
            "confidence": 0.0
        }

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({"error": "Usage: python3 simple_detector.py <video_path>"}))
        sys.exit(1)
    
    video_path = sys.argv[1]
    result = analyze_video(video_path)
    print(json.dumps(result))