#!/usr/bin/env python3
import torch
import torch.nn as nn
import cv2
import numpy as np
import json
import sys
import os
from torchvision import transforms
from PIL import Image

class LightweightDeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(LightweightDeepfakeDetector, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def analyze_video(video_path):
    try:
        # Load your model
        device = torch.device("cpu")
        model = LightweightDeepfakeDetector(num_classes=2)
        
        model_path = os.path.join(os.path.dirname(__file__), "..", "attached_assets", "lightweight_deepfake_detector.pth")
        checkpoint = torch.load(model_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        # Define transform
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Extract frames from video
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Sample up to 10 frames
        step = max(1, frame_count // 10)
        frame_idx = 0
        
        while cap.isOpened() and len(frames) < 10:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % step == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append((frame_rgb, frame_idx / fps))
            
            frame_idx += 1
        
        cap.release()
        
        if not frames:
            return {"error": "Could not extract frames from video"}
        
        # Analyze frames with your model
        deepfake_scores = []
        frame_results = []
        
        for i, (frame, timestamp) in enumerate(frames):
            pil_frame = Image.fromarray(frame)
            input_tensor = transform(pil_frame).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence = probabilities[0][1].item()  # Deepfake probability
                deepfake_scores.append(confidence)
                
                frame_results.append({
                    "frame_index": i,
                    "timestamp": timestamp,
                    "confidence": confidence,
                    "is_deepfake": confidence > 0.5
                })
        
        # Calculate overall results
        avg_confidence = np.mean(deepfake_scores)
        max_confidence = np.max(deepfake_scores)
        is_deepfake_overall = avg_confidence > 0.5
        
        # Create timeline markers
        timeline = []
        for result in frame_results:
            if result["confidence"] > 0.7:
                timeline.append({
                    "position": (result["timestamp"] / (frame_count / fps)) * 100,
                    "tooltip": f"High deepfake probability: {result['confidence']:.1%}",
                    "type": "danger"
                })
            elif result["confidence"] > 0.4:
                timeline.append({
                    "position": (result["timestamp"] / (frame_count / fps)) * 100,
                    "tooltip": f"Moderate probability: {result['confidence']:.1%}",
                    "type": "warning"
                })
        
        # Generate findings
        findings = []
        if max_confidence > 0.8:
            findings.append({
                "title": "High Confidence Deepfake Detection",
                "icon": "AlertTriangle",
                "severity": "high",
                "timespan": f"Peak at {frame_results[np.argmax(deepfake_scores)]['timestamp']:.1f}s",
                "description": f"Model detected deepfake with {max_confidence:.1%} confidence"
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
            "processingTime": len(frames) * 0.2,
            "maxConfidence": max_confidence,
            "framesAnalyzed": len(frames),
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