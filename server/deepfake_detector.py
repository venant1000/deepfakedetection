import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import os
import json
import sys
import time
from torchvision import transforms
from scipy.ndimage import gaussian_filter

class LightweightDeepfakeDetector(nn.Module):
    """
    Lightweight CNN model for deepfake detection
    This is a common architecture for video frame classification
    """
    def __init__(self, num_classes=2):
        super(LightweightDeepfakeDetector, self).__init__()
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth conv block
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

class DeepfakeAnalyzer:
    def __init__(self, model_path="attached_assets/lightweight_deepfake_detector.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LightweightDeepfakeDetector(num_classes=2)
        
        # Load the model weights
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.eval()
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
        
        # Define preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_frames(self, video_path, max_frames=30):
        """Extract frames from video for analysis"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Sample frames evenly throughout the video
        step = max(1, frame_count // max_frames)
        
        frame_idx = 0
        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % step == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append((frame_rgb, frame_idx / fps))  # Store frame and timestamp
            
            frame_idx += 1
        
        cap.release()
        return frames, fps, frame_count
    
    def preprocess_frame(self, frame):
        """Apply preprocessing to improve accuracy for real videos"""
        # Check if the frame has sufficient quality for reliable analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Apply noise reduction
        denoised = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
        
        # Adjust color contrast to normalize lighting variations
        lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        
        # Check if the image quality is too low for reliable analysis
        if blur_score < 50:  # Very blurry images often cause false positives
            quality_factor = 0.7  # Reduce confidence for low-quality frames
        else:
            quality_factor = 1.0
            
        return enhanced, quality_factor
    
    def analyze_frame(self, frame):
        """Analyze a single frame for deepfake detection with preprocessing"""
        # Apply preprocessing
        enhanced_frame, quality_factor = self.preprocess_frame(frame)
        
        # Convert to PIL image for the model
        pil_frame = Image.fromarray(enhanced_frame)
        input_tensor = self.transform(pil_frame).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            raw_confidence = probabilities[0][1].item()  # Probability of being deepfake
            
            # Apply quality adjustment
            adjusted_confidence = raw_confidence * quality_factor
            
            # Apply additional threshold adjustment for more realistic results
            # This makes the model more conservative in its deepfake predictions
            confidence_threshold = 0.65  # Increased from 0.5 to reduce false positives
            is_deepfake = adjusted_confidence > confidence_threshold
        
        return is_deepfake, adjusted_confidence
    
    def analyze_video(self, video_path):
        """Analyze entire video and return detailed results"""
        try:
            frames, fps, total_frames = self.extract_frames(video_path)
            
            if not frames:
                return {
                    "error": "Could not extract frames from video",
                    "isDeepfake": False,
                    "confidence": 0.0
                }
            
            frame_results = []
            deepfake_scores = []
            timeline_markers = []
            suspicious_segments = []
            
            for i, (frame, timestamp) in enumerate(frames):
                is_deepfake, confidence = self.analyze_frame(frame)
                deepfake_scores.append(confidence)
                frame_results.append({
                    "frame_index": i,
                    "timestamp": timestamp,
                    "is_deepfake": is_deepfake,
                    "confidence": confidence
                })
                
                # Create timeline markers for significant detections
                if confidence > 0.7:
                    timeline_markers.append({
                        "position": (timestamp / (total_frames / fps)) * 100,
                        "tooltip": f"High deepfake probability: {confidence:.1%}",
                        "type": "danger"
                    })
                elif confidence > 0.4:
                    timeline_markers.append({
                        "position": (timestamp / (total_frames / fps)) * 100,
                        "tooltip": f"Moderate deepfake probability: {confidence:.1%}",
                        "type": "warning"
                    })
            
            # Calculate overall statistics
            avg_confidence = np.mean(deepfake_scores)
            max_confidence = np.max(deepfake_scores)
            deepfake_frame_count = sum(1 for score in deepfake_scores if score > 0.5)
            
            # Determine if video is likely deepfake
            is_deepfake_overall = avg_confidence > 0.5 or max_confidence > 0.8
            
            # Generate findings based on analysis
            findings = []
            if max_confidence > 0.8:
                findings.append({
                    "title": "High Confidence Deepfake Detection",
                    "icon": "AlertTriangle",
                    "severity": "high",
                    "timespan": f"Peak at {frame_results[np.argmax(deepfake_scores)]['timestamp']:.1f}s",
                    "description": f"Model detected deepfake indicators with {max_confidence:.1%} confidence"
                })
            
            if deepfake_frame_count > len(frames) * 0.3:
                findings.append({
                    "title": "Widespread Manipulation",
                    "icon": "Eye",
                    "severity": "medium",
                    "timespan": f"{deepfake_frame_count}/{len(frames)} frames",
                    "description": "Significant portion of video shows deepfake characteristics"
                })
            
            # Generate issues list
            issues = []
            if is_deepfake_overall:
                issues.append({
                    "type": "deepfake",
                    "text": f"Video shows signs of AI manipulation (confidence: {avg_confidence:.1%})"
                })
            
            if max_confidence > 0.9:
                issues.append({
                    "type": "high_confidence",
                    "text": f"Very high deepfake probability detected in some frames"
                })
            
            return {
                "isDeepfake": is_deepfake_overall,
                "confidence": avg_confidence,
                "processingTime": len(frames) * 0.1,  # Approximate processing time
                "maxConfidence": max_confidence,
                "framesAnalyzed": len(frames),
                "deepfakeFrames": deepfake_frame_count,
                "issues": issues,
                "findings": findings,
                "timeline": timeline_markers,
                "frameResults": frame_results
            }
            
        except Exception as e:
            return {
                "error": f"Analysis failed: {str(e)}",
                "isDeepfake": False,
                "confidence": 0.0
            }

def main():
    if len(sys.argv) != 2:
        print("Usage: python deepfake_detector.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found")
        sys.exit(1)
    
    analyzer = DeepfakeAnalyzer()
    result = analyzer.analyze_video(video_path)
    
    # Output result as JSON
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()