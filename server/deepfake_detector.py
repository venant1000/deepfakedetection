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
        """Extract frames from video for analysis with smart selection"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps > 0 else 0
        
        if frame_count <= 0:
            print(f"Warning: Could not determine frame count for {video_path}")
            frame_count = 1000  # Assume a reasonable number
            
        # First pass: analyze video structure to identify candidate frames
        # For short videos (<10s), we'll do denser sampling
        if duration < 10 and frame_count > max_frames:
            # For short videos, we want more granular analysis
            step = max(1, frame_count // (max_frames * 2))
        else:
            # Standard sampling rate
            step = max(1, frame_count // max_frames)
            
        # Store some candidate frames with their "interestingness" score
        candidate_frames = []
        last_gray = None
        frame_idx = 0
        scene_changes = []
        
        # First pass to detect scene changes and interesting frames
        while cap.isOpened() and frame_idx < frame_count:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process every nth frame to save computation
            if frame_idx % (step // 2) == 0:
                # Convert to grayscale for analysis
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Score this frame for "interestingness"
                interest_score = 0
                
                # 1. Check for scene changes
                if last_gray is not None:
                    # Calculate frame difference to detect scene changes
                    diff = cv2.absdiff(gray, last_gray)
                    score = np.mean(diff)
                    if score > 20:  # Threshold for scene change
                        scene_changes.append(frame_idx)
                        interest_score += 100  # High priority for scene changes
                
                # 2. Check for faces (deepfakes often target faces)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                if len(faces) > 0:
                    interest_score += 50 * len(faces)  # Prioritize frames with faces
                
                # 3. Check for image quality/complexity
                blur = cv2.Laplacian(gray, cv2.CV_64F).var()
                if blur > 100:  # Clear images
                    interest_score += 30
                
                # Store this frame as a candidate with its score
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                candidate_frames.append((frame_rgb, frame_idx / fps, interest_score))
                
                last_gray = gray
            
            frame_idx += 1
        
        # Reset the video capture for the second pass
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # If we didn't get any candidates with the first approach, fall back to regular sampling
        if not candidate_frames:
            frame_idx = 0
            while cap.isOpened() and len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % step == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append((frame_rgb, frame_idx / fps))
                
                frame_idx += 1
        else:
            # Sort candidates by interest score (highest first)
            candidate_frames.sort(key=lambda x: x[2], reverse=True)
            
            # Take the top frames up to max_frames
            selected_frames = candidate_frames[:max_frames]
            
            # Sort them by timestamp to maintain chronological order
            selected_frames.sort(key=lambda x: x[1])
            
            # Extract just the frame and timestamp
            frames = [(frame, timestamp) for frame, timestamp, _ in selected_frames]
            
            # If we have too few frames from interesting points, add some regular samples
            if len(frames) < max_frames * 0.7:
                remaining = max_frames - len(frames)
                regular_step = max(1, frame_count // remaining)
                
                frame_idx = 0
                existing_timestamps = set(timestamp for _, timestamp in frames)
                
                while cap.isOpened() and len(frames) < max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    timestamp = frame_idx / fps
                    if frame_idx % regular_step == 0 and timestamp not in existing_timestamps:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append((frame_rgb, timestamp))
                    
                    frame_idx += 1
                
                # Sort frames by timestamp again
                frames.sort(key=lambda x: x[1])
        
        cap.release()
        
        # Log summary of frame extraction
        print(f"Extracted {len(frames)} frames from {video_path}")
        print(f"Video duration: {duration:.2f}s, Frame count: {frame_count}, FPS: {fps:.2f}")
        if scene_changes:
            print(f"Detected {len(scene_changes)} scene changes")
        
        return frames, fps, frame_count
    
    def preprocess_frame(self, frame):
        """Apply enhanced preprocessing to improve accuracy for deepfake detection"""
        # Check if the frame has sufficient quality for reliable analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Step 1: Apply noise reduction with a more adaptive approach
        # Use different parameters based on the amount of noise detected
        if blur_score < 100:
            # More aggressive denoising for noisy images
            denoised = cv2.fastNlMeansDenoisingColored(frame, None, 15, 15, 7, 21)
        else:
            # Lighter denoising for cleaner images
            denoised = cv2.fastNlMeansDenoisingColored(frame, None, 7, 7, 7, 15)
        
        # Step 2: Adjust color contrast to normalize lighting variations
        lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        
        # Step 3: Edge enhancement to highlight facial features and potential artifacts
        # This helps in detecting inconsistencies in deepfakes
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Step 4: Apply bilateral filter to preserve edges while smoothing
        # This reduces noise while maintaining important edge details
        smoothed = cv2.bilateralFilter(sharpened, 9, 75, 75)
        
        # Step 5: Equalize histogram in RGB channels to improve contrast
        # Split the image into the three color channels
        r, g, b = cv2.split(smoothed)
        # Apply histogram equalization to each channel
        r_eq = cv2.equalizeHist(r)
        g_eq = cv2.equalizeHist(g)
        b_eq = cv2.equalizeHist(b)
        # Merge the equalized channels
        balanced = cv2.merge((r_eq, g_eq, b_eq))
        
        # Step 6: Apply a small amount of Gaussian blur to reduce artifacts introduced by previous steps
        final_enhanced = cv2.GaussianBlur(balanced, (3, 3), 0)
        
        # Dynamically adjust quality factor based on blur score and other image properties
        # A more granular approach to quality assessment
        if blur_score < 50:  # Very blurry images
            quality_factor = 0.6  # Further reduce confidence for very low-quality frames
        elif blur_score < 100:  # Somewhat blurry
            quality_factor = 0.8  # Slightly reduce confidence
        else:
            quality_factor = 1.0  # Normal confidence for clear frames
        
        # Additional adjustment based on brightness uniformity
        # Deepfakes sometimes have unusual lighting patterns
        brightness = np.mean(gray)
        brightness_std = np.std(gray)
        brightness_uniformity = brightness_std / brightness if brightness > 0 else 0
        
        # If brightness is too uniform or non-uniform, it might be suspicious
        if brightness_uniformity < 0.1 or brightness_uniformity > 0.5:
            quality_factor *= 0.9
            
        return final_enhanced, quality_factor
    
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