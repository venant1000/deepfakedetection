#!/usr/bin/env python3

# Simple test to see what's failing
import sys
print("Python script started")

try:
    import torch
    print("PyTorch imported successfully")
    
    # Test loading the model
    model_path = "../attached_assets/lightweight_deepfake_detector.pth"
    print(f"Attempting to load model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    print("Model loaded successfully!")
    print("Type:", type(checkpoint))
    
    if isinstance(checkpoint, dict):
        print("Keys:", list(checkpoint.keys()))
    
    print("Test completed successfully")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()