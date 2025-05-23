import os
import cv2
import argparse
import random

def extract_frames(video_path, output_dir, frame_skip=1, max_frames=64):
    """
    Extract frames from a video and save them as JPEG images.
    
    Args:
        video_path (str): Path to the input video.
        output_dir (str): Directory where extracted frames will be saved.
        frame_skip (int): Save every n-th frame (default is 1, i.e. all frames).
        max_frames (int): Maximum number of frames to extract per video.
    
    Returns:
        int: The number of frames saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Process only every nth frame as specified by frame_skip
        if frame_count % frame_skip == 0:
            frame_filename = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
            # Stop extracting if we've reached the maximum number of frames
            if saved_count >= max_frames:
                break
        frame_count += 1
    cap.release()
    return saved_count

def process_videos(raw_dir, train_dir, val_dir, train_split=0.8, frame_skip=1, max_frames=64):
    """
    Process videos by extracting frames (up to max_frames per video) and splitting them into training and validation sets.
    
    The expected raw directory structure is:
    
        raw_dir/
          real/
            video1.mp4
            video2.avi
          fake/
            video3.mp4
            video4.mov
    
    For each video, frames will be extracted into a folder named after the video file (without extension).
    The output structure will be:
    
        train_dir/
          real/
            video1/
              frame_0000.jpg, frame_0001.jpg, ...
          fake/
            video3/
              frame_0000.jpg, frame_0001.jpg, ...
    
    Args:
        raw_dir (str): Directory containing raw videos organized by class.
        train_dir (str): Output directory for training data.
        val_dir (str): Output directory for validation data.
        train_split (float): Proportion of videos to allocate for training.
        frame_skip (int): Extract every n-th frame.
        max_frames (int): Maximum number of frames to extract per video.
    """
    # Create destination directories for each class in train and val.
    for target_dir in [train_dir, val_dir]:
        for label in ['real', 'fake']:
            os.makedirs(os.path.join(target_dir, label), exist_ok=True)

    for label in ['real', 'fake']:
        raw_label_dir = os.path.join(raw_dir, label)
        if not os.path.isdir(raw_label_dir):
            print(f"Warning: {raw_label_dir} not found, skipping label '{label}'.")
            continue

        # List video files for this label.
        video_files = [f for f in os.listdir(raw_label_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
        video_files = [os.path.join(raw_label_dir, f) for f in video_files]
        random.shuffle(video_files)
        train_count = int(len(video_files) * train_split)
        train_files = video_files[:train_count]
        val_files = video_files[train_count:]

        print(f"Processing label '{label}': {len(video_files)} videos found. "
              f"{len(train_files)} for training, {len(val_files)} for validation.")

        # Process training videos
        for video_path in train_files:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_subdir = os.path.join(train_dir, label, video_name)
            if os.path.exists(output_subdir):
                print(f"Directory {output_subdir} already exists, skipping extraction.")
                continue
            print(f"Extracting frames from {video_path} to {output_subdir}")
            num_frames = extract_frames(video_path, output_subdir, frame_skip, max_frames)
            print(f"Extracted {num_frames} frames from {video_path}")

        # Process validation videos
        for video_path in val_files:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_subdir = os.path.join(val_dir, label, video_name)
            if os.path.exists(output_subdir):
                print(f"Directory {output_subdir} already exists, skipping extraction.")
                continue
            print(f"Extracting frames from {video_path} to {output_subdir}")
            num_frames = extract_frames(video_path, output_subdir, frame_skip, max_frames)
            print(f"Extracted {num_frames} frames from {video_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from raw videos (max N frames per video) and organize them into training and validation directories."
    )
    parser.add_argument("--raw_dir", type=str, default="data/raw",
                        help="Directory containing raw videos organized by class (e.g., 'real' and 'fake').")
    parser.add_argument("--train_dir", type=str, default="data/processed/train",
                        help="Output directory for training data (frames).")
    parser.add_argument("--val_dir", type=str, default="data/processed/val",
                        help="Output directory for validation data (frames).")
    parser.add_argument("--train_split", type=float, default=0.8,
                        help="Fraction of videos to use for training (rest used for validation).")
    parser.add_argument("--frame_skip", type=int, default=1,
                        help="Extract every n-th frame from each video (default: 1, i.e. all frames).")
    parser.add_argument("--max_frames", type=int, default=64,
                        help="Maximum number of frames to extract per video (default: 64).")
    args = parser.parse_args()

    process_videos(args.raw_dir, args.train_dir, args.val_dir, args.train_split, args.frame_skip, args.max_frames)
    print("Preprocessing complete.")

if __name__ == '__main__':
    main()
