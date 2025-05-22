import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from src.models.lightweight_detector import LightweightDeepfakeDetector
from src.dataset.deepfake_dataset import DeepfakeDataset
from src.utils.data_loader import get_dataloader

def load_config(config_path='config/config.yaml'):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def train():
    # Load configuration parameters.
    config = load_config()
    model_cfg = config['model']
    training_cfg = config['training']
    data_cfg = config['data']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define image transforms (resize and conversion to tensor).
    transform = transforms.Compose([
        transforms.Resize((data_cfg['frame_height'], data_cfg['frame_width'])),
        transforms.ToTensor(),
    ])

    # Create the training dataset and DataLoader.
    train_dataset = DeepfakeDataset(
        data_dir=data_cfg['train_dir'],
        sequence_length=data_cfg['sequence_length'],
        transform=transform
    )
    train_loader = get_dataloader(
        train_dataset, 
        batch_size=training_cfg['batch_size'], 
        shuffle=True, 
        num_workers=0  # Set to 0 on Windows.
    )

    # Create the validation dataset and DataLoader.
    val_dataset = DeepfakeDataset(
        data_dir=data_cfg['val_dir'],
        sequence_length=data_cfg['sequence_length'],
        transform=transform
    )
    val_loader = get_dataloader(
        val_dataset, 
        batch_size=training_cfg['batch_size'], 
        shuffle=False, 
        num_workers=0
    )

    # Instantiate the model.
    model = LightweightDeepfakeDetector(
        cnn_backbone=model_cfg['cnn_backbone'],
        rnn_type=model_cfg['rnn_type'],
        hidden_size=model_cfg['hidden_size'],
        num_layers=model_cfg['num_layers'],
        num_classes=model_cfg['num_classes'],
        dropout=model_cfg['dropout']
    )
    model = model.to(device)

    # Set up loss function and optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=training_cfg['learning_rate'])
    num_epochs = training_cfg['num_epochs']

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (frames, labels) in enumerate(train_loader):
            frames, labels = frames.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (i + 1) % 10 == 0:
                avg_loss = running_loss / 10
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Training Loss: {avg_loss:.4f}")
                running_loss = 0.0

        # Notify that validation is starting.
        print(f"\nEpoch [{epoch+1}/{num_epochs}] finished training. Now running validation...\n")
        
        # Run evaluation on the validation set after each epoch.
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for frames, labels in val_loader:
                frames, labels = frames.to(device), labels.to(device)
                outputs = model(frames)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Calculate accuracy.
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%\n")

    # Save the trained model checkpoint.
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/lightweight_deepfake_detector.pth')
    print("Training complete. Model saved to 'models/lightweight_deepfake_detector.pth'.")

if __name__ == '__main__':
    train()
