import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class LightweightDeepfakeDetector(nn.Module):
    def __init__(self, cnn_backbone='mobilenet_v2', rnn_type='lstm', hidden_size=256,
                 num_layers=1, num_classes=2, dropout=0.3):
        super(LightweightDeepfakeDetector, self).__init__()
        
        # Load pretrained CNN backbone for per-frame feature extraction
        if cnn_backbone == 'mobilenet_v2':
            cnn = models.mobilenet_v2(pretrained=True)
            # Use only the feature extractor part
            self.cnn = cnn.features
            cnn_out_features = 1280  # MobileNetV2’s final feature size
        elif cnn_backbone == 'resnet18':
            cnn = models.resnet18(pretrained=True)
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
