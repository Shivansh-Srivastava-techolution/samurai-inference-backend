import torch
import torch.nn as nn

class CNN1DModel(nn.Module):
    """
    A simple 1D CNN that processes sequences of shape:
      (batch, seq_len, num_features)
    
    We'll reshape to (batch, num_features, seq_len) so that 'num_features' acts like "channels",
    and 'seq_len' is the time dimension for the 1D convolution.
    
    Architecture Overview:
      - Conv1d -> ReLU
      - Conv1d -> ReLU
      - AdaptiveMaxPool1d(1) to reduce the time dimension to 1
      - Fully-connected layer to output classification
    """
    
    def __init__(self, num_features=9, num_classes=2):
        super(CNN1DModel, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = torch.nn.AdaptiveMaxPool1d(output_size=1)
        self.fc   = torch.nn.Linear(128, num_classes)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, num_features=9)
        We permute to: (batch_size, num_features=9, seq_len) for Conv1d.
        """
        x = x.permute(0, 2, 1)  # => (batch, 9, seq_len)
        x = torch.relu(self.conv1(x))    # => (batch, 64, seq_len)
        x = torch.relu(self.conv2(x))    # => (batch, 128, seq_len)
        x = self.pool(x)                 # => (batch, 128, 1)
        x = x.squeeze(-1)                # => (batch, 128)
        out = self.fc(x)                 # => (batch, num_classes)
        return out