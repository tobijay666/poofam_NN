import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionPooling(nn.Module):
    """
    Self-Attention Pooling layer.
    Learns to weight specific spatial regions (voxels) more heavily than others
    before aggregating them, preserving spatial context better than Average Pooling.
    """
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        # 1x1x1 Conv acts as a learnable score function for each voxel
        self.attention_conv = nn.Conv3d(input_dim, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2) # Apply softmax over flattened spatial dimensions

    def forward(self, x):
        # x shape: (Batch, Channels, Depth, Height, Width)
        batch_size, C, D, H, W = x.size()
        
        # 1. Calculate Attention Scores
        attn_logits = self.attention_conv(x) 
        
        # 2. Flatten spatial dimensions for Softmax
        attn_logits = attn_logits.view(batch_size, 1, -1)
        
        # 3. Flatten feature map
        x_flat = x.view(batch_size, C, -1)
        
        # 4. Normalize scores (Softmax)
        attn_weights = self.softmax(attn_logits)
        
        # 5. Weighted Average
        out = torch.sum(x_flat * attn_weights, dim=2)
        
        return out

class ThreeDResNet_V27(nn.Module):
    def __init__(self, input_channels=1, num_outputs=2, dropout_prob=0.5):
        super(ThreeDResNet_V27, self).__init__()
        
        # --- Stem ---
        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)

        # --- Residual Backbone (Wider Variant) ---
        self.layer1 = self._make_layer(64, 64, stride=2) 
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)

        # --- Attention Head ---
        self.attention_pool = SelfAttentionPooling(input_dim=512)
        
        # --- Regression Head ---
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_outputs)

    def _make_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Feature Extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Attention Pooling
        x = self.attention_pool(x)
        
        # Prediction
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        # V27 CHANGE: No Activation.
        # We want raw linear output because the target "Fix" values 
        # range from -1.0 to +1.0 (after normalization).
        return x