import torch
import torch.nn as nn

class STGCNBlock(nn.Module):
    """
    Adaptive Spatio-Temporal Graph Convolutional Block.
    Uses a learnable adjacency matrix instead of a fixed physical graph.
    """
    def __init__(self, in_channels, out_channels, num_nodes, temporal_kernel_size=5, stride=1, dropout=0.1):
        super().__init__()
        # Learnable adjacency matrix (initialized to identity + small noise)
        self.A = nn.Parameter(torch.eye(num_nodes) + torch.randn(num_nodes, num_nodes) * 0.01)
        
        # Spatial convolution is a 1x1 conv on the node features after graph multiplication
        self.spatial_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # Temporal convolution operates over the T dimension
        padding = ((temporal_kernel_size - 1) // 2, 0)
        self.temporal_conv = nn.Conv2d(
            out_channels, out_channels, 
            kernel_size=(temporal_kernel_size, 1), 
            padding=padding, 
            stride=(stride, 1)
        )
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        """
        x: (N, C, T, V)
        """
        res = self.residual(x)
        
        # Spatial Graph Convolution
        # multiply feature channels by adjacency matrix: sum over w (V)
        x = torch.einsum('vw, nctw -> nctv', self.A, x)
        x = self.spatial_conv(x)
        
        # Temporal Convolution
        x = self.temporal_conv(x)
        
        x = self.bn(x)
        x = self.dropout(x)
        
        return self.relu(x + res)

class STGCN_Frontend(nn.Module):
    """
    Front-end module that replaces linear projection.
    Takes (B, T, 822) -> reshapes to (B, C, T, V) -> processes through ST-GCN -> pools to (B, T, d_model)
    """
    def __init__(self, in_features=822, num_nodes=137, coords=6, d_model=512, dropout=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.coords = coords
        assert in_features == num_nodes * coords, f"in_features {in_features} != num_nodes {num_nodes} * coords {coords}"

        self.block1 = STGCNBlock(coords, 64, num_nodes, dropout=dropout)
        self.block2 = STGCNBlock(64, 128, num_nodes, dropout=dropout)
        self.block3 = STGCNBlock(128, 256, num_nodes, dropout=dropout)
        
        self.proj = nn.Conv2d(256, d_model, kernel_size=1)

    def forward(self, x):
        """
        x: (B, T, 822)
        Returns: (B, T, d_model)
        """
        B, T, _ = x.shape
        # Reshape and permute to (B, C, T, V)
        x = x.view(B, T, self.num_nodes, self.coords).permute(0, 3, 1, 2).contiguous()
        
        # Pass through ST-GCN blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Project to d_model channels
        x = self.proj(x) # (B, d_model, T, V)
        
        # Spatial Mean Pooling: aggregate across all nodes (V)
        x = x.mean(dim=-1) # (B, d_model, T)
        
        # Permute back to sequence format (B, T, d_model)
        x = x.permute(0, 2, 1).contiguous()
        return x
