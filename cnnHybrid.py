import torch.nn as nn
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
    

class ConvNeXtWithTransformer(nn.Module):
    def __init__(self, num_classes=212, embed_dim=512, nhead=16, num_transformer_layers=1):
        super().__init__()

        self.convnext = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        self.convnext = nn.Sequential(*list(self.convnext.children())[:-1])

        self.orig_embed_dim = 1024
        self.projection = nn.Linear(self.orig_embed_dim, embed_dim)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True),
            num_layers=num_transformer_layers
        )

        self.fc = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        x = self.convnext(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.projection(x)
        x = self.transformer(x)
        x = x.squeeze(1)
        x = self.fc(x)
        return x