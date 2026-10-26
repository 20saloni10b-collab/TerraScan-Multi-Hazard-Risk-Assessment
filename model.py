"""
STEP 2: MULTIMODAL FUSION MODEL
--------------------------------
Teen parts:
  1. Image Branch  -> CNN (ResNet18 backbone, pretrained) satellite patches se
                       spatial/visual features nikalta hai.
  2. Tabular Branch -> MLP jo numeric/environmental features process karta hai.
  3. Fusion Head     -> Dono branches ke embeddings ko concatenate karke
                       final classifier (risk level: low/medium/high) tak le jata hai.

Fusion strategy: "Late Fusion" (feature-level concatenation).
Chaho to isay "attention-based fusion" mein bhi upgrade kar sakte ho (advanced version end me diya hai).
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ImageEncoder(nn.Module):
    def __init__(self, out_dim=128, pretrained=True, backbone="resnet18"):
        super().__init__()
        if backbone == "resnet18":
            net = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
            feat_dim = 512
        elif backbone == "resnet50":
            net = models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
            feat_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Remove final FC layer, keep feature extractor
        self.features = nn.Sequential(*list(net.children())[:-1])  # -> (B, feat_dim, 1, 1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.proj(x)
        return x  # (B, out_dim)


class ResidualBlock(nn.Module):
    """Ek residual fully-connected block — skip-connection se deeper network
    train karna aasan hota hai aur vanishing-gradient issue kam hoti hai
    (CNNs mein ResNet jo karta hai, wahi idea tabular data pe)."""
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        out = out + identity          # <- skip connection
        return self.act(out)


class TabularEncoder(nn.Module):
    """Deep Residual MLP for tabular/environmental features.
    Simple MLP ke bajaye ab ye:
      - Input projection (Linear + BatchNorm + ReLU)
      - N residual blocks (deeper representation learning)
      - Output projection to embedding space
    Ye TabNet / FT-Transformer jaise modern tabular deep-learning models ke
    "residual feature learning" idea se inspired hai, lekin lightweight rakha
    gaya hai taaki chhote environmental datasets pe overfit na ho."""
    def __init__(self, in_dim, out_dim=64, hidden_dim=128, n_blocks=3, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout=dropout) for _ in range(n_blocks)
        ])
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)  # (B, out_dim)


class MultimodalFusionModel(nn.Module):
    def __init__(self, tabular_in_dim, num_classes=3,
                 img_embed_dim=128, tab_embed_dim=64, pretrained=True, backbone="resnet18"):
        super().__init__()
        self.image_encoder = ImageEncoder(out_dim=img_embed_dim, pretrained=pretrained, backbone=backbone)
        self.tabular_encoder = TabularEncoder(tabular_in_dim, out_dim=tab_embed_dim)

        fused_dim = img_embed_dim + tab_embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, image, tabular):
        img_feat = self.image_encoder(image)
        tab_feat = self.tabular_encoder(tabular)
        fused = torch.cat([img_feat, tab_feat], dim=1)
        out = self.classifier(fused)
        return out


# -----------------------------------------------------------------
# ADVANCED (optional): Attention-based fusion instead of simple concat
# -----------------------------------------------------------------
class AttentionFusionModel(nn.Module):
    """
    Cross-modal attention: tabular features 'query' ban ke image features
    'key/value' pe attend karte hain -> zyada informative fusion.
    """
    def __init__(self, tabular_in_dim, num_classes=3,
                 img_embed_dim=128, tab_embed_dim=128, pretrained=True):
        super().__init__()
        self.image_encoder = ImageEncoder(out_dim=img_embed_dim, pretrained=pretrained)
        self.tabular_encoder = TabularEncoder(tabular_in_dim, out_dim=tab_embed_dim)

        assert img_embed_dim == tab_embed_dim, "Attention ke liye dono dims equal rakho"
        self.attn = nn.MultiheadAttention(embed_dim=img_embed_dim, num_heads=4, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(img_embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, image, tabular):
        img_feat = self.image_encoder(image).unsqueeze(1)     # (B,1,D)
        tab_feat = self.tabular_encoder(tabular).unsqueeze(1)  # (B,1,D)

        attn_out, _ = self.attn(query=tab_feat, key=img_feat, value=img_feat)
        fused = attn_out.squeeze(1)
        out = self.classifier(fused)
        return out


if __name__ == "__main__":
    # quick shape sanity check
    model = MultimodalFusionModel(tabular_in_dim=7, num_classes=3)
    dummy_img = torch.randn(4, 3, 64, 64)
    dummy_tab = torch.randn(4, 7)
    out = model(dummy_img, dummy_tab)
    print("Output shape:", out.shape)  # (4, 3)
