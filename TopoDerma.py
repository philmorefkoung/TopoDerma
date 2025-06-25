import torch
import torch.nn as nn
import timm

class TopoDerma(nn.Module):
    def __init__(
        self,
        num_classes: int,
        ph_in_dim: int,
        ph_token_len: int,
        ph_embed_dim: int,
        ph_num_heads: int,
        tf_num_heads: int,
        tf_layers: int,
        dropout: float,
    ):
        super().__init__()

        #   Image branch (Vision Transformer)
        self.swin = timm.create_model(
            "swin_tiny_patch4_window7_224", pretrained=True, num_classes=0
        )
        self.common_dim = self.swin.num_features  # 768 for Swin
        assert self.common_dim in {512, 768}, f"Unexpected feature dim {self.common_dim}"

        #   Persistent homology (PH) branch 
        self.PH_embed = nn.Linear(ph_in_dim, ph_embed_dim)

        ph_enc_layer = nn.TransformerEncoderLayer(
            d_model=ph_embed_dim,
            nhead=ph_num_heads,
            dim_feedforward=2 * ph_embed_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.ph_encoder = nn.TransformerEncoder(ph_enc_layer, num_layers=1)

        self.ph_token_len = ph_token_len
        self.ph_in_dim = ph_in_dim
        self.ph_to_common = nn.Linear(ph_embed_dim, self.common_dim)

        #  Classification transformer 
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.common_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.common_dim,
            nhead=tf_num_heads,
            dim_feedforward=2 * self.common_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, tf_layers)
        self.classifier = nn.Linear(self.common_dim, num_classes)

        self.img_dropout = nn.Dropout(dropout)
        self.ph_dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, image: torch.Tensor, PH: torch.Tensor):
        """
        Args
        ----
        image : Tensor[B,3,224,224]
        PH    : Tensor[B,L,ph_in_dim] or Tensor[B, ph_in_dim * L]
        """
        B = image.size(0)

        #   Image tokens 
        img_feats = self.swin.forward_features(image)  # (B,N,common_dim)
        if img_feats.dim() == 4:  # (B,C,H,W) or (B,H,W,C)
            if img_feats.shape[1] == self.common_dim:       # NCHW
                img_feats = img_feats.flatten(2).transpose(1, 2)
            elif img_feats.shape[-1] == self.common_dim:    # NHWC
                img_feats = img_feats.reshape(B, -1, self.common_dim)
            else:
                raise RuntimeError("Unrecognized Swin output layout")
        img_tok = self.img_dropout(img_feats)

        #   PH tokens 
        if PH.dim() == 2:  # flat vector
            PH = PH.view(B, self.ph_token_len, self.ph_in_dim)
        ph_tok = self.PH_embed(PH)
        ph_tok = self.ph_dropout(ph_tok)
        ph_tok = self.ph_encoder(ph_tok)
        ph_tok = self.ph_to_common(ph_tok)

        #   Fuse and classify 
        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, img_tok, ph_tok], dim=1)
        enc = self.transformer(seq)
        logits = self.classifier(enc[:, 0])
        return logits