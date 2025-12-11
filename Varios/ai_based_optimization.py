import torch
import torch.nn as nn


# ---------------------------------------------------------
# PointNet-like encoder por curva
# ---------------------------------------------------------
class CurveEncoder(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, hidden)
        )

    def forward(self, curve_points):
        """
        curve_points: list of tensors, each shape [N_i, 2]
        Returns: tensor shape [M, hidden]
        """
        zs = []
        for pts in curve_points:
            h = self.mlp(pts)          # [N_i, hidden]
            z = h.mean(dim=0)          # [hidden]
            zs.append(z)
        return torch.stack(zs, dim=0)


# ---------------------------------------------------------
# Set Transformer blocks
# ---------------------------------------------------------
class MultiheadAttention(nn.Module):
    def __init__(self, d_model=128, n_heads=4):
        super().__init__()
        self.att = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x):
        h, _ = self.att(x, x, x)
        x = x + h
        x = x + self.ff(x)
        return x


class SetTransformer(nn.Module):
    def __init__(self, d_model=128, n_heads=4, depth=2):
        super().__init__()
        self.layers = nn.ModuleList(
            [MultiheadAttention(d_model, n_heads) for _ in range(depth)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------
# Modelo completo
# ---------------------------------------------------------
class CurveSetModel(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.encoder = CurveEncoder(hidden)
        self.set_att = SetTransformer(hidden, n_heads=4, depth=2)

        self.decoder = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 3)   # m, b0, Δ
        )

    def forward(self, curves):
        """
        curves: list of tensors [N_i, 2]
        """
        Z = self.encoder(curves)       # [M, hidden]
        Z = self.set_att(Z.unsqueeze(0)).squeeze(0)  # atención sobre curvas

        # pool global
        Zg = Z.mean(dim=0)  # [hidden]
        out = self.decoder(Zg)
        m, b0, delta = out
        return m, b0, delta
