import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Thin Plate Spline (TPS) Grid Generator ---
def tps_grid(source_points, target_points, out_h, out_w):
    """
    source_points: [B, K, 2] (normalized -1~1)
    target_points: [B, K, 2] (normalized -1~1)
    out_h, out_w: output grid size
    Returns: grid [B, out_h, out_w, 2] for grid_sample
    """
    B, K, _ = source_points.shape
    device = source_points.device
    # Create mesh grid
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, out_h, device=device),
        torch.linspace(-1, 1, out_w, device=device),
        indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=-1)  # [out_h, out_w, 2]
    grid_flat = grid.view(-1, 2)  # [out_h*out_w, 2]

    # Compute radial basis function
    def U_func(r):
        # r: [N, K]
        return r**2 * torch.log(r**2 + 1e-6)

    N = out_h * out_w
    grid_flat = grid_flat.unsqueeze(0).expand(B, N, 2)  # [B, N, 2]
    dists = grid_flat.unsqueeze(2) - source_points.unsqueeze(1)  # [B, N, K, 2]
    r = torch.norm(dists, dim=-1)  # [B, N, K]
    U = U_func(r)  # [B, N, K]

    # Build TPS system
    ones = torch.ones((B, K, 1), device=device)
    zeros = torch.zeros((B, 3, 3), device=device)
    P = torch.cat([ones, source_points], dim=2)  # [B, K, 3]
    L = torch.zeros((B, K+3, K+3), device=device)
    L[:, :K, :K] = U_func(torch.cdist(source_points, source_points))
    L[:, :K, K:] = P
    L[:, K:, :K] = P.transpose(1, 2)
    # Y = [target_points; zeros]
    Y = torch.cat([target_points, torch.zeros((B, 3, 2), device=device)], dim=1)
    # Solve L * W = Y
    W = torch.linalg.solve(L, Y)  # [B, K+3, 2]
    # Compute mapping for all grid points
    U_ = U  # [B, N, K]
    P_ = torch.cat([torch.ones((B, N, 1), device=device), grid_flat], dim=2)  # [B, N, 3]
    mapped = torch.bmm(U_, W[:, :K]) + torch.bmm(P_, W[:, K:])  # [B, N, 2]
    mapped = mapped.view(B, out_h, out_w, 2)
    return mapped

class TPSGeometricTransform(nn.Module):
    def __init__(self, channels, num_ctrl_pts=16, out_size=(256, 256)):
        super().__init__()
        self.num_ctrl_pts = num_ctrl_pts
        self.out_h, self.out_w = out_size
        self.localization = nn.Sequential(
            nn.Conv2d(channels, 8, kernel_size=7),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(8, num_ctrl_pts * 2)
        # 初始化控制点为均匀分布
        ctrl_pts = self._build_ctrl_pts(num_ctrl_pts)
        self.fc.weight.data.zero_()
        self.fc.bias.data.copy_(ctrl_pts.view(-1))
    def _build_ctrl_pts(self, num_pts):
        # 均匀分布在[-1,1]范围内
        import math
        grid_size = int(math.sqrt(num_pts))
        x = torch.linspace(-1, 1, grid_size)
        y = torch.linspace(-1, 1, grid_size)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        ctrl_pts = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)
        return ctrl_pts
    def forward(self, x):
        B = x.size(0)
        xs = self.localization(x)
        xs = xs.view(B, -1)
        pred_ctrl_pts = self.fc(xs).view(B, self.num_ctrl_pts, 2)
        source_ctrl_pts = self._build_ctrl_pts(self.num_ctrl_pts).to(x.device).unsqueeze(0).expand(B, -1, -1)
        grid = tps_grid(source_ctrl_pts, pred_ctrl_pts, self.out_h, self.out_w)  # [B, H, W, 2]
        x = F.grid_sample(x, grid, align_corners=False)
        return x
