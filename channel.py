import torch

def embedding(x):
    x = x.float()
    B, C, H, W = x.shape
    device = x.device

    xs = torch.linspace(0, 1, W, device=device, dtype=x.dtype).view(1, 1, 1, W).expand(B, 1, H, W)
    ys = torch.linspace(0, 1, H, device=device, dtype=x.dtype).view(1, 1, H, 1).expand(B, 1, H, W)

    x = torch.cat([x, xs, ys], dim=1)  # (B,3,H,W)
    return x

def hessiana(x, hess):
    x = x.float()
    B, C, H, W = x.shape
    device = x.device

    x = torch.cat([x, hess], dim=1)  # (B,2,H,W)
    return x
