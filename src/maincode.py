import torch
import numpy as np

def calculate_best_m_height(G, x_vectors):
    xG = x_vectors @ G
    sorted_abs_values = torch.sort(torch.abs(xG), dim=1, descending=True)[0]
    m_heights = sorted_abs_values[:, 0] / torch.clamp(sorted_abs_values[:, 4], min=1e-9)
    return torch.max(m_heights).item()

best_Gs = torch.empty((0, 5, 11), device='cuda')
best_m_heights = []

