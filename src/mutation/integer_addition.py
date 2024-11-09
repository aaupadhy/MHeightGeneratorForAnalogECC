import numpy as np

def mutate(G):
    return G + torch.randint(-1, 2, G.shape, device='cuda')