from tqdm import tqdm
import torch
import numpy as np
import random

alpha = 1
beta = 1

x_vectors = torch.randn(100000, 5, device='cuda')

def generate_full_rank_matrix():
    while True:
        G = np.random.randint(-1000, 1000, (5, 11))
        if np.linalg.matrix_rank(G) == 5:
            return G

G_matrices = torch.tensor(np.array([generate_full_rank_matrix() for _ in range(100)]), dtype=torch.float32).to('cuda')

for iteration in tqdm(range(1000)):
    m_heights = torch.tensor([calculate_best_m_height(G, x_vectors) for G in G_matrices])
    sorted_indices = torch.argsort(m_heights)
    best_Gs = G_matrices[sorted_indices[:10]]
    best_m_heights = m_heights[sorted_indices[:10]].tolist()

    mutated_Gs = torch.stack([mutate(G) for G in best_Gs])
    crossover_Gs = torch.stack([crossover(best_Gs[i], best_Gs[j]) for i in range(10) for j in range(i + 1, 10)])
    G_matrices = torch.cat([mutated_Gs, crossover_Gs])

print("Final 10 Best G Matrices and Corresponding m-heights:")
for G, m_height in zip(best_Gs, best_m_heights):
    print("m-height:", m_height)

if torch.linalg.matrix_rank(best_Gs[0]) == 5:
    print("YES")
    print(best_Gs[0].cpu().numpy())
