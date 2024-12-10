import torch
from tqdm import tqdm
import numpy as np
import pickle
import matplotlib.pyplot as plt
from Dual_Optimizer import optimize_G_and_X
from Genetics import crossover # type: ignore
from Genetics import mutate # type: ignore
from m-height import calculate_m_height # type: ignore
import matplotlib.pyplot as plt


def store_best_G(best_G, filename="best_Gs.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(best_G.cpu().numpy(), f)
    print(f"Best G matrices saved to {filename}")

# Parameters
k, n, m = 6, 11, 4  # Example dimensions
G_population_size = 100
x_vector_count = 10000
iterations = 100
G_top_count = 20
G_mutation_range = (-10, 10)

# Run optimization
best_Gs, best_m_heights = optimize_G_and_X(k, n, m, G_population_size, x_vector_count,
                                            iterations, G_top_count, G_mutation_range)

store_best_G(best_Gs[0])

# Output results
print("Best G matrices and their m-heights:")
for G, height in zip(best_Gs, best_m_heights):
    print("G:\n", G.cpu().numpy(), "\nm-height:", height)