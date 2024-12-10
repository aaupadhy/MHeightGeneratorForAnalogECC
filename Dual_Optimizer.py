# Unified optimization function for G and X
from Genetics import crossover
from Genetics import mutate
from m-height import calculate_m_height # type: ignore
from utils import generate_full_rank_matrix
import matplotlib.pyplot as plt

def optimize_G_and_X(k, n, m, G_population_size=100, x_vector_count=10000,
                     iterations=10, G_top_count=20, G_mutation_range=(-1, 1)):
    # Initialize G population and X vectors
    G_population = torch.stack([generate_full_rank_matrix(k, n) for _ in range(G_population_size)]).to('cuda')
    x_vectors = torch.randn(x_vector_count, k, device='cuda')

    best_Gs = torch.empty((0, k, n), device='cuda')
    best_m_heights = []
    avg_m_heights_per_iteration = []

    for iteration in tqdm(range(iterations)):
        # Phase 1: Optimize G matrices
        G_m_heights = torch.tensor([calculate_m_height(x_vectors, G, m).max().item() for G in G_population])
        sorted_indices = torch.argsort(G_m_heights)
        top_Gs = G_population[sorted_indices[:G_top_count]]
        top_m_heights = G_m_heights[sorted_indices[:G_top_count]]
        
        avg_m_heights_per_iteration.append(top_m_heights.mean().item())

        # Genetic programming for G
        mutated_Gs = mutate(top_Gs, mutation_range=G_mutation_range, flag = 0)
        i, j = torch.triu_indices(G_top_count, G_top_count, 1, device=top_Gs.device)
        crossover_Gs = torch.vstack([crossover(top_Gs[i], top_Gs[j])])
        random_Gs = torch.stack([generate_full_rank_matrix(k, n) for _ in range(G_top_count)]).to('cuda')
        G_population = torch.cat([mutated_Gs, crossover_Gs, top_Gs, random_Gs])

        # Phase 2: Optimize X vectors for each top G
        refined_x_vectors = []
        for G in top_Gs:
            best_xs, _ = optimize_X_for_G(x_vectors, G, k, m)
            refined_x_vectors.append(best_xs)

        # Update X vector pool
        x_vectors = torch.cat(refined_x_vectors, dim=0)

        # Update best Gs
        best_Gs = top_Gs
        best_m_heights = top_m_heights

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, iterations + 1), avg_m_heights_per_iteration, marker='o', label="Avg m-height of top Gs")
    plt.xlabel("Iteration")
    plt.ylabel("Average m-height")
    plt.title("Average m-height of Top Gs per Iteration")
    plt.legend()
    plt.grid()
    plt.show()    

    return best_Gs, best_m_heights