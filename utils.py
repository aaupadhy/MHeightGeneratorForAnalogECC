# Function for generating a random full-rank matrix
def generate_full_rank_matrix(k, n, int_range=(-1000, 1000)):
    while True:
        G = np.random.randint(int_range[0], int_range[1] + 1, (k, n))
        if np.linalg.matrix_rank(G) == k:
            return torch.tensor(G)