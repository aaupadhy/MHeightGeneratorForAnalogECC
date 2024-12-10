# Function to calculate m-heights for a given G and X
def calculate_m_height(x_vectors, G, m):
    x_vectors = x_vectors.float()
    G = G.float()
    codewords = torch.matmul(x_vectors, G)
    abs_codewords = torch.abs(codewords)
    top_1_values = torch.amax(abs_codewords, dim=1)
    top_m_values, _ = torch.topk(abs_codewords, k=m + 1, dim=1, largest=True)
    mth_values = top_m_values[:, -1]
    mth_values = torch.clamp(mth_values, min=1e-8)  # Avoid division by zero
    return top_1_values / mth_values