import numpy as np

def generate_full_rank_matrix(start, end, num_rows, num_cols):
    """
    Generates a full-matrix with each entry in range [start, end] of shape (x,y)
    
    Input:
    start: Range Start for each entry in the matrix
    end: Range End for each entry in the matrix
    num_rows: Number of Rows in the matrix
    num_cols: Number of Cols in the matrix
    
    Returns:
    Full-Rank ND-array of shape (num_rows, num_cols)
    
    """
    while True:
        G = np.random.randint(start, end, (num_rows, num_cols))
        if np.linalg.matrix_rank(G) == min(num_rows, num_cols):
            return G
        
