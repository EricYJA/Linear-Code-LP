import numpy as np
import pickle

def generate_matrix(rows, cols, rank):
    while True:
        matrix = np.random.rand(rows, cols)
        if np.linalg.matrix_rank(matrix) == rank:
            return matrix

def generate_random_vectors(num_vectors, length):
    # Generate `num_vectors` random vectors, each of length `length`
    return [np.random.rand(length) for _ in range(num_vectors)]

# Generate G1 and G2 with the specified ranks
G1 = generate_matrix(5, 11, 5)
G2 = generate_matrix(6, 11, 6)

# Store the matrices in a dictionary
matrices = {
    "setting1": G1,
    "setting2": G2
}

# Save the generator matrices to a file named "Task1"
# with open("Task1", "wb") as f:
#     pickle.dump(matrices, f)

with open("G1", "wb") as f1:
    pickle.dump(G1, f1)

with open("G2", "wb") as f2:
    pickle.dump(G2, f2)

# Generate 100 random vectors for each generator matrix
x_vectors_G1 = generate_random_vectors(100, 5)  # 100 vectors of length 5 for G1
x_vectors_G2 = generate_random_vectors(100, 6)  # 100 vectors of length 6 for G2

# Save the random vectors for G1 and G2 to separate files
with open("X_vectors_G1", "wb") as f1:
    pickle.dump(x_vectors_G1, f1)

with open("X_vectors_G2", "wb") as f2:
    pickle.dump(x_vectors_G2, f2)
