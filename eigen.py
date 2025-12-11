import numpy as np
import networkx as nx

# Example: your 12x12 interaction matrix A
A = np.array([
    [-0.008, -0.003, -0.000, -0.005, -0.001, -0.001, 0.009, -0.006, -0.001, -0.002, 0.012, -0.002],
    [-0.001, 0.007, -0.013, -0.000, 0.006, 0.000, 0.011, 0.010, 0.001, -0.001, 0.000, 0.003],
    [-0.004, -0.016, -0.011, 0.024, -0.012, -0.004, -0.015, -0.114, -0.006, -0.002, 0.090, -0.009],
    [0.107, -0.010, -0.040, -0.007, -0.011, 0.004, -0.086, -0.102, 0.000, 0.006, 0.115, -0.002],
    [-0.000, 0.001, -0.009, -0.005, 0.003, 0.001, 0.012, 0.009, 0.000, -0.002, 0.003, 0.002],
    [-0.000, 0.002, -0.005, 0.001, 0.001, 0.000, 0.004, -0.006, 0.000, -0.001, 0.008, 0.001],
    [0.050, 0.011, -0.017, -0.012, 0.004, 0.008, -0.036, -0.043, 0.006, 0.007, 0.046, 0.011],
    [-0.007, -0.010, -0.011, 0.013, -0.011, -0.002, -0.004, -0.086, -0.004, -0.002, 0.075, -0.009],
    [-0.001, 0.000, -0.005, -0.000, 0.002, -0.000, 0.006, -0.003, -0.000, -0.001, 0.006, 0.000],
    [-0.002, 0.000, -0.003, 0.003, 0.000, -0.000, 0.002, -0.005, -0.000, -0.001, 0.005, -0.000],
    [0.134, 0.053, -0.018, -0.044, 0.029, 0.032, -0.105, -0.083, 0.028, 0.025, 0.069, 0.042],
    [-0.000, 0.003, -0.010, -0.003, 0.003, 0.000, 0.007, 0.010, 0.000, -0.001, 0.001, 0.002]
])

species = ['T17', 'M', 'T', 'I17', 'I23', 'Ta', 'I6', 'S', 'G', 'P', 'Q', 'Qr']

# Use absolute values for weighted centrality if you want magnitude-based influence
A_abs = np.abs(A)

# Create directed graph
G = nx.from_numpy_array(A_abs, create_using=nx.DiGraph)

# Compute eigenvector centrality
centrality = nx.eigenvector_centrality_numpy(G)

# Sort species by descending centrality
ranked_species = sorted(centrality.items(), key=lambda x: x[1], reverse=True)

print("Species ranked by eigencentrality:")
for sp, val in ranked_species:
    print(f"{species[sp]}: {val:.4f}")
