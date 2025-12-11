import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Variable names
vars = ["T17", "M", "T", "I17", "I23", "Ta", "I6", "S", "G", "P", "Q", "Qr"]

# Interaction matrix (from your table)
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

# Create directed graph
G = nx.DiGraph()

# Add nodes
G.add_nodes_from(vars)

# Add edges with weight and color
for i, src in enumerate(vars):
    for j, tgt in enumerate(vars):
        weight = A[i, j]
        if abs(weight) > 0.001:  # optional threshold for clarity
            G.add_edge(src, tgt, weight=weight, color='green' if weight>0 else 'red')

# Get edge colors and weights
edges = G.edges()
colors = [G[u][v]['color'] for u,v in edges]
weights = [abs(G[u][v]['weight'])*5 for u,v in edges]  # scale for visibility

# Draw graph
plt.figure(figsize=(12,12))
pos = nx.spring_layout(G, seed=42)  # force-directed layout
nx.draw(G, pos, with_labels=True, node_size=1200, node_color='lightblue', 
        arrowsize=20, edge_color=colors, width=weights)
plt.title("SINDy Linear Interaction Network")
plt.savefig('adjacency_matrix.png')
plt.show()
