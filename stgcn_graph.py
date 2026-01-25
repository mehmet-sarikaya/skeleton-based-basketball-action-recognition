import numpy as np

def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node), dtype=np.float32)
    for i, j in link:
        A[j, i] = 1.0
    return A

def normalize_digraph(A):
    Dl = np.sum(A, 0)  # column sum
    w = A.shape[0]
    Dn = np.zeros((w, w), dtype=np.float32)
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    return A @ Dn

def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out), axis=0).astype(np.float32)  # (K=3, V, V)
    return A

class Graph:
    """
    Minimaler Graph wie im GitHub-Repo, aber parametrisierbar.
    edges: Liste von (u, v) mit 0-basierten Indizes, und Richtung ist egal:
           Wir nutzen edges als "inward" und erzeugen outward automatisch.
    """
    def __init__(self, num_node: int, edges, labeling_mode='spatial'):
        self.num_node = int(num_node)
        self.self_link = [(i, i) for i in range(self.num_node)]
        self.inward = [(int(u), int(v)) for (u, v) in edges]
        self.outward = [(v, u) for (u, v) in self.inward]
        self.neighbor = self.inward + self.outward
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode='spatial'):
        if labeling_mode == 'spatial':
            return get_spatial_graph(self.num_node, self.self_link, self.inward, self.outward)
        raise ValueError(f"Unknown labeling_mode: {labeling_mode}")
