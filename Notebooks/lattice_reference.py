import networkx as nx
import numpy as np
import random
from networkx.utils import cumulative_distribution, discrete_sequence
from tqdm import tqdm
import statistics as st
from joblib import Parallel, delayed
import os

threads = os.cpu_count()

def lattice_reference(G, niter=1, D=None, connectivity=True):
    """
    We have removed the 'seed attribute' and put inside the code a 
    random module to choose between neighbours.
    Moreover, we have reduced the default 'niter' to 1 instead of 5, as declared from 
    the original algorithm.
    """
    local_conn = nx.connectivity.local_edge_connectivity
    
    if len(G) < 4:
        raise nx.NetworkXError("Graph has less than four nodes.")
    # Instead of choosing uniformly at random from a generated edge list,
    # this algorithm chooses nonuniformly from the set of nodes with
    # probability weighted by degree.

    G = G.copy()
    keys, degrees = zip(*G.degree())  # keys, degree
    cdf = cumulative_distribution(degrees)  # cdf of degree

    nnodes = len(G)
    nedges = nx.number_of_edges(G)
    if D is None:
        D = np.zeros((nnodes, nnodes))
        un = np.arange(1, nnodes)
        um = np.arange(nnodes - 1, 0, -1)
        u = np.append((0,), np.where(un < um, un, um))

        for v in range(int(np.ceil(nnodes / 2))):
            D[nnodes - v - 1, :] = np.append(u[v + 1 :], u[: v + 1])
            D[v, :] = D[nnodes - v - 1, :][::-1]

    niter = niter * nedges
    # maximal number of rewiring attempts per 'niter'
    max_attempts = int(nnodes * nedges / (nnodes * (nnodes - 1) / 2))
    # Parallelization
    # run the main function in parallel to speed up the process
    Parallel(n_jobs=threads)(delayed(latticization)(keys, cdf, G, D, connectivity, local_conn, max_attempts) for _ in tqdm(range(niter)))
    return G
"""
--- ORIGINAL FUNCTION ---

            (ai, ci) = discrete_sequence(2, cdistribution=cdf)
            if ai == ci:
                continue  # same source, skip
            a = keys[ai]  # convert index to label
            c = keys[ci]
            # choose target uniformly from neighbors
            b = random.choice(list(G.neighbors(a)))
            d = random.choice(list(G.neighbors(c)))
            bi = keys.index(b)
            di = keys.index(d)

            if b in [a, c, d] or d in [a, b, c]:
                continue  # all vertices should be different

            # don't create parallel edges
            if (d not in G[a]) and (b not in G[c]):
                if D[ai, bi] + D[ci, di] >= D[ai, ci] + D[bi, di]:
                    # only swap if we get closer to the diagonal
                    G.add_edge(a, d)
                    G.add_edge(c, b)
                    G.remove_edge(a, b)
                    G.remove_edge(c, d)

                    # Check if the graph is still connected
                    if connectivity and local_conn(G, a, b) == 0:
                        # Not connected, revert the swap
                        G.remove_edge(a, d)
                        G.remove_edge(c, b)
                        G.add_edge(a, b)
                        G.add_edge(c, d)
                    else:
                        break
            n += 1

    return G
"""
# Function to run in parallel
def latticization(keys, cdf, G, D, connectivity, local_conn, max_attempts):
    n = 0
    while n < max_attempts:
        (ai, ci) = discrete_sequence(2, cdistribution=cdf)
        if ai == ci:
            continue  # same source, skip
        a = keys[ai]  # convert index to label
        c = keys[ci]
        # choose target uniformly from neighbors
        b = random.choice(list(G.neighbors(a)))
        d = random.choice(list(G.neighbors(c)))
        bi = keys.index(b)
        di = keys.index(d)
        if b in [a, c, d] or d in [a, b, c]:
            continue  # all vertices should be differen
        # don't create parallel edges
        if (d not in G[a]) and (b not in G[c]):
            if D[ai, bi] + D[ci, di] >= D[ai, ci] + D[bi, di]:
                # only swap if we get closer to the diagonal
                G.add_edge(a, d)
                G.add_edge(c, b)
                G.remove_edge(a, b)
                G.remove_edge(c, d)
                # Check if the graph is still connected
                if connectivity and local_conn(G, a, b) == 0:
                    # Not connected, revert the swap
                    G.remove_edge(a, d)
                    G.remove_edge(c, b)
                    G.add_edge(a, b)
                    G.add_edge(c, d)
                else:
                    break
        n += 1
#-------------------------
def unconnected_average_path_length(network):
    store = list()
    for N in (network.subgraph(n).copy() for n in nx.connected_components(network)): # for each connected component of the network (N) 
        store.append(nx.average_shortest_path_length(N)) # compute L and append it to the list
    if len(store) == 1:
        return store[0]
    else:
        meaningful_paths = [single_path for single_path in store if single_path != 1.0] # Remove null values
        return st.mean(meaningful_paths)

def random_graph_builder(n, E):
    # Build a random graph with the same number of nodes and edges of the original network
    random_graph = nx.gnm_random_graph(n, E)
    if nx.is_connected(random_graph):
        shortest_path = nx.average_shortest_path_length(random_graph)
    else:
        shortest_path = unconnected_average_path_length(random_graph)
    return shortest_path