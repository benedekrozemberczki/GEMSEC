import math
import random
import community
import numpy as np
import networkx as nx
from tqdm import tqdm
from collections import Counter
from sklearn.cluster import KMeans

def normalized_overlap(g, node_1, node_2):
    """
    Function to calculate the normalized neighborhood overlap.
    :param g: NX graph.
    :param node_1: Node 1. of a pair.
    :param node_2: Node 2. of a pair.
    """
    nebs_1 = set(nx.neighbors(g, node_1))
    nebs_2 = set(nx.neighbors(g, node_2))
    inter = len(nebs_1.intersection(nebs_2))
    unio = len(nebs_1.union(set(nx.neighbors(g, node_2))))
    return float(inter)/float(unio)

def overlap(g, node_1, node_2):
    """
    Function to calculate the neighborhood overlap.
    :param g: NX graph.
    :param node_1: Node 1. of a pair.
    :param node_2: Node 2. of a pair.
    """
    nebs_1 = set(nx.neighbors(g, node_1))
    nebs_2 = set(nx.neighbors(g, node_2))
    inter = len(nebs_1.intersection(nebs_2))
    return float(inter)

def unit(g, node_1, node_2):
    """
    Function to calculate the "unit" weight.
    :param g: NX graph.
    :param node_1: Node 1. of a pair.
    :param node_2: Node 2. of a pair.
    """
    return 1

def min_norm(g, node_1, node_2):
    """
    Function to calculate the minimum normalized neighborhood overlap.
    :param g: NX graph.
    :param node_1: Node 1. of a pair.
    :param node_2: Node 2. of a pair.
    """
    nebs_1 = set(nx.neighbors(g, node_1))
    nebs_2 = set(nx.neighbors(g, node_2))
    inter = len(nebs_1.intersection(nebs_2))
    min_norm = min(len(nebs_1), len(nebs_2))
    return float(inter)/float(min_norm)

def overlap_generator(args, graph):
    """
    Function to generate weight for all of the edges.
    """
    if args.overlap_weighting == "normalized_overlap":
        overlap_weighter = normalized_overlap
    elif args.overlap_weighting == "overlap":
        overlap_weighter = overlap
    elif args.overlap_weighting == "min_norm":
        overlap_weighter = min_norm
    else:
        overlap_weighter = unit
    print(" ")
    print("Weight calculation started.")
    print(" ")
    edges = nx.edges(graph)
    weights = {e: overlap_weighter(graph, e[0], e[1]) for e in tqdm(edges)}
    weights_prime = {(e[1], e[0]): value for e, value in weights.items()}
    weights.update(weights_prime)
    print(" ")
    return weights

def index_generation(weights, a_walk):
    """
    Function to generate overlaps and indices.
    """
    edges = [(a_walk[i], a_walk[i+1]) for i in range(0, len(a_walk)-1)]
    edge_set_1 = np.array(range(0, len(a_walk)-1))
    edge_set_2 = np.array(range(1, len(a_walk)))
    overlaps = np.array(list(map(lambda x: weights[x], edges))).reshape((-1, 1))
    return edge_set_1, edge_set_2, overlaps

def batch_input_generator(a_walk, random_walk_length, window_size):
    """
    Function to generate features from a node sequence.
    """
    seq_1 = [a_walk[j] for j in range(random_walk_length-window_size)]
    seq_2 = [a_walk[j] for j in range(window_size, random_walk_length)]
    return np.array(seq_1 + seq_2)

def batch_label_generator(a_walk, random_walk_length, window_size):
    """
    Function to generate labels from a node sequence.
    """
    grams_1 = [a_walk[j+1:j+1+window_size] for j in range(random_walk_length-window_size)]
    grams_2 = [a_walk[j-window_size:j] for j in range(window_size, random_walk_length)]
    return np.array(grams_1 + grams_2)

def gamma_incrementer(step, gamma_0, gamma_final, current_gamma, num_steps):
    if step > 1:
        exponent = (0-np.log10(gamma_0))/float(num_steps)
        current_gamma = current_gamma * (10 **exponent)*(gamma_final-gamma_0)
        current_gamma = current_gamma + gamma_0
    return current_gamma

def neural_modularity_calculator(graph, embedding, means):
    """
    Function to calculate the GEMSEC cluster assignments.
    """
    assignments = {}
    for node in graph.nodes():
        positions = means-embedding[node, :]
        values = np.sum(np.square(positions), axis=1)
        index = np.argmin(values)
        assignments[int(node)] = int(index)
    modularity = community.modularity(assignments, graph)
    return modularity, assignments

def classical_modularity_calculator(graph, embedding, args):
    """
    Function to calculate the DeepWalk cluster centers and assignments.
    """
    kmeans = KMeans(n_clusters=args.cluster_number, random_state=0, n_init=1).fit(embedding)
    assignments = {i: int(kmeans.labels_[i]) for i in range(embedding.shape[0])}
    modularity = community.modularity(assignments, graph)
    return modularity, assignments

class RandomWalker:
    """
    Class to generate vertex sequences.
    """
    def __init__(self, graph, repetitions, length):
        print("Model initialization started.")
        self.graph = graph
        self.nodes = [node for node in self.graph.nodes()]
        self.repetitions = repetitions
        self.length = length

    def small_walk(self, start_node):
        """
        Generate a node sequence from a start node.
        """
        walk = [start_node]
        while len(walk) != self.length:
            end_point = walk[-1]
            neighbors = [neb for neb in nx.neighbors(self.graph, end_point)]
            if len(neighbors) > 0:
                walk.append(random.choice(neighbors))
            else:
                break
        return walk

    def count_frequency_values(self):
        """
        Calculate the co-occurence frequencies.
        """
        raw_counts = [node for walk in self.walks for node in walk]
        counts = Counter(raw_counts)
        self.degrees = [counts[i] for i in range(len(self.nodes))]

    def do_walks(self):
        """
        Do a series of random walks.
        """
        self.walks = []
        for rep in range(0, self.repetitions):
            random.shuffle(self.nodes)
            print(" ")
            print("Random walk series " + str(rep+1) + ". initiated.")
            print(" ")
            for node in tqdm(self.nodes):
                walk = self.small_walk(node)
                self.walks.append(walk)
        self.count_frequency_values()
        return self.degrees, self.walks

class SecondOrderRandomWalker:

    def __init__(self, nx_G, is_directed, p, q):
        self.G = nx_G
        self.nodes = nx.nodes(self.G)
        print("Edge weighting.\n")
        for edge in tqdm(self.G.edges()):
            self.G[edge[0]][edge[1]]["weight"] = 1.0
            self.G[edge[1]][edge[0]]["weight"] = 1.0
        self.is_directed = is_directed
        self.p = p
        self.q = q

    def node2vec_walk(self, walk_length, start_node):
        """
        Simulate a random walk starting from start node.
        """
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], alias_edges[(prev, cur)][1])]
                    walk.append(next)
            else:
                break

        return walk

    def count_frequency_values(self, walks):
        """ 
        Calculate the co-occurence frequencies.
        """
        raw_counts = [node for walk in walks for node in walk]
        counts = Counter(raw_counts)
        self.degrees = [counts[i] for i in range(0,len(self.nodes))]
        return self.degrees 

    def simulate_walks(self, num_walks, walk_length):
        """
        Repeatedly simulate random walks from each node.
        """
        G = self.G
        walks = []
        nodes = list(G.nodes())
        for walk_iter in range(num_walks):
            print(" ")
            print("Random walk series " + str(walk_iter+1) + ". initiated.")
            print(" ")
            random.shuffle(nodes)
            for node in tqdm(nodes):
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

        return walks, self.count_frequency_values(walks)

    def get_alias_edge(self, src, dst):
        """
        Get the alias edge setup lists for a given edge.
        """
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]["weight"]/p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]["weight"])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]["weight"]/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        """
        Preprocessing of transition probabilities for guiding the random walks.
        """
        G = self.G
        is_directed = self.is_directed

        alias_nodes = {}
        print("")
        print("Preprocesing.\n")
        for node in tqdm(G.nodes()):
             unnormalized_probs = [G[node][nbr]["weight"] for nbr in sorted(G.neighbors(node))]
             norm_const = sum(unnormalized_probs)
             normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
             alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}
        triads = {}

        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in tqdm(G.edges()):
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return

def alias_setup(probs):
    """
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    """
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)
    smaller = []
    larger = []

    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q

def alias_draw(J, q):
    """
    Draw sample from a non-uniform discrete distribution using alias sampling.
    """
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]
