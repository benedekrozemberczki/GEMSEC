import networkx as nx
import numpy as np
import random
import math
import community
from collections import Counter
from tqdm import tqdm
from sklearn.cluster import KMeans

def normalized_overlap(g, node_1, node_2):
    inter = len(set(nx.neighbors(g, node_1)).intersection(set(nx.neighbors(g, node_2))))
    unio = len(set(nx.neighbors(g, node_1)).union(set(nx.neighbors(g, node_2))))
    return float(inter)/float(unio)

def overlap(g, node_1, node_2):
    inter = len(set(nx.neighbors(g, node_1)).intersection(set(nx.neighbors(g, node_2))))
    return float(inter)

def unit(g, node_1, node_2):
    return 1

def min_norm(g, node_1, node_2):
    inter = len(set(nx.neighbors(g, node_1)).intersection(set(nx.neighbors(g, node_2))))
    min_norm = min(len(set(nx.neighbors(g, node_1))), len(set(nx.neighbors(g, node_2))))
    return float(inter)/float(min_norm)

def overlap_generator(metric, graph):
    edges = nx.edges(graph)
    weights = {edge: metric(graph, edge[0], edge[1]) for edge in tqdm(edges)}
    weights_prime = {(edge[1],edge[0]): value for edge, value in tqdm(weights.iteritems())}
    weights.update(weights_prime)
    return weights

def index_generation(weights, a_random_walk):
    edges = [(a_random_walk[i], a_random_walk[i+1]) for i in range(0,len(a_random_walk)-1)]
    edge_set_1 = np.array(range(0,len(a_random_walk)-1))
    edge_set_2 = np.array(range(1,len(a_random_walk)))
    overlaps = np.array(map(lambda x: weights[x] , edges)).reshape((-1,1))
    return edge_set_1, edge_set_2, overlaps

def batch_input_generator(a_random_walk, random_walk_length, window_size):

    seq_1 = [a_random_walk[j] for j in range(random_walk_length-window_size)]
    seq_2 = [a_random_walk[j] for j in range(window_size,random_walk_length)]

    return np.array(seq_1 + seq_2)

def batch_label_generator(a_random_walk, random_walk_length, window_size):
    grams_1 = [a_random_walk[j+1:j+1+window_size] for j in range(random_walk_length-window_size)]
    grams_2 = [a_random_walk[j-window_size:j] for j in range(window_size,random_walk_length)]

        
    return np.array(grams_1 + grams_2)

def gamma_incrementer(step, gamma_0, current_gamma, num_steps):
    if step >1:
        exponent = (0-np.log10(gamma_0))/float(num_steps)
        current_gamma = current_gamma * (10 **exponent)
    return current_gamma


def neural_modularity_calculator(graph, embedding, means):
    assignments = {}
    for node in graph.nodes():
        positions = means-embedding[node,:]
        values = np.sum(np.square(positions),axis=1)
        index = np.argmin(values)
        assignments[node]= index
    modularity = community.modularity(assignments,graph)
    return modularity, assignments

def classical_modularity_calculator(graph, embedding, args):
    kmeans = KMeans(n_clusters=args.cluster_number, random_state=0, n_init = 1).fit(embedding)
    assignments = {i: int(kmeans.labels_[i]) for i in range(0, embedding.shape[0])}
    modularity = community.modularity(assignments,graph)
    return modularity, assignments

class RandomWalker:
    
    def __init__(self, graph, nodes, repetitions, length):
        print("Model initialization started.")
        self.graph = graph
        self.nodes = nodes
        self.repetitions = repetitions 
        self.length = length

    def small_walk(self, start_node):
        walk = [start_node]
        while len(walk) != self.length:
            end_point = walk[-1]
            neighbors = nx.neighbors(self.graph, end_point)
            if len(neighbors) > 0:
                walk = walk + random.sample(neighbors, 1)
            else:
                break

        return walk

    def count_frequency_values(self):
        raw_counts = [node for walk in self.walks for node in walk]
        counts = Counter(raw_counts)
        self.degrees = [counts[i] for i in range(0,len(self.nodes))]
       
    def do_walks(self):
        self.walks = []
        for rep in range(0,self.repetitions):
            random.shuffle(self.nodes)
            print(" ")
            print("Random walk series " + str(rep+1) + ". initiated.")
            print(" ")
            for node in tqdm(self.nodes):
                walk = self.small_walk(node)
                self.walks.append(walk)
        self.count_frequency_values()
        return self.degrees, self.walks

