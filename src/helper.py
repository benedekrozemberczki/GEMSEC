import argparse
import pandas as pd
import networkx as nx
import numpy as np
import random
import json
import math

def parameter_parser():

    """
    A method to parse up command line parameters. By default it gives an embedding of the Facebook politicians network.
    The default hyperparameters give a good quality representation and good candidate cluster means without grid search.
    """

    parser = argparse.ArgumentParser(description = "Run GEMSEC.")

    #------------------------------------------------------------------
    #
    # Input and output file parameters.
    # 
    #------------------------------------------------------------------

    parser.add_argument('--input',
                        nargs = '?',
                        default = './data/politician_edges.csv',
	                help = 'Input graph path.')

    parser.add_argument('--embedding-output',
                        nargs = '?',
                        default = './output/embeddings/politician_embedding.csv',
	                help = 'Embeddings path.')

    parser.add_argument('--cluster-mean-output',
                        nargs = '?',
                        default = './output/cluster_means/politician_means.csv',
	                help = 'Cluster means path.')

    parser.add_argument('--log-output',
                        nargs = '?',
                        default = './output/logs/politician.log',
	                help = 'Log path.')

    parser.add_argument('--dump-matrices',
                        type = bool,
                        default = True,
	                help = 'Save the embeddings to disk or not. Default is not.')

    parser.add_argument('--model',
                        nargs = '?',
                        default = 'GEMSECWithRegularization',
	                help = 'The model type.')

    #------------------------------------------------------------------
    #
    # Skipgram parameters.
    # 
    #------------------------------------------------------------------

    parser.add_argument('--dimensions',
                        type = int,
                        default = 16,
	                help = 'Number of dimensions. Default is 16.')

    parser.add_argument('--random-walk-length',
                        type = int,
                        default = 80,
	                help = 'Length of random walk per source. Default is 40.')

    parser.add_argument('--num-of-walks',
                        type = int,
                        default = 5,
	                help = 'Number of random walks per source. Default is 5.')

    parser.add_argument('--window-size',
                        type = int,
                        default = 5,
                    	help = 'Window size for proximity statistic extraction. Default is 5.')

    parser.add_argument('--distortion',
                        type = float,
                        default = 0.75,
	                help = 'Downsampling distortion. Default is 0.75.')

    parser.add_argument('--negative-sample-number',
                        type = int,
                        default = 10,
	                help = 'Number of negative samples to draw. Default is 10.')

    #------------------------------------------------------------------
    #
    # Optimization parameters.
    # 
    #------------------------------------------------------------------

    parser.add_argument('--initial-learning-rate',
                        type = float,
                        default = 0.001,
	                help = 'Initial learning rate. Default is 0.001.')

    parser.add_argument('--minimal-learning-rate',
                        type = float,
                        default = 0.0001,
	                help = 'Minimal learning rate. Default is 0.0001.')

    parser.add_argument('--annealing-step-size',
                        type = int,
                        default = 1000,
	                help = 'Annealing step size. Default is 1000.')

    parser.add_argument('--annealing-factor',
                        type = float,
                        default = 0.99,
	                help = 'Annealing factor. Default is 0.999.')

    parser.add_argument('--initial-gamma',
                        type = float,
                        default = 0.1,
	                help = 'Initial clustering weight. Default is 0.1.')

    parser.add_argument('--lambd',
                        type = float,
                        default = 2.0**-4,
	                help = 'Smoothness regularization penalty. Default is 1.0.')

    parser.add_argument('--cluster-number',
                        type = int,
                        default = 20,
	                help = 'Number of clusters. Default is 5.')

    parser.add_argument('--overlap-weighting',
                        nargs = '?',
                        default = 'normalized_overlap',
	                help = 'Weight construction technique for regularization.')

    parser.add_argument('--regularization-noise',
                        type = float,
                        default = 10**-8,
	                help = 'Uniform noise max and min on the feature vector distance.')

    parser.add_argument('--regularization-norm',
                        nargs = '?',
                        default = 'euclidean',
	                help = 'The distance metric used for calculating the smoothness regularizer.')

    parser.add_argument('--clustering-norm',
                        nargs = '?',
                        default = 'euclidean',
	                help = 'The distance metric used for calculating the cluster distances.')
    
    return parser.parse_args()

def graph_reader(input_path):
    edges = pd.read_csv(input_path)
    graph = nx.from_edgelist(edges.values.tolist())
    return graph


def normalized_overlap(g, node_1, node_2):
    inter = len(set(nx.neighbors(g, node_1)).intersection(set(nx.neighbors(g, node_2))))
    unio = len(set(nx.neighbors(g, node_1)).union(set(nx.neighbors(g, node_2))))
    return float(inter)/float(unio)

def overlap(g, node_1, node_2):
    inter = len(set(nx.neighbors(g, node_1)).intersection(set(nx.neighbors(g, node_2))))
    return float(inter)

def unit(g, node_1, node_2):
    return 1

def preferential(g, node_1, node_2):
    return len(set(nx.neighbors(g, node_1))) * len(set(nx.neighbors(g, node_2)))

def min_norm(g, node_1, node_2):
    inter = len(set(nx.neighbors(g, node_1)).intersection(set(nx.neighbors(g, node_2))))
    min_norm = min(len(set(nx.neighbors(g, node_1))), len(set(nx.neighbors(g, node_2))))
    return float(inter)/float(min_norm)



def small_walk(G, start_node, walk_length):
    walk = [start_node]
    while len(walk) != walk_length:
        end_point = walk[-1]
        neighbors = nx.neighbors(G, end_point)
        if len(neighbors) > 0:
            walk = walk + random.sample(neighbors, 1)
        else:
            break

    return walk

def gamma_incrementer(step, gamma_0, current_gamma, num_steps):
    if step >1:
        exponent = (0-np.log10(gamma_0))/float(num_steps)
        current_gamma = current_gamma * (10 **exponent)
    return current_gamma



def json_dumper(data, path):
    with open(path, 'w') as outfile:
        json.dump(data, outfile)

def step_calculator(time_steps, annealing_rate, end_point, lr):
    solution = int(-(time_steps*math.log(annealing_rate))/(-math.log(end_point/lr)))
    if solution <1:
        solution = 1
    return solution


