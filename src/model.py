import tensorflow as tf
import math
import random
import numpy as np
from helper import gamma_incrementer,  small_walk,  json_dumper, normalized_overlap, overlap, unit, min_norm, preferential, step_calculator
from sklearn.cluster import KMeans
import community
import time
import pandas as pd
import networkx as nx
from texttable import Texttable

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass

class GEMSECWithRegularization(Model):
    def __init__(self, args, graph, **kwargs):
        super(GEMSECWithRegularization, self).__init__(**kwargs)

        self.args = args
        self.graph = graph
        self.degrees = nx.degree(self.graph).values()
        self.vocab_size = len(self.degrees)
        self.true_step_size = step_calculator(self.args.num_of_walks*self.vocab_size, self.args.annealing_factor, self.args.minimal_learning_rate, self.args.initial_learning_rate)
        print(self.true_step_size)
        self.build()

    def _build(self):
        self.computation_graph = tf.Graph()
        with self.computation_graph.as_default():
            self.train_labels = tf.placeholder(tf.int64, shape=[None, self.args.window_size])
            self.train_labels_flat = tf.reshape(self.train_labels, [-1, 1])
    
            self.edge_indices_right = tf.placeholder(tf.int64, shape=[None])
            self.edge_indices_left = tf.placeholder(tf.int64, shape=[None])
            self.train_inputs = tf.placeholder(tf.int64, shape=[None])
    
            self.overlap = tf.placeholder(tf.float32, shape=[None, 1])
    
            self.input_ones = tf.ones_like(self.train_labels)
            self.train_inputs_flat = tf.reshape(tf.multiply(self.input_ones, tf.reshape(self.train_inputs,[-1,1])),[-1])
    
            self.embedding_matrix = tf.Variable(tf.random_uniform([self.vocab_size, self.args.dimensions],
                                                -0.1/self.args.dimensions, 0.1/self.args.dimensions), name = "embed_down")
    
            self.embedding_partial = tf.nn.embedding_lookup(self.embedding_matrix, self.train_inputs_flat, max_norm = 1)
    
    
            self.nce_weights = tf.Variable(tf.truncated_normal([self.vocab_size, self.args.dimensions],
                                           stddev=1.0 / math.sqrt(self.args.dimensions)))
    
            self.nce_biases = tf.Variable(tf.random_uniform([self.vocab_size], -0.1/self.args.dimensions, 0.1/self.args.dimensions))
        
            self.sampler = tf.nn.fixed_unigram_candidate_sampler(true_classes = self.train_labels_flat,
                                                                 num_true = 1,
                                                                 num_sampled = self.args.negative_sample_number,
                                                                 unique = True,
                                                                 range_max = self.vocab_size,
                                                                 distortion = self.args.distortion,
                                                                 unigrams = self.degrees)
    
            self.embedding_losses = tf.nn.sampled_softmax_loss(weights = self.nce_weights,
                                                               biases = self.nce_biases,
                                                               labels = self.train_labels_flat,
                                                               inputs = self.embedding_partial,
                                                               num_true = 1,
                                                               num_sampled = self.args.negative_sample_number,
                                                               num_classes = self.vocab_size,
                                                               sampled_values = self.sampler)
    
            self.embedding_loss = tf.reduce_mean(self.embedding_losses)
    
            #-------------------------------------------------
            #
            #-------------------------------------------------
    
    
            self.cluster_means = tf.Variable(tf.random_uniform([self.args.cluster_number, self.args.dimensions],
                                            -0.1/self.args.dimensions, 0.1/self.args.dimensions))
           
            self.clustering_differences = tf.expand_dims(self.embedding_partial,1) - self.cluster_means

            if self.args.clustering_norm == "infinity":      
                self.cluster_distances = tf.norm(self.clustering_differences, ord = np.inf, axis = 2)
            elif self.args.clustering_norm == "manhattan":
                self.cluster_distances = tf.norm(self.clustering_differences, ord = 1, axis = 2)
            elif self.args.clustering_norm == "euclidean":
                self.cluster_distances = tf.norm(self.clustering_differences, ord = 2, axis = 2)

            self.to_be_averaged = tf.reduce_min(self.cluster_distances, axis = 1)
            self.clustering_loss = tf.reduce_mean(self.to_be_averaged)
    
            #-------------------------------------------------
            #
            #-------------------------------------------------
    
            self.left_features = tf.nn.embedding_lookup(self.embedding_matrix, self.edge_indices_left, max_norm = 1)
            self.right_features = tf.nn.embedding_lookup(self.embedding_matrix, self.edge_indices_right, max_norm = 1)
            
            self.regularization_differences = self.left_features - self.right_features + np.random.uniform(self.args.regularization_noise,self.args.regularization_noise, (self.args.random_walk_length-1, self.args.dimensions))

            if self.args.regularization_norm == "infinity":  
                self.regularization_distances = tf.norm(self.regularization_differences,ord = np.inf, axis=1)#/self.args.dimensions
            elif self.args.regularization_norm == "manhattan":  
                self.regularization_distances = tf.norm(self.regularization_differences,ord = 1,axis=1)#/self.args.dimensions
            elif self.args.regularization_norm == "euclidean":  
                self.regularization_distances = tf.norm(self.regularization_differences,ord = 2,axis=1)#/self.args.dimensions

            self.regularization_distances = tf.reshape(self.regularization_distances, [ -1, 1])
            self.regularization_loss = tf.reduce_mean(tf.matmul(tf.transpose(self.overlap), self.regularization_distances))
    
            #-------------------------------------------------
            #
            #-------------------------------------------------

            self.learning_rate = tf.placeholder("float")
            self.gamma = tf.placeholder("float")
            self.loss = self.embedding_loss + self.gamma * self.clustering_loss + self.args.lambd*self.regularization_loss

    
            self.step = tf.placeholder("float")
            self.learning_rate_base = tf.train.exponential_decay(self.learning_rate,
                                                                 self.step,
                                                                 self.true_step_size,
                                                                 self.args.annealing_factor,
                                                                 staircase=True)
    
            self.learning_rate_new = tf.maximum(self.learning_rate_base, self.args.minimal_learning_rate)
    
            self.train_op = tf.train.AdamOptimizer(self.learning_rate_new).minimize(self.loss)
    
            self.init = tf.global_variables_initializer()

        if self.args.overlap_weighting == "normalized_overlap":
            self.overlap_weighter = normalized_overlap
        elif self.args.overlap_weighting == "overlap":
            self.overlap_weighter = overlap
        elif self.args.overlap_weighting == "min_norm":
            self.overlap_weighter = min_norm
        elif self.args.overlap_weighting == "preferential":
            self.overlap_weighter = preferential
        else:
            self.overlap_weighter = unit
        self.check = tf.add_check_numerics_ops()

    def index_generation(self, a_random_walk):
        #small_g = self.graph.subgraph(set(list(a_random_walk)))
        edges = [(a_random_walk[i], a_random_walk[i+1])for i in range(0,len(a_random_walk)-1)]
        edge_set_1 = np.array(map(lambda x: x[0], edges))
        edge_set_2 = np.array(map(lambda x: x[1], edges))
        overlaps = np.array(map(lambda x: self.overlap_weighter(self.graph, x[0], x[1]), edges)).reshape((-1,1))
        return edge_set_1, edge_set_2, overlaps
        


    def feed_dict_generator(self, node, step, learning_rate, gamma):
        start = time.time()
        a_random_walk = small_walk(self.graph, node, self.args.random_walk_length)

        index_1, index_2, overlaps = self.index_generation(a_random_walk)
        seq_1 = [a_random_walk[j] for j in xrange(self.args.random_walk_length-self.args.window_size)]
        seq_2 = [a_random_walk[j] for j in range(self.args.window_size,self.args.random_walk_length)]
        batch_inputs = np.array(seq_1 + seq_2)

        
        grams_1 = [a_random_walk[j+1:j+1+self.args.window_size] for j in xrange(self.args.random_walk_length-self.args.window_size)]
        grams_2 = [a_random_walk[j-self.args.window_size:j] for j in range(self.args.window_size,self.args.random_walk_length)]

        
        batch_labels = np.array(grams_1 + grams_2)


        feed_dict = {self.train_inputs: batch_inputs,
                     self.train_labels: batch_labels,
                     self.learning_rate: learning_rate,
                     self.gamma: gamma,
                     self.step: float(step),
                     self.edge_indices_left: index_1,
                     self.edge_indices_right: index_2,
                     self.overlap: overlaps}
        end = time.time()
        
        data_generation_time = end-start
        return feed_dict, data_generation_time 



    
    def train(self):
        
        self.nodes = self.graph.nodes()
        self.current_step = 1
        self.num_steps = self.args.num_of_walks * len(self.nodes)
        self.current_gamma = self.args.initial_gamma        
        self.log = dict()
        self.log["times"] = []
        self.log["losses"] = []
        self.log["cluster_quality"] = []
        with tf.Session(graph=self.computation_graph) as session:
            self.init.run()
            print('Initialized')
            for repetition in range(0, self.args.num_of_walks):
                random.shuffle(self.nodes)
                
                generation_time = 0
                optimization_time = 0 
                
                self.average_embedding_loss = 0
                self.average_clustering_loss = 0
                self.average_regularization_loss = 0
                
                for node in self.nodes:
                    self.current_step = self.current_step + 1
                    self.current_gamma = gamma_incrementer(self.current_step, self.args.initial_gamma,self.current_gamma, self.num_steps)
                    feed_dict, data_generation_time  = self.feed_dict_generator(node, self.current_step,self.args.initial_learning_rate, self.current_gamma)
                    generation_time = generation_time + data_generation_time
                    start = time.time()
                    _, loss_val, second, third = session.run([self.train_op , self.embedding_loss, self.clustering_loss, self.regularization_loss], feed_dict=feed_dict)
                    end = time.time()
                    optimization_time = optimization_time + (end-start)
                    self.average_embedding_loss += loss_val
                    self.average_clustering_loss += second
                    self.average_regularization_loss += third


                self.average_embedding_loss /= self.vocab_size
                self.average_clustering_loss /= self.vocab_size
                self.average_regularization_loss /= self.vocab_size

                self.final_embeddings = self.embedding_matrix.eval()
                self.c_means = self.cluster_means.eval()
                assignments = {}
                for node in self.graph.nodes():

                    positions = self.c_means-self.final_embeddings[node,:]
                    values = np.sum(np.square(positions),axis=1)
                    index = np.argmin(values)
                    assignments[node]= index
                neural = community.modularity(assignments,self.graph)
                kmeans = KMeans(n_clusters=self.args.cluster_number, random_state=0, n_init = 1).fit(self.final_embeddings)
                labs = {i:kmeans.labels_[i] for i in range(0, self.vocab_size)}
                in_embedding_space = community.modularity(labs,self.graph)

                self.log["losses"] = self.log["losses"] + [[repetition + 1, self.average_embedding_loss, self.average_clustering_loss, self.average_regularization_loss]]
                self.log["times"] = self.log["times"] + [[repetition + 1, generation_time,optimization_time]]
                self.log["cluster_quality"] = self.log["cluster_quality"] + [[repetition+ 1, neural,in_embedding_space]]
                self.log["params"] = vars(self.args)
                self.tab_printer()

        
    def initiate_dump(self):
        json_dumper(self.log, self.args.log_output)
        self.c_means = pd.DataFrame(self.c_means)
        self.final_embeddings = pd.DataFrame(self.final_embeddings)
        if self.args.dump_matrices:
            self.c_means.to_csv(self.args.cluster_mean_output, index = None)
            self.final_embeddings.to_csv(self.args.embedding_output, index = None)

    def tab_printer(self):
        t = Texttable() 
        t.add_rows([['Epoch', self.log["losses"][-1][0]]])
        print t.draw()

        t = Texttable()
        t.add_rows([['Loss type', 'Loss value'], ['Embedding', self.log["losses"][-1][1]], ['Clustering', self.log["losses"][-1][2]],['Regularization', self.log["losses"][-1][3]]])
        print t.draw() 

        t = Texttable()
        t.add_rows([['Clustering Method', 'Modularity'], ['Neural K-means', self.log["cluster_quality"][-1][1]], ['Classical K-means', self.log["cluster_quality"][-1][2]]])
        print t.draw()    


class GEMSEC(Model):
    def __init__(self, args, graph, **kwargs):
        super(GEMSEC, self).__init__(**kwargs)

        self.args = args
        self.graph = graph
        self.degrees = nx.degree(self.graph).values()
        self.vocab_size = len(self.degrees)
        self.true_step_size = step_calculator(self.args.num_of_walks*self.vocab_size, self.args.annealing_factor, self.args.minimal_learning_rate, self.args.initial_learning_rate)
        self.build()

    def _build(self):
        self.computation_graph = tf.Graph()
        with self.computation_graph.as_default():
            self.train_labels = tf.placeholder(tf.int64, shape=[None, self.args.window_size])
            self.train_labels_flat = tf.reshape(self.train_labels, [-1, 1])
    
            self.train_inputs = tf.placeholder(tf.int64, shape=[None])
    
    
            self.input_ones = tf.ones_like(self.train_labels)
            self.train_inputs_flat = tf.reshape(tf.multiply(self.input_ones, tf.reshape(self.train_inputs,[-1,1])),[-1])
    
            self.embedding_matrix = tf.Variable(tf.random_uniform([self.vocab_size, self.args.dimensions],
                                                -0.1/self.args.dimensions, 0.1/self.args.dimensions), name = "embed_down")
    
            self.embedding_partial = tf.nn.embedding_lookup(self.embedding_matrix, self.train_inputs_flat, max_norm = 1)
    
    
            self.nce_weights = tf.Variable(tf.truncated_normal([self.vocab_size, self.args.dimensions],
                                           stddev=1.0 / math.sqrt(self.args.dimensions)))
    
            self.nce_biases = tf.Variable(tf.random_uniform([self.vocab_size], -0.1/self.args.dimensions, 0.1/self.args.dimensions))
        
            self.sampler = tf.nn.fixed_unigram_candidate_sampler(true_classes = self.train_labels_flat,
                                                                 num_true = 1,
                                                                 num_sampled = self.args.negative_sample_number,
                                                                 unique = True,
                                                                 range_max = self.vocab_size,
                                                                 distortion = self.args.distortion,
                                                                 unigrams = self.degrees)
    
            self.embedding_losses = tf.nn.sampled_softmax_loss(weights = self.nce_weights,
                                                               biases = self.nce_biases,
                                                               labels = self.train_labels_flat,
                                                               inputs = self.embedding_partial,
                                                               num_true = 1,
                                                               num_sampled = self.args.negative_sample_number,
                                                               num_classes = self.vocab_size,
                                                               sampled_values = self.sampler)
    
            self.embedding_loss = tf.reduce_mean(self.embedding_losses)
    
            #-------------------------------------------------
            #
            #-------------------------------------------------
            self.cluster_means = tf.Variable(tf.random_uniform([self.args.cluster_number, self.args.dimensions],
                                            -0.1/self.args.dimensions, 0.1/self.args.dimensions))
           
            self.clustering_differences = tf.expand_dims(self.embedding_partial,1) - self.cluster_means

            if self.args.clustering_norm == "infinity":      
                self.cluster_distances = tf.norm(self.clustering_differences, ord = np.inf, axis = 2)
            elif self.args.clustering_norm == "manhattan":
                self.cluster_distances = tf.norm(self.clustering_differences, ord = 1, axis = 2)
            elif self.args.clustering_norm == "euclidean":
                self.cluster_distances = tf.norm(self.clustering_differences, ord = 2, axis = 2)

            self.to_be_averaged = tf.reduce_min(self.cluster_distances, axis = 1)
            self.clustering_loss = tf.reduce_mean(self.to_be_averaged)
    
            #-------------------------------------------------
            #
            #-------------------------------------------------
    
            self.learning_rate = tf.placeholder("float")
            self.gamma = tf.placeholder("float")
            self.loss = self.embedding_loss + self.gamma * self.clustering_loss

    
            self.step = tf.placeholder("float")
            self.learning_rate_base = tf.train.exponential_decay(self.learning_rate,
                                                                 self.true_step_size,
                                                                 self.args.annealing_step_size,
                                                                 self.args.annealing_factor,
                                                                 staircase=True)
    
            self.learning_rate_new = tf.maximum(self.learning_rate_base, self.args.minimal_learning_rate)
    
            self.train_op = tf.train.AdamOptimizer(self.learning_rate_new).minimize(self.loss)
    
            self.init = tf.global_variables_initializer()



    def feed_dict_generator(self, node, step, learning_rate, gamma):
        start = time.time()
        a_random_walk = small_walk(self.graph, node, self.args.random_walk_length)

        seq_1 = [a_random_walk[j] for j in xrange(self.args.random_walk_length-self.args.window_size)]
        seq_2 = [a_random_walk[j] for j in range(self.args.window_size,self.args.random_walk_length)]
        batch_inputs = np.array(seq_1 + seq_2)

        
        grams_1 = [a_random_walk[j+1:j+1+self.args.window_size] for j in xrange(self.args.random_walk_length-self.args.window_size)]
        grams_2 = [a_random_walk[j-self.args.window_size:j] for j in range(self.args.window_size,self.args.random_walk_length)]

        
        batch_labels = np.array(grams_1 + grams_2)


        feed_dict = {self.train_inputs: batch_inputs,
                     self.train_labels: batch_labels,
                     self.learning_rate: learning_rate,
                     self.gamma: gamma,
                     self.step: float(step)}
        end = time.time()
        
        data_generation_time = end-start
        return feed_dict, data_generation_time 



    
    def train(self):
        
        self.nodes = self.graph.nodes()
        self.current_step = 1
        self.num_steps = self.args.num_of_walks * len(self.nodes)
        self.current_gamma = self.args.initial_gamma        
        self.log = dict()
        self.log["times"] = []
        self.log["losses"] = []
        self.log["cluster_quality"] = []
        with tf.Session(graph=self.computation_graph) as session:
            self.init.run()
            print('Initialized')
            for repetition in range(0, self.args.num_of_walks):
                random.shuffle(self.nodes)
                
                generation_time = 0
                optimization_time = 0 
                
                self.average_embedding_loss = 0
                self.average_clustering_loss = 0
                
                for node in self.nodes:
                    self.current_step = self.current_step + 1
                    self.current_gamma = gamma_incrementer(self.current_step, self.args.initial_gamma,self.current_gamma, self.num_steps)
                    feed_dict, data_generation_time  = self.feed_dict_generator(node, self.current_step,self.args.initial_learning_rate, self.current_gamma)
                    generation_time = generation_time + data_generation_time
                    start = time.time()
                    _, loss_val, second= session.run([self.train_op , self.embedding_loss, self.clustering_loss], feed_dict=feed_dict)
                    end = time.time()
                    optimization_time = optimization_time + (end-start)
                    self.average_embedding_loss += loss_val
                    self.average_clustering_loss += second


                self.average_embedding_loss /= self.vocab_size
                self.average_clustering_loss /= self.vocab_size

                self.final_embeddings = self.embedding_matrix.eval()
                self.c_means = self.cluster_means.eval()
                assignments = {}
                for node in self.graph.nodes():

                    positions = self.c_means-self.final_embeddings[node,:]
                    values = np.sum(np.square(positions),axis=1)
                    index = np.argmin(values)
                    assignments[node]= index
                neural = community.modularity(assignments,self.graph)
                kmeans = KMeans(n_clusters=self.args.cluster_number, random_state=0, n_init = 1).fit(self.final_embeddings)
                labs = {i:kmeans.labels_[i] for i in range(0, self.vocab_size)}
                in_embedding_space = community.modularity(labs,self.graph)

                self.log["losses"] = self.log["losses"] + [[repetition + 1, self.average_embedding_loss, self.average_clustering_loss]]
                self.log["times"] = self.log["times"] + [[repetition + 1, generation_time,optimization_time]]
                self.log["cluster_quality"] = self.log["cluster_quality"] + [[repetition+ 1, neural,in_embedding_space]]
                self.log["params"] = vars(self.args)
                self.tab_printer()

        
    def initiate_dump(self):
        print(self.log)
        json_dumper(self.log, self.args.log_output)
        self.c_means = pd.DataFrame(self.c_means)
        self.final_embeddings = pd.DataFrame(self.final_embeddings)
        if self.args.dump_matrices:
            self.c_means.to_csv(self.args.cluster_mean_output, index = None)
            self.final_embeddings.to_csv(self.args.embedding_output, index = None)

    def tab_printer(self):
        t = Texttable() 
        t.add_rows([['Epoch', self.log["losses"][-1][0]]])
        print t.draw()

        t = Texttable()
        t.add_rows([['Loss type', 'Loss value'], ['Embedding', self.log["losses"][-1][1]], ['Clustering', self.log["losses"][-1][2]]])
        print t.draw() 

        t = Texttable()
        t.add_rows([['Clustering Method', 'Modularity'], ['Neural K-means', self.log["cluster_quality"][-1][1]], ['Classical K-means', self.log["cluster_quality"][-1][2]]])
        print t.draw()




class DWWithRegularization(Model):
    def __init__(self, args, graph, **kwargs):
        super(DWWithRegularization, self).__init__(**kwargs)

        self.args = args
        self.graph = graph
        self.degrees = nx.degree(self.graph).values()
        self.vocab_size = len(self.degrees)
        self.true_step_size = step_calculator(self.args.num_of_walks*self.vocab_size, self.args.annealing_factor, self.args.minimal_learning_rate, self.args.initial_learning_rate) 
        self.build()

    def _build(self):
        self.computation_graph = tf.Graph()
        with self.computation_graph.as_default():
            self.train_labels = tf.placeholder(tf.int64, shape=[None, self.args.window_size])
            self.train_labels_flat = tf.reshape(self.train_labels, [-1, 1])
    
            self.edge_indices_right = tf.placeholder(tf.int64, shape=[None])
            self.edge_indices_left = tf.placeholder(tf.int64, shape=[None])
            self.train_inputs = tf.placeholder(tf.int64, shape=[None])
    
            self.overlap = tf.placeholder(tf.float32, shape=[None, 1])
    
            self.input_ones = tf.ones_like(self.train_labels)
            self.train_inputs_flat = tf.reshape(tf.multiply(self.input_ones, tf.reshape(self.train_inputs,[-1,1])),[-1])
    
            self.embedding_matrix = tf.Variable(tf.random_uniform([self.vocab_size, self.args.dimensions],
                                                -0.1/self.args.dimensions, 0.1/self.args.dimensions), name = "embed_down")
    
            self.embedding_partial = tf.nn.embedding_lookup(self.embedding_matrix, self.train_inputs_flat, max_norm = 1)
    
    
            self.nce_weights = tf.Variable(tf.truncated_normal([self.vocab_size, self.args.dimensions],
                                           stddev=1.0 / math.sqrt(self.args.dimensions)))
    
            self.nce_biases = tf.Variable(tf.random_uniform([self.vocab_size], -0.1/self.args.dimensions, 0.1/self.args.dimensions))
        
            self.sampler = tf.nn.fixed_unigram_candidate_sampler(true_classes = self.train_labels_flat,
                                                                 num_true = 1,
                                                                 num_sampled = self.args.negative_sample_number,
                                                                 unique = True,
                                                                 range_max = self.vocab_size,
                                                                 distortion = self.args.distortion,
                                                                 unigrams = self.degrees)
    
            self.embedding_losses = tf.nn.sampled_softmax_loss(weights = self.nce_weights,
                                                               biases = self.nce_biases,
                                                               labels = self.train_labels_flat,
                                                               inputs = self.embedding_partial,
                                                               num_true = 1,
                                                               num_sampled = self.args.negative_sample_number,
                                                               num_classes = self.vocab_size,
                                                               sampled_values = self.sampler)
    
            self.embedding_loss = tf.reduce_mean(self.embedding_losses)
    
            #-------------------------------------------------
            #
            #-------------------------------------------------
    
            self.left_features = tf.nn.embedding_lookup(self.embedding_matrix, self.edge_indices_left, max_norm = 1)
            self.right_features = tf.nn.embedding_lookup(self.embedding_matrix, self.edge_indices_right, max_norm = 1)
            
            self.regularization_differences = self.left_features - self.right_features + np.random.uniform(self.args.regularization_noise,self.args.regularization_noise, (self.args.random_walk_length-1, self.args.dimensions))

            if self.args.regularization_norm == "infinity":  
                self.regularization_distances = tf.norm(self.regularization_differences,ord = np.inf, axis=1)
            elif self.args.regularization_norm == "manhattan":  
                self.regularization_distances = tf.norm(self.regularization_differences,ord = 1,axis=1)
            elif self.args.regularization_norm == "euclidean":  
                self.regularization_distances = tf.norm(self.regularization_differences,ord = 2,axis=1)

            self.regularization_distances = tf.reshape(self.regularization_distances, [ -1, 1])
            self.regularization_loss = tf.reduce_mean(tf.matmul(tf.transpose(self.overlap), self.regularization_distances))
    


            self.learning_rate = tf.placeholder("float")
            self.loss = self.embedding_loss  + self.args.lambd * self.regularization_loss

    
            self.step = tf.placeholder("float")
            self.learning_rate_base = tf.train.exponential_decay(self.learning_rate,
                                                                 self.true_step_size,
                                                                 self.args.annealing_step_size,
                                                                 self.args.annealing_factor,
                                                                 staircase=True)
    
            self.learning_rate_new = tf.maximum(self.learning_rate_base, self.args.minimal_learning_rate)
    
            self.train_op = tf.train.AdamOptimizer(self.learning_rate_new).minimize(self.loss)
    
            self.init = tf.global_variables_initializer()

        if self.args.overlap_weighting == "normalized_overlap":
            self.overlap_weighter = normalized_overlap
        elif self.args.overlap_weighting == "overlap":
            self.overlap_weighter = overlap
        elif self.args.overlap_weighting == "min_norm":
            self.overlap_weighter = min_norm
        elif self.args.overlap_weighting == "preferential":
            self.overlap_weighter = preferential
        else:
            self.overlap_weighter = unit

    def index_generation(self, a_random_walk):
        #small_g = self.graph.subgraph(set(list(a_random_walk)))
        edges = [(a_random_walk[i], a_random_walk[i+1])for i in range(0,len(a_random_walk)-1)]
        edge_set_1 = np.array(map(lambda x: x[0], edges))
        edge_set_2 = np.array(map(lambda x: x[1], edges))
        overlaps = np.array(map(lambda x: self.overlap_weighter(self.graph, x[0], x[1]), edges)).reshape((-1,1))
        return edge_set_1, edge_set_2, overlaps
        


    def feed_dict_generator(self, node, step, learning_rate, gamma):
        start = time.time()
        a_random_walk = small_walk(self.graph, node, self.args.random_walk_length)

        index_1, index_2, overlaps = self.index_generation(a_random_walk)
        seq_1 = [a_random_walk[j] for j in xrange(self.args.random_walk_length-self.args.window_size)]
        seq_2 = [a_random_walk[j] for j in range(self.args.window_size,self.args.random_walk_length)]
        batch_inputs = np.array(seq_1 + seq_2)

        
        grams_1 = [a_random_walk[j+1:j+1+self.args.window_size] for j in xrange(self.args.random_walk_length-self.args.window_size)]
        grams_2 = [a_random_walk[j-self.args.window_size:j] for j in range(self.args.window_size,self.args.random_walk_length)]

        
        batch_labels = np.array(grams_1 + grams_2)


        feed_dict = {self.train_inputs: batch_inputs,
                     self.train_labels: batch_labels,
                     self.learning_rate: learning_rate,
                     self.step: float(step),
                     self.edge_indices_left: index_1,
                     self.edge_indices_right: index_2,
                     self.overlap: overlaps}
        end = time.time()
        
        data_generation_time = end-start
        return feed_dict, data_generation_time 



    
    def train(self):
        
        self.nodes = self.graph.nodes()
        self.current_step = 1
        self.num_steps = self.args.num_of_walks * len(self.nodes)
        self.current_gamma = self.args.initial_gamma        
        self.log = dict()
        self.log["times"] = []
        self.log["losses"] = []
        self.log["cluster_quality"] = []
        with tf.Session(graph=self.computation_graph) as session:
            self.init.run()
            print('Initialized')
            for repetition in range(0, self.args.num_of_walks):
                random.shuffle(self.nodes)
                
                generation_time = 0
                optimization_time = 0 
                
                self.average_embedding_loss = 0
                self.average_regularization_loss = 0
                
                for node in self.nodes:
                    self.current_step = self.current_step + 1
                    self.current_gamma = gamma_incrementer(self.current_step, self.args.initial_gamma,self.current_gamma, self.num_steps)
                    feed_dict, data_generation_time  = self.feed_dict_generator(node, self.current_step,self.args.initial_learning_rate, self.current_gamma)
                    generation_time = generation_time + data_generation_time
                    start = time.time()
                    _, loss_val, second = session.run([self.train_op , self.embedding_loss, self.regularization_loss], feed_dict=feed_dict)
                    end = time.time()
                    optimization_time = optimization_time + (end-start)
                    self.average_embedding_loss += loss_val
                    self.average_regularization_loss += second


                self.average_embedding_loss /= self.vocab_size
                self.average_regularization_loss /= self.vocab_size

                self.final_embeddings = self.embedding_matrix.eval()
                kmeans = KMeans(n_clusters=self.args.cluster_number, random_state=0, n_init = 1).fit(self.final_embeddings)
                labs = {i:kmeans.labels_[i] for i in range(0, self.vocab_size)}
                in_embedding_space = community.modularity(labs,self.graph)

                self.log["losses"] = self.log["losses"] + [[repetition + 1, self.average_embedding_loss, self.average_regularization_loss]]
                self.log["times"] = self.log["times"] + [[repetition + 1, generation_time,optimization_time]]
                self.log["cluster_quality"] = self.log["cluster_quality"] + [[repetition+ 1, in_embedding_space]]
                self.log["params"] = vars(self.args)
                self.tab_printer()

        
    def initiate_dump(self):
        json_dumper(self.log, self.args.log_output)
        self.final_embeddings = pd.DataFrame(self.final_embeddings)
        if self.args.dump_matrices:
            self.final_embeddings.to_csv(self.args.embedding_output, index = None)

    def tab_printer(self):
        t = Texttable() 
        t.add_rows([['Epoch', self.log["losses"][-1][0]]])
        print t.draw()

        t = Texttable()
        t.add_rows([['Loss type', 'Loss value'], ['Embedding', self.log["losses"][-1][1]], ['Regularization', self.log["losses"][-1][2]]])
        print t.draw() 

        t = Texttable()
        t.add_rows([['Clustering Method', 'Modularity'],  ['Classical K-means', self.log["cluster_quality"][-1][1]]])
        print t.draw()   





class DW(Model):
    def __init__(self, args, graph, **kwargs):
        super(DW, self).__init__(**kwargs)

        self.args = args
        self.graph = graph
        self.degrees = nx.degree(self.graph).values()
        self.vocab_size = len(self.degrees)
        self.true_step_size = step_calculator(self.args.num_of_walks*self.vocab_size, self.args.annealing_factor, self.args.minimal_learning_rate, self.args.initial_learning_rate)
        self.build()

    def _build(self):
        self.computation_graph = tf.Graph()
        with self.computation_graph.as_default():
            self.train_labels = tf.placeholder(tf.int64, shape=[None, self.args.window_size])
            self.train_labels_flat = tf.reshape(self.train_labels, [-1, 1])
            self.train_inputs = tf.placeholder(tf.int64, shape=[None])

            self.input_ones = tf.ones_like(self.train_labels)
            self.train_inputs_flat = tf.reshape(tf.multiply(self.input_ones, tf.reshape(self.train_inputs,[-1,1])),[-1])
    
            self.embedding_matrix = tf.Variable(tf.random_uniform([self.vocab_size, self.args.dimensions],
                                                -0.1/self.args.dimensions, 0.1/self.args.dimensions), name = "embed_down")
    
            self.embedding_partial = tf.nn.embedding_lookup(self.embedding_matrix, self.train_inputs_flat, max_norm = 1)
    
    
            self.nce_weights = tf.Variable(tf.truncated_normal([self.vocab_size, self.args.dimensions],
                                           stddev=1.0 / math.sqrt(self.args.dimensions)))
    
            self.nce_biases = tf.Variable(tf.random_uniform([self.vocab_size], -0.1/self.args.dimensions, 0.1/self.args.dimensions))
        
            self.sampler = tf.nn.fixed_unigram_candidate_sampler(true_classes = self.train_labels_flat,
                                                                 num_true = 1,
                                                                 num_sampled = self.args.negative_sample_number,
                                                                 unique = True,
                                                                 range_max = self.vocab_size,
                                                                 distortion = self.args.distortion,
                                                                 unigrams = self.degrees)
    
            self.embedding_losses = tf.nn.sampled_softmax_loss(weights = self.nce_weights,
                                                               biases = self.nce_biases,
                                                               labels = self.train_labels_flat,
                                                               inputs = self.embedding_partial,
                                                               num_true = 1,
                                                               num_sampled = self.args.negative_sample_number,
                                                               num_classes = self.vocab_size,
                                                               sampled_values = self.sampler)
    
            self.embedding_loss = tf.reduce_mean(self.embedding_losses)
    
            #-------------------------------------------------
            #
            #-------------------------------------------------
    

            self.learning_rate = tf.placeholder("float")
            self.loss = self.embedding_loss

    
            self.step = tf.placeholder("float")
            self.learning_rate_base = tf.train.exponential_decay(self.learning_rate,
                                                                 self.true_step_size ,
                                                                 self.args.annealing_step_size,
                                                                 self.args.annealing_factor,
                                                                 staircase=True)
    
            self.learning_rate_new = tf.maximum(self.learning_rate_base, self.args.minimal_learning_rate)
    
            self.train_op = tf.train.AdamOptimizer(self.learning_rate_new).minimize(self.loss)
    
            self.init = tf.global_variables_initializer()


    def index_generation(self, a_random_walk):
        edges = [(a_random_walk[i], a_random_walk[i+1])for i in range(0,len(a_random_walk)-1)]
        edge_set_1 = np.array(map(lambda x: x[0], edges))
        edge_set_2 = np.array(map(lambda x: x[1], edges))
        overlaps = np.array(map(lambda x: self.overlap_weighter(self.graph, x[0], x[1]), edges)).reshape((-1,1))
        return edge_set_1, edge_set_2, overlaps
        


    def feed_dict_generator(self, node, step, learning_rate, gamma):
        start = time.time()
        a_random_walk = small_walk(self.graph, node, self.args.random_walk_length)

        seq_1 = [a_random_walk[j] for j in xrange(self.args.random_walk_length-self.args.window_size)]
        seq_2 = [a_random_walk[j] for j in range(self.args.window_size,self.args.random_walk_length)]
        batch_inputs = np.array(seq_1 + seq_2)

        
        grams_1 = [a_random_walk[j+1:j+1+self.args.window_size] for j in xrange(self.args.random_walk_length-self.args.window_size)]
        grams_2 = [a_random_walk[j-self.args.window_size:j] for j in range(self.args.window_size,self.args.random_walk_length)]

        
        batch_labels = np.array(grams_1 + grams_2)


        feed_dict = {self.train_inputs: batch_inputs,
                     self.train_labels: batch_labels,
                     self.learning_rate: learning_rate,
                     self.step: float(step)}
        end = time.time()
        
        data_generation_time = end-start
        return feed_dict, data_generation_time 



    
    def train(self):
        
        self.nodes = self.graph.nodes()
        self.current_step = 1
        self.num_steps = self.args.num_of_walks * len(self.nodes)
        self.current_gamma = self.args.initial_gamma        
        self.log = dict()
        self.log["times"] = []
        self.log["losses"] = []
        self.log["cluster_quality"] = []
        with tf.Session(graph=self.computation_graph) as session:
            self.init.run()
            print('Initialized')
            for repetition in range(0, self.args.num_of_walks):
                random.shuffle(self.nodes)
                
                generation_time = 0
                optimization_time = 0 
                
                self.average_embedding_loss = 0
                
                for node in self.nodes:
                    self.current_step = self.current_step + 1
                    self.current_gamma = gamma_incrementer(self.current_step, self.args.initial_gamma,self.current_gamma, self.num_steps)
                    feed_dict, data_generation_time  = self.feed_dict_generator(node, self.current_step,self.args.initial_learning_rate, self.current_gamma)
                    generation_time = generation_time + data_generation_time
                    start = time.time()
                    _, loss_val = session.run([self.train_op , self.embedding_loss], feed_dict=feed_dict)
                    end = time.time()
                    optimization_time = optimization_time + (end-start)
                    self.average_embedding_loss += loss_val


                self.average_embedding_loss /= self.vocab_size

                self.final_embeddings = self.embedding_matrix.eval()
                kmeans = KMeans(n_clusters=self.args.cluster_number, random_state=0, n_init = 1).fit(self.final_embeddings)
                labs = {i:kmeans.labels_[i] for i in range(0, self.vocab_size)}
                in_embedding_space = community.modularity(labs,self.graph)

                self.log["losses"] = self.log["losses"] + [[repetition + 1, self.average_embedding_loss]]
                self.log["times"] = self.log["times"] + [[repetition + 1, generation_time,optimization_time]]
                self.log["cluster_quality"] = self.log["cluster_quality"] + [[repetition+ 1, in_embedding_space]]
                self.log["params"] = vars(self.args)
                self.tab_printer()

        
    def initiate_dump(self):
        json_dumper(self.log, self.args.log_output)
        self.final_embeddings = pd.DataFrame(self.final_embeddings)
        if self.args.dump_matrices:
            self.final_embeddings.to_csv(self.args.embedding_output, index = None)

    def tab_printer(self):
        t = Texttable() 
        t.add_rows([['Epoch', self.log["losses"][-1][0]]])
        print t.draw()

        t = Texttable()
        t.add_rows([['Loss type', 'Loss value'], ['Embedding', self.log["losses"][-1][1]]])
        print t.draw() 

        t = Texttable()
        t.add_rows([['Clustering Method', 'Modularity'],  ['Classical K-means', self.log["cluster_quality"][-1][1]]])
        print t.draw()     
