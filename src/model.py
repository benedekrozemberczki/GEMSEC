import random
import numpy as np
import math
import time

import networkx as nx
import tensorflow as tf

from tqdm import tqdm

from calculation_helper import gamma_incrementer,  RandomWalker, index_generation, batch_input_generator, batch_label_generator
from calculation_helper import normalized_overlap, overlap, unit, min_norm, overlap_generator
from calculation_helper import neural_modularity_calculator, classical_modularity_calculator
from print_and_read import json_dumper, log_setup, initiate_dump_gemsec, initiate_dump_dw, tab_printer, epoch_printer, log_updater

class Model(object):
    """
    Abstract model class.
    """    
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

        self.walker = RandomWalker(self.graph, nx.nodes(graph), self.args.num_of_walks, self.args.random_walk_length)
        self.degrees, self.walks = self.walker.do_walks()
        self.nodes = self.graph.nodes()
        del self.walker
        
        self.vocab_size = len(self.degrees)
        self.true_step_size = self.args.num_of_walks*self.vocab_size
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
    
            #------------------------------------------------------------
            # Defining the cluster means and calculcating the distances.
            #------------------------------------------------------------
    
    
            self.cluster_means = tf.Variable(tf.random_uniform([self.args.cluster_number, self.args.dimensions],
                                            -0.1/self.args.dimensions, 0.1/self.args.dimensions))
           
            self.clustering_differences = tf.expand_dims(self.embedding_partial,1) - self.cluster_means

            self.cluster_distances = tf.norm(self.clustering_differences, ord = 2, axis = 2)

            self.to_be_averaged = tf.reduce_min(self.cluster_distances, axis = 1)
            self.clustering_loss = tf.reduce_mean(self.to_be_averaged)
    
            #------------------------------------------
            # Defining the smoothness regularization.
            #------------------------------------------
    
            self.left_features = tf.nn.embedding_lookup(self.embedding_partial, self.edge_indices_left, max_norm = 1)
            self.right_features = tf.nn.embedding_lookup(self.embedding_partial, self.edge_indices_right, max_norm = 1)
            
            self.regularization_differences = self.left_features - self.right_features + np.random.uniform(self.args.regularization_noise,self.args.regularization_noise, (self.args.random_walk_length-1, self.args.dimensions))

            self.regularization_distances = tf.norm(self.regularization_differences, ord = 2,axis=1)

            self.regularization_distances = tf.reshape(self.regularization_distances, [ -1, 1])
            self.regularization_loss = tf.reduce_mean(tf.matmul(tf.transpose(self.overlap), self.regularization_distances))
    
            #------------------------------------------------
            # Defining the combined loss and initialization.
            #------------------------------------------------

            self.gamma = tf.placeholder("float")
            self.loss = self.embedding_loss + self.gamma*self.clustering_loss + self.args.lambd*self.regularization_loss

    
            self.batch = tf.Variable(0)
            self.step = tf.placeholder("float")
    
            self.learning_rate_new = tf.train.polynomial_decay(self.args.initial_learning_rate,
                                                               self.batch,
                                                               self.true_step_size,
                                                               self.args.minimal_learning_rate,
                                                               self.args.annealing_factor)
    
            self.train_op = tf.train.AdamOptimizer(self.learning_rate_new).minimize(self.loss, global_step = self.batch)
    
            self.init = tf.global_variables_initializer()

        if self.args.overlap_weighting == "normalized_overlap":
            self.overlap_weighter = normalized_overlap
        elif self.args.overlap_weighting == "overlap":
            self.overlap_weighter = overlap
        elif self.args.overlap_weighting == "min_norm":
            self.overlap_weighter = min_norm
        else:
            self.overlap_weighter = unit
        print(" ")
        print("Weight calculation started.")
        print(" ")
        self.weights = overlap_generator(self.overlap_weighter, self.graph)
        print(" ")

    def feed_dict_generator(self, a_random_walk, step, gamma):

        index_1, index_2, overlaps = index_generation(self.weights, a_random_walk)

        batch_inputs = batch_input_generator(a_random_walk, self.args.random_walk_length, self.args.window_size)

        batch_labels = batch_label_generator(a_random_walk, self.args.random_walk_length, self.args.window_size)

        feed_dict = {self.train_inputs: batch_inputs,
                     self.train_labels: batch_labels,
                     self.gamma: gamma,
                     self.step: float(step),
                     self.edge_indices_left: index_1,
                     self.edge_indices_right: index_2,
                     self.overlap: overlaps}
 
        return feed_dict

    def train(self):
 
        self.current_step = 0
        self.current_gamma = self.args.initial_gamma
        self.log = log_setup(self.args)

        with tf.Session(graph = self.computation_graph) as session:
            self.init.run()
            print("Model Initialized.")
            for repetition in range(0, self.args.num_of_walks):

                random.shuffle(self.nodes)
                self.optimization_time = 0 
                self.average_loss = 0

                epoch_printer(repetition)

                for node in tqdm(self.nodes):
                    self.current_step = self.current_step + 1
                    self.current_gamma = gamma_incrementer(self.current_step, self.args.initial_gamma, self.current_gamma, self.true_step_size)
                    feed_dict = self.feed_dict_generator(self.walks[self.current_step-1], self.current_step, self.current_gamma)
                    start = time.time()
                    _, loss = session.run([self.train_op , self.loss], feed_dict=feed_dict)
                    end = time.time()
                    self.optimization_time = self.optimization_time + (end-start)
                    self.average_loss = self.average_loss + loss

                print("")
                self.average_loss = self.average_loss/self.vocab_size

                self.final_embeddings = self.embedding_matrix.eval()
                self.c_means = self.cluster_means.eval()

                self.modularity_score, assignments = neural_modularity_calculator(self.graph, self.final_embeddings, self.c_means)

                self.log = log_updater(self.log, repetition, self.average_loss, self.optimization_time, self.modularity_score)
                tab_printer(self.log)

        initiate_dump_gemsec(self.log, assignments, self.args, self.final_embeddings, self.c_means)

class GEMSEC(Model):
    def __init__(self, args, graph, **kwargs):
        super(GEMSEC, self).__init__(**kwargs)

        self.args = args
        self.graph = graph

        self.walker = RandomWalker(self.graph, nx.nodes(graph), self.args.num_of_walks, self.args.random_walk_length)
        self.degrees, self.walks = self.walker.do_walks()
        self.nodes = self.graph.nodes()
        del self.walker
        
        self.vocab_size = len(self.degrees)
        self.true_step_size = self.args.num_of_walks*self.vocab_size
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
    
            #------------------------------------------------------------
            # Defining the cluster means and calculcating the distances.
            #------------------------------------------------------------
    
    
            self.cluster_means = tf.Variable(tf.random_uniform([self.args.cluster_number, self.args.dimensions],
                                            -0.1/self.args.dimensions, 0.1/self.args.dimensions))
           
            self.clustering_differences = tf.expand_dims(self.embedding_partial,1) - self.cluster_means

            self.cluster_distances = tf.norm(self.clustering_differences, ord = 2, axis = 2)

            self.to_be_averaged = tf.reduce_min(self.cluster_distances, axis = 1)
            self.clustering_loss = tf.reduce_mean(self.to_be_averaged)
    
            #------------------------------------------------
            # Defining the combined loss and initialization.
            #------------------------------------------------

            self.gamma = tf.placeholder("float")
            self.loss = self.embedding_loss + self.gamma*self.clustering_loss

            self.batch = tf.Variable(0)
            self.step = tf.placeholder("float")
    
            self.learning_rate_new = tf.train.polynomial_decay(self.args.initial_learning_rate,
                                                               self.batch,
                                                               self.true_step_size,
                                                               self.args.minimal_learning_rate,
                                                               self.args.annealing_factor)
    
            self.train_op = tf.train.AdamOptimizer(self.learning_rate_new).minimize(self.loss, global_step = self.batch)
    
            self.init = tf.global_variables_initializer()



    def feed_dict_generator(self, a_random_walk, step, gamma):

        batch_inputs = batch_input_generator(a_random_walk, self.args.random_walk_length, self.args.window_size)

        batch_labels = batch_label_generator(a_random_walk, self.args.random_walk_length, self.args.window_size)

        feed_dict = {self.train_inputs: batch_inputs,
                     self.train_labels: batch_labels,
                     self.gamma: gamma,
                     self.step: float(step)}
 
        return feed_dict

    def train(self):
 
        self.current_step = 0
        self.current_gamma = self.args.initial_gamma
        self.log = log_setup(self.args)

        with tf.Session(graph = self.computation_graph) as session:
            self.init.run()
            print("Model Initialized.")
            for repetition in range(0, self.args.num_of_walks):

                random.shuffle(self.nodes)
                self.optimization_time = 0 
                self.average_loss = 0

                epoch_printer(repetition)

                for node in tqdm(self.nodes):
                    self.current_step = self.current_step + 1
                    self.current_gamma = gamma_incrementer(self.current_step, self.args.initial_gamma, self.current_gamma, self.true_step_size)
                    feed_dict = self.feed_dict_generator(self.walks[self.current_step-1], self.current_step, self.current_gamma)
                    start = time.time()
                    _, loss = session.run([self.train_op , self.loss], feed_dict=feed_dict)
                    end = time.time()
                    self.optimization_time = self.optimization_time + (end-start)
                    self.average_loss = self.average_loss + loss

                print("")
                self.average_loss = self.average_loss/self.vocab_size

                self.final_embeddings = self.embedding_matrix.eval()
                self.c_means = self.cluster_means.eval()

                self.modularity_score, assignments = neural_modularity_calculator(self.graph, self.final_embeddings, self.c_means)

                self.log = log_updater(self.log, repetition, self.average_loss, self.optimization_time, self.modularity_score)
                tab_printer(self.log)

        initiate_dump_gemsec(self.log, assignments, self.args, self.final_embeddings, self.c_means)

class DWWithRegularization(Model):
    def __init__(self, args, graph, **kwargs):
        super(DWWithRegularization, self).__init__(**kwargs)

        self.args = args
        self.graph = graph

        self.walker = RandomWalker(self.graph, nx.nodes(graph), self.args.num_of_walks, self.args.random_walk_length)
        self.degrees, self.walks = self.walker.do_walks()
        self.nodes = self.graph.nodes()
        del self.walker
        
        self.vocab_size = len(self.degrees)
        self.true_step_size = self.args.num_of_walks*self.vocab_size
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
    
            #------------------------------------------
            # Defining the smoothness regularization.
            #------------------------------------------
    
            self.left_features = tf.nn.embedding_lookup(self.embedding_partial, self.edge_indices_left, max_norm = 1)
            self.right_features = tf.nn.embedding_lookup(self.embedding_partial, self.edge_indices_right, max_norm = 1)
            
            self.regularization_differences = self.left_features - self.right_features + np.random.uniform(self.args.regularization_noise,self.args.regularization_noise, (self.args.random_walk_length-1, self.args.dimensions))

            self.regularization_distances = tf.norm(self.regularization_differences, ord = 2,axis=1)

            self.regularization_distances = tf.reshape(self.regularization_distances, [ -1, 1])
            self.regularization_loss = tf.reduce_mean(tf.matmul(tf.transpose(self.overlap), self.regularization_distances))
    
            #------------------------------------------------
            # Defining the combined loss and initialization.
            #------------------------------------------------

            self.gamma = tf.placeholder("float")
            self.loss = self.embedding_loss + self.args.lambd*self.regularization_loss

    
            self.batch = tf.Variable(0)
            self.step = tf.placeholder("float")
    
            self.learning_rate_new = tf.train.polynomial_decay(self.args.initial_learning_rate,
                                                               self.batch,
                                                               self.true_step_size,
                                                               self.args.minimal_learning_rate,
                                                               self.args.annealing_factor)
    
            self.train_op = tf.train.AdamOptimizer(self.learning_rate_new).minimize(self.loss, global_step = self.batch)
    
            self.init = tf.global_variables_initializer()

        if self.args.overlap_weighting == "normalized_overlap":
            self.overlap_weighter = normalized_overlap
        elif self.args.overlap_weighting == "overlap":
            self.overlap_weighter = overlap
        elif self.args.overlap_weighting == "min_norm":
            self.overlap_weighter = min_norm
        else:
            self.overlap_weighter = unit
        print(" ")
        print("Weight calculation started.")
        print(" ")
        self.weights = overlap_generator(self.overlap_weighter, self.graph)
        print(" ")

    def feed_dict_generator(self, a_random_walk, step, gamma):

        index_1, index_2, overlaps = index_generation(self.weights, a_random_walk)

        batch_inputs = batch_input_generator(a_random_walk, self.args.random_walk_length, self.args.window_size)

        batch_labels = batch_label_generator(a_random_walk, self.args.random_walk_length, self.args.window_size)

        feed_dict = {self.train_inputs: batch_inputs,
                     self.train_labels: batch_labels,
                     self.gamma: gamma,
                     self.step: float(step),
                     self.edge_indices_left: index_1,
                     self.edge_indices_right: index_2,
                     self.overlap: overlaps}
 
        return feed_dict

    def train(self):
 
        self.current_step = 0
        self.current_gamma = self.args.initial_gamma
        self.log = log_setup(self.args)

        with tf.Session(graph = self.computation_graph) as session:
            self.init.run()
            print("Model Initialized.")
            for repetition in range(0, self.args.num_of_walks):

                random.shuffle(self.nodes)
                self.optimization_time = 0 
                self.average_loss = 0

                epoch_printer(repetition)

                for node in tqdm(self.nodes):
                    self.current_step = self.current_step + 1
                    self.current_gamma = gamma_incrementer(self.current_step, self.args.initial_gamma, self.current_gamma, self.true_step_size)
                    feed_dict = self.feed_dict_generator(self.walks[self.current_step-1], self.current_step, self.current_gamma)
                    start = time.time()
                    _, loss = session.run([self.train_op , self.loss], feed_dict=feed_dict)
                    end = time.time()
                    self.optimization_time = self.optimization_time + (end-start)
                    self.average_loss = self.average_loss + loss

                print("")
                self.average_loss = self.average_loss/self.vocab_size

                self.final_embeddings = self.embedding_matrix.eval()

                self.modularity_score, assignments = classical_modularity_calculator(self.graph, self.final_embeddings, self.args)

                self.log = log_updater(self.log, repetition, self.average_loss, self.optimization_time, self.modularity_score)
                tab_printer(self.log)

        initiate_dump_dw(self.log, assignments, self.args, self.final_embeddings)

class DW(Model):
    def __init__(self, args, graph, **kwargs):
        super(DW, self).__init__(**kwargs)

        self.args = args
        self.graph = graph

        self.walker = RandomWalker(self.graph, nx.nodes(graph), self.args.num_of_walks, self.args.random_walk_length)
        self.degrees, self.walks = self.walker.do_walks()
        self.nodes = self.graph.nodes()
        del self.walker
        
        self.vocab_size = len(self.degrees)
        self.true_step_size = self.args.num_of_walks*self.vocab_size
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
    

            self.gamma = tf.placeholder("float")
            self.loss = self.embedding_loss

    
            self.batch = tf.Variable(0)
            self.step = tf.placeholder("float")
    
            self.learning_rate_new = tf.train.polynomial_decay(self.args.initial_learning_rate,
                                                               self.batch,
                                                               self.true_step_size,
                                                               self.args.minimal_learning_rate,
                                                               self.args.annealing_factor)
    
            self.train_op = tf.train.AdamOptimizer(self.learning_rate_new).minimize(self.loss, global_step = self.batch)
    
            self.init = tf.global_variables_initializer()


    def feed_dict_generator(self, a_random_walk, step, gamma):

        batch_inputs = batch_input_generator(a_random_walk, self.args.random_walk_length, self.args.window_size)

        batch_labels = batch_label_generator(a_random_walk, self.args.random_walk_length, self.args.window_size)

        feed_dict = {self.train_inputs: batch_inputs,
                     self.train_labels: batch_labels,
                     self.gamma: gamma,
                     self.step: float(step)}
 
        return feed_dict

    def train(self):
 
        self.current_step = 0
        self.current_gamma = self.args.initial_gamma
        self.log = log_setup(self.args)

        with tf.Session(graph = self.computation_graph) as session:
            self.init.run()
            print("Model Initialized.")
            for repetition in range(0, self.args.num_of_walks):

                random.shuffle(self.nodes)
                self.optimization_time = 0 
                self.average_loss = 0

                epoch_printer(repetition)

                for node in tqdm(self.nodes):
                    self.current_step = self.current_step + 1
                    self.current_gamma = gamma_incrementer(self.current_step, self.args.initial_gamma, self.current_gamma, self.true_step_size)
                    feed_dict = self.feed_dict_generator(self.walks[self.current_step-1], self.current_step, self.current_gamma)
                    start = time.time()
                    _, loss = session.run([self.train_op , self.loss], feed_dict=feed_dict)
                    end = time.time()
                    self.optimization_time = self.optimization_time + (end-start)
                    self.average_loss = self.average_loss + loss

                print("")
                self.average_loss = self.average_loss/self.vocab_size

                self.final_embeddings = self.embedding_matrix.eval()

                self.modularity_score, assignments = classical_modularity_calculator(self.graph, self.final_embeddings, self.args)

                self.log = log_updater(self.log, repetition, self.average_loss, self.optimization_time, self.modularity_score)
                tab_printer(self.log)

        initiate_dump_dw(self.log, assignments, self.args, self.final_embeddings)
