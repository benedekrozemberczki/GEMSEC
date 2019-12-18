"""GEMSEC model classes."""

import math
import time
import random
import numpy as np
import networkx as nx
from tqdm import tqdm
import tensorflow as tf
from calculation_helper import overlap_generator
from layers import DeepWalker, Clustering, Regularization
from calculation_helper import neural_modularity_calculator, classical_modularity_calculator
from calculation_helper import gamma_incrementer, RandomWalker, SecondOrderRandomWalker
from calculation_helper import index_generation, batch_input_generator, batch_label_generator
from print_and_read import json_dumper, log_setup
from print_and_read import initiate_dump_gemsec, initiate_dump_dw
from print_and_read import tab_printer, epoch_printer, log_updater

class Model(object):
    """
    Abstract model class.
    """
    def __init__(self, args, graph):
        """
        Every model needs the same initialization -- args, graph.
        We delete the sampler object to save memory.
        We also build the computation graph up.
        """
        self.args = args
        self.graph = graph
        if self.args.walker == "first":
            self.walker = RandomWalker(self.graph,
                                       self.args.num_of_walks,
                                       self.args.random_walk_length)

            self.degrees, self.walks = self.walker.do_walks()
        else:
            self.walker = SecondOrderRandomWalker(self.graph, False, self.args.P, self.args.Q)
            self.walker.preprocess_transition_probs()
            self.walks, self.degrees = self.walker.simulate_walks(self.args.num_of_walks,
                                                                  self.args.random_walk_length)
        self.nodes = [node for node in self.graph.nodes()]
        del self.walker
        self.vocab_size = len(self.degrees)
        self.true_step_size = self.args.num_of_walks*self.vocab_size
        self.build()

    def build(self):
        """
        Building the model.
        """
        pass

    def feed_dict_generator(self):
        """
        Creating the feed generator
        """
        pass

    def train(self):
        """
        Training the model.
        """
        pass

class GEMSECWithRegularization(Model):
    """
    Regularized GEMSEC class.
    """
    def build(self):
        """
        Method to create the computational graph.
        """
        self.computation_graph = tf.Graph()
        with self.computation_graph.as_default():

            self.walker_layer = DeepWalker(self.args, self.vocab_size, self.degrees)
            self.cluster_layer = Clustering(self.args)
            self.regularizer_layer = Regularization(self.args)

            self.gamma = tf.placeholder("float")
            self.loss = self.walker_layer()
            self.loss = self.loss + self.gamma*self.cluster_layer(self.walker_layer)
            self.loss = self.loss + self.regularizer_layer(self.walker_layer)

            self.batch = tf.Variable(0)
            self.step = tf.placeholder("float")

            self.learning_rate_new = tf.train.polynomial_decay(self.args.initial_learning_rate,
                                                               self.batch,
                                                               self.true_step_size,
                                                               self.args.minimal_learning_rate,
                                                               self.args.annealing_factor)

            self.train_op = tf.train.AdamOptimizer(self.learning_rate_new).minimize(self.loss,
                                                                                    global_step=self.batch)

            self.init = tf.global_variables_initializer()

        self.weights = overlap_generator(self.args, self.graph)

    def feed_dict_generator(self, a_random_walk, step, gamma):
        """
        Method to generate:
        1. random walk features.
        2. left and right handside matrices.
        3. proper time index and overlap vector.
        """
        index_1, index_2, overlaps = index_generation(self.weights, a_random_walk)

        batch_inputs = batch_input_generator(a_random_walk,
                                             self.args.random_walk_length,
                                             self.args.window_size)

        batch_labels = batch_label_generator(a_random_walk,
                                             self.args.random_walk_length,
                                             self.args.window_size)

        feed_dict = {self.walker_layer.train_labels: batch_labels,
                     self.walker_layer.train_inputs: batch_inputs,
                     self.gamma: gamma,
                     self.step: float(step),
                     self.regularizer_layer.edge_indices_left: index_1,
                     self.regularizer_layer.edge_indices_right: index_2,
                     self.regularizer_layer.overlap: overlaps}
        return feed_dict

    def train(self):
        """
        Method for:
        1. training the embedding.
        2. logging.
        This method is inherited by GEMSEC and DeepWalk variants without an override.
        """
        self.current_step = 0
        self.current_gamma = self.args.initial_gamma
        self.log = log_setup(self.args)
        with tf.Session(graph=self.computation_graph) as session:
            self.init.run()
            print("Model Initialized.")
            for repetition in range(self.args.num_of_walks):

                random.shuffle(self.nodes)
                self.optimization_time = 0
                self.average_loss = 0

                epoch_printer(repetition)

                for node in tqdm(self.nodes):
                    self.current_step = self.current_step + 1
                    self.current_gamma = gamma_incrementer(self.current_step,
                                                           self.args.initial_gamma,
                                                           self.args.final_gamma,
                                                           self.current_gamma,
                                                           self.true_step_size)

                    feed_dict = self.feed_dict_generator(self.walks[self.current_step-1],
                                                         self.current_step,
                                                         self.current_gamma)
                    start = time.time()
                    _, loss = session.run([self.train_op, self.loss], feed_dict=feed_dict)
                    end = time.time()
                    self.optimization_time = self.optimization_time + (end-start)
                    self.average_loss = self.average_loss + loss

                print("")
                self.average_loss = self.average_loss/self.vocab_size
                self.final_embeddings = self.walker_layer.embedding_matrix.eval()
                if "GEMSEC" in self.args.model:
                    self.c_means = self.cluster_layer.cluster_means.eval()
                    self.modularity_score, assignments = neural_modularity_calculator(self.graph,
                                                                                      self.final_embeddings,
                                                                                      self.c_means)
                else:
                    self.modularity_score, assignments = classical_modularity_calculator(self.graph,
                                                                                         self.final_embeddings,
                                                                                         self.args)
                self.log = log_updater(self.log, repetition, self.average_loss,
                                       self.optimization_time, self.modularity_score)
                tab_printer(self.log)
        if "GEMSEC" in self.args.model:
            initiate_dump_gemsec(self.log, assignments, self.args,
                                 self.final_embeddings, self.c_means)
        else:
            initiate_dump_dw(self.log, assignments,
                             self.args, self.final_embeddings)

class GEMSEC(GEMSECWithRegularization):
    """
    GEMSEC class.
    """
    def build(self):
        """
        Method to create the computational graph.
        """
        self.computation_graph = tf.Graph()
        with self.computation_graph.as_default():

            self.walker_layer = DeepWalker(self.args, self.vocab_size, self.degrees)
            self.cluster_layer = Clustering(self.args)

            self.gamma = tf.placeholder("float")
            self.loss = self.walker_layer()+self.gamma*self.cluster_layer(self.walker_layer)

            self.batch = tf.Variable(0)
            self.step = tf.placeholder("float")

            self.learning_rate_new = tf.train.polynomial_decay(self.args.initial_learning_rate,
                                                               self.batch,
                                                               self.true_step_size,
                                                               self.args.minimal_learning_rate,
                                                               self.args.annealing_factor)

            self.train_op = tf.train.AdamOptimizer(self.learning_rate_new).minimize(self.loss,
                                                                                    global_step=self.batch)

            self.init = tf.global_variables_initializer()

    def feed_dict_generator(self, a_random_walk, step, gamma):
        """
        Method to generate:
        1. random walk features.
        2. left and right handside matrices.
        Proper time index and overlap vector.
        """
        batch_inputs = batch_input_generator(a_random_walk,
                                             self.args.random_walk_length,
                                             self.args.window_size)

        batch_labels = batch_label_generator(a_random_walk,
                                             self.args.random_walk_length,
                                             self.args.window_size)

        feed_dict = {self.walker_layer.train_labels: batch_labels,
                     self.walker_layer.train_inputs: batch_inputs,
                     self.gamma: gamma,
                     self.step: float(step)}

        return feed_dict


class DeepWalkWithRegularization(GEMSECWithRegularization):
    """
    Regularized DeepWalk class.
    """
    def build(self):
        """
        Method to create the computational graph and initialize weights.
        """
        self.computation_graph = tf.Graph()
        with self.computation_graph.as_default():

            self.walker_layer = DeepWalker(self.args, self.vocab_size, self.degrees)
            self.regularizer_layer = Regularization(self.args)

            self.gamma = tf.placeholder("float")
            self.loss = self.walker_layer()+self.regularizer_layer(self.walker_layer)

            self.batch = tf.Variable(0)
            self.step = tf.placeholder("float")

            self.learning_rate_new = tf.train.polynomial_decay(self.args.initial_learning_rate,
                                                               self.batch,
                                                               self.true_step_size,
                                                               self.args.minimal_learning_rate,
                                                               self.args.annealing_factor)

            self.train_op = tf.train.AdamOptimizer(self.learning_rate_new).minimize(self.loss,
                                                                                    global_step=self.batch)

            self.init = tf.global_variables_initializer()

        self.weights = overlap_generator(self.args, self.graph)

    def feed_dict_generator(self, a_random_walk, step, gamma):
        """
        Method to generate:
        1. random walk features.
        2. left and right handside matrices.
        3. proper time index and overlap vector.
        """
        index_1, index_2, overlaps = index_generation(self.weights, a_random_walk)

        batch_inputs = batch_input_generator(a_random_walk,
                                             self.args.random_walk_length,
                                             self.args.window_size)

        batch_labels = batch_label_generator(a_random_walk,
                                             self.args.random_walk_length,
                                             self.args.window_size)

        feed_dict = {self.walker_layer.train_labels: batch_labels,
                     self.walker_layer.train_inputs: batch_inputs,
                     self.gamma: gamma,
                     self.step: float(step),
                     self.regularizer_layer.edge_indices_left: index_1,
                     self.regularizer_layer.edge_indices_right: index_2,
                     self.regularizer_layer.overlap: overlaps}

        return feed_dict

class DeepWalk(GEMSECWithRegularization):
    """
    DeepWalk class.
    """
    def build(self):
        """
        Method to create the computational graph and initialize weights.
        """
        self.computation_graph = tf.Graph()
        with self.computation_graph.as_default():

            self.walker_layer = DeepWalker(self.args, self.vocab_size, self.degrees)

            self.gamma = tf.placeholder("float")
            self.loss = self.walker_layer()

            self.batch = tf.Variable(0)
            self.step = tf.placeholder("float")

            self.learning_rate_new = tf.train.polynomial_decay(self.args.initial_learning_rate,
                                                               self.batch,
                                                               self.true_step_size,
                                                               self.args.minimal_learning_rate,
                                                               self.args.annealing_factor)

            self.train_op = tf.train.AdamOptimizer(self.learning_rate_new).minimize(self.loss,
                                                                                    global_step=self.batch)

            self.init = tf.global_variables_initializer()

    def feed_dict_generator(self, a_random_walk, step, gamma):
        """
        Method to generate random walk features, gamma and proper time index.
        """

        batch_inputs = batch_input_generator(a_random_walk,
                                             self.args.random_walk_length,
                                             self.args.window_size)

        batch_labels = batch_label_generator(a_random_walk,
                                             self.args.random_walk_length,
                                             self.args.window_size)

        feed_dict = {self.walker_layer.train_labels: batch_labels,
                     self.walker_layer.train_inputs: batch_inputs,
                     self.gamma: gamma,
                     self.step: float(step)}

        return feed_dict
