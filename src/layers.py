"""Definition of computational layers."""

import math
import numpy as np
import tensorflow as tf

class DeepWalker:
    """
    DeepWalk embedding layer class.
    """
    def __init__(self, args, vocab_size, degrees):
        """
        Initialization of the layer with proper matrices and biases.
        The input variables are also initialized here.
        """
        self.args = args
        self.vocab_size = vocab_size
        self.degrees = degrees
        self.train_labels = tf.placeholder(tf.int64, shape=[None, self.args.window_size])

        self.train_inputs = tf.placeholder(tf.int64, shape=[None])

        self.embedding_matrix = tf.Variable(tf.random_uniform([self.vocab_size, self.args.dimensions],
                                                              -0.1/self.args.dimensions,
                                                              0.1/self.args.dimensions))


        self.nce_weights = tf.Variable(tf.truncated_normal([self.vocab_size, self.args.dimensions],
                                       stddev=1.0/math.sqrt(self.args.dimensions)))

        self.nce_biases = tf.Variable(tf.random_uniform([self.vocab_size],
                                                        -0.1/self.args.dimensions,
                                                        0.1/self.args.dimensions))

    def __call__(self):
        """
        Calculating the embedding cost with NCE and returning it.
        """
        self.train_labels_flat = tf.reshape(self.train_labels, [-1, 1])
        self.input_ones = tf.ones_like(self.train_labels)
        self.train_inputs_flat = tf.reshape(tf.multiply(self.input_ones, tf.reshape(self.train_inputs,[-1, 1])), [-1])
        self.embedding_partial = tf.nn.embedding_lookup(self.embedding_matrix,
                                                        self.train_inputs_flat,
                                                        max_norm=1)    

        self.sampler = tf.nn.fixed_unigram_candidate_sampler(true_classes=self.train_labels_flat,
                                                             num_true=1,
                                                             num_sampled=self.args.negative_sample_number,
                                                             unique=True,
                                                             range_max=self.vocab_size,
                                                             distortion=self.args.distortion,
                                                             unigrams=self.degrees)
    
        self.embedding_losses = tf.nn.sampled_softmax_loss(weights=self.nce_weights,
                                                           biases=self.nce_biases,
                                                           labels=self.train_labels_flat,
                                                           inputs=self.embedding_partial,
                                                           num_true=1,
                                                           num_sampled=self.args.negative_sample_number,
                                                           num_classes=self.vocab_size,
                                                           sampled_values=self.sampler)
    
        return tf.reduce_mean(self.embedding_losses)

class Clustering:
    """
    Latent space clustering class.
    """
    def __init__(self, args):
        """
        Initializing the cluster center matrix.
        """
        self.args = args
        self.cluster_means = tf.Variable(tf.random_uniform([self.args.cluster_number, self.args.dimensions],
                                         -0.1/self.args.dimensions,
                                         0.1/self.args.dimensions))
    def __call__(self, Walker):
        """
        Calculating the clustering cost.
        """
           
        self.clustering_differences = tf.expand_dims(Walker.embedding_partial, 1) - self.cluster_means
        self.cluster_distances = tf.norm(self.clustering_differences, ord=2, axis=2)
        self.to_be_averaged = tf.reduce_min(self.cluster_distances, axis=1)
        return tf.reduce_mean(self.to_be_averaged)

class Regularization:
    """
    Smoothness regularization class.
    """
    def __init__(self, args):
        """
        Initializing the indexing variables and the weight vector.
        """
        self.args = args
        self.edge_indices_right = tf.placeholder(tf.int64, shape=[None])
        self.edge_indices_left = tf.placeholder(tf.int64, shape=[None])
        self.overlap = tf.placeholder(tf.float32, shape=[None, 1])

    def __call__(self, Walker):
        """
        Calculating the regularization cost.
        """
        self.left_features = tf.nn.embedding_lookup(Walker.embedding_partial,
                                                    self.edge_indices_left,
                                                    max_norm=1)

        self.right_features = tf.nn.embedding_lookup(Walker.embedding_partial,
                                                     self.edge_indices_right,
                                                     max_norm=1)
        self.regularization_differences = self.left_features - self.right_features
        noise =  np.random.uniform(-self.args.regularization_noise,
                                   self.args.regularization_noise,
                                   (self.args.random_walk_length-1, self.args.dimensions))
        self.regularization_differences = self.regularization_differences + noise
        self.regularization_distances = tf.norm(self.regularization_differences, ord=2,axis=1)
        self.regularization_distances = tf.reshape(self.regularization_distances, [-1, 1])
        self.regularization_loss = tf.reduce_mean(tf.matmul(tf.transpose(self.overlap), self.regularization_distances))
        return self.args.lambd*self.regularization_loss
