GEMSEC
============================================
<p align="justify">
GEMSEC is a graph embedding algorithm which learns an embedding and clustering jointly. The procedure places nodes in an abstract feature space where the vertex features minimize the negative log likelihood of preserving sampled vertex neighborhoods while the nodes are clustered into a fixed number of groups in this space. GEMSEC is a general extension of earlier work in the domain as it is an augmentation of the core optimization problem of sequence based graph embedding procedures and it is agnostic of the neighborhood sampling strategy
</p>

This repository provides a reference implementation for GEMSEC as described in the paper:
> GEMSEC: Graph Embedding with Self Clustering.
> [Benedek Rozemberczki](http://homepages.inf.ed.ac.uk/s1668259/), [Ryan Davies](https://www.inf.ed.ac.uk/people/students/Ryan_Davies.html), [Rik Sarkar](https://homepages.inf.ed.ac.uk/rsarkar/) and [Charles Sutton](http://homepages.inf.ed.ac.uk/csutton/) .
> International Conference on Complex Networks, 2018.
> http://homepages.inf.ed.ac.uk/s1668259/papers/gemsec.pdf


### Requirements

The codebase is implemented in Python 2.7.

### Datasets

The code takes an input graph in a csv file. Every row indicates an edge between two nodes separated by a comma. The first row is a header. Nodes should be indexed starting with 0. A sample graph for the `Facebook Restaurants` dataset is included in the  `data/` directory.

### Options

Learning of the embedding is handled by the `src/embedding_clustering.py` script which provides the following command line arguments.

#### Input and output options

```
  --input STR                   Input graph path.                                 Default is `data/restaurant_edges.csv`.
  --embedding-output STR        Embeddings path.                                  Default is `output/embeddings/restaurant_embedding.csv`.
  --cluster-mean-output STR     Cluster centers path.                             Default is `output/cluster_means/pubmed_means.csv`.
  --log-output STR              Log path.                                         Default is `output/logs/restaurant.log`.
  --dump-matrices BOOL          Whether the trained model should be saved.        Default is `True`.
  --model STR                   The model type.                                   Default is `GEMSECWithRegularization`.
```

#### Skipgram options

```
  --dimensions INT                Number of dimensions.                               Default is 16.
  --random-walk-length INT        Length of random walk per source.                   Default is 80.
  --num-of-walks INT              Number of random walks per source.                  Default is 5.
  --window-size INT               Window size for proximity statistic extraction.     Default is 5.
  --distortion FLOAT              Downsampling distortion.                            Default is 0.75.
  --negative-sample-number INT    Number of negative samples to draw.                 Default is 10.
```

#### Model options

```
  --initial-learning-rate FLOAT   Initial learning rate.                                        Default is 0.001.
  --minimal-learning-rate FLOAT   Final learning rate.                                          Default is 0.0001.
  --annealing-factor FLOAT        Annealing factor for learning rate.                           Default is 0.99.
  --initial-gamma FLOAT           Initial clustering weight coefficient.                        Default is 0.1.
  --lambd FLOAR                   Smoothness regularization penalty.                            Default is 1.0
  --cluster-number INT            Number of clusters.                                           Default is 5.
  --overlap-weighting STR         Weight construction technique for regularization.             Default is `normalized_overlap`.
  --regularization-noise FLOAT    Uniform noise max and min on the feature vector distance.     Default is 10**-8.
  --regularization-norm STR       Metric used for the smoothness regularization.                Default is `euclidean`.
  --clustering-norm STR           Metric used for the cluster distances.                        Default is `euclidean`.
```

### Examples

The following commands learn a graph embedding and writes it to disk. The first column in the embedding file is the node ID.

Creating an embedding of the default dataset with the default hyperparameter settings.

```
python src/diffusion_2_vec.py
```
Creating an embedding of an other dataset the `Facebook Politicians`.

```
python src/diffusion_2_vec.py --input data/politician_edges.csv --output output/politician.out
```

Creating an embedding of the default dataset in 32 dimensions, 5 sequences per source node with maximal vertex set cardinality of 40.

```
python src/diffusion_2_vec.py --dimensions 32 --num-diffusions 5 --vertex-set-cardinality 40
```

### Citing

If you find Diff2Vec useful in your research, please consider citing the following paper:

>@inproceedings{rozemberczki2018fastsequence,  
  title={Fast Sequence Based Embedding with Diffusion Graphs},  
  author={Rozemberczki, Benedek and Sarkar, Rik},  
  booktitle={International Conference on Complex Networks},  
  year={2018}}
