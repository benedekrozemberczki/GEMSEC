GEMSEC
============================================
<p align="justify">
GEMSEC is a graph embedding algorithm which learns an embedding and clustering jointly. The procedure places nodes in an abstract feature space where the vertex features minimize the negative log likelihood of preserving sampled vertex neighborhoods while the nodes are clustered into a fixed number of groups in this space. GEMSEC is a general extension of earlier work in the domain as it is an augmentation of the core optimization problem of sequence based graph embedding procedures and it is agnostic of the neighborhood sampling strategy
</p>

This repository provides a reference implementation for GEMSEC as described in the paper:
> GEMSEC: Graph Embedding with Self Clustering.
> [Benedek Rozemberczki](http://homepages.inf.ed.ac.uk/s1668259/), [Benedek Rozemberczki](http://homepages.inf.ed.ac.uk/s1668259/), [Rik Sarkar](https://homepages.inf.ed.ac.uk/rsarkar/) and [Charles Sutton](https://homepages.inf.ed.ac.uk/rsarkar/) .
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
  --input STR                   Path to the edge list csv.                Default is `data/restaurant_edges.csv`
  --output STR                  Path to to embedding.                     Default is `emb/restaurant.out`
  --dimensions INT              Number of embedding dimensions.           Default is 128.
  --vertex-set-cardinality INT  Number of nodes per diffusion tree.       Default is 80.
  --num-diffusions INT          Number of diffusion per source node.      Default is 10.
  --window-size INT             Context size for optimization.            Default is 10.
  --iter INT                    Number of ASGD iterations.                Default is 1.
  --workers INT                 Number of cores.                          Default is 4.
  --alpha FLOAT                 Initial learning rate.                    Default is 0.025.
  --type STR                    Type of diffusion tree linearization.     Default is `eulerian`.
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
