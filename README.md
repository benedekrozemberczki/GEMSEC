Boosted Generative Models
============================================

This repository provides a reference implementation for Graph Embedding with Self Clustering as described in the paper:


> GEMSEC: Graph Embedding with Self Clustering
[Benedek Rozemberczki](http://homepages.inf.ed.ac.uk/s1668259/), [Ryan Davies],  [Rik Sarkar](https://cs.stanford.edu/~ermon/)  and [Charles Sutton](https://cs.stanford.edu/~ermon/).     
https://arxiv.org/pdf/1702.08484.pdf


### Requirements

The codebase is implemented in Python 2.7. To install the necessary requirements, run the following commands:

```
pip install -r requirements.txt
bash install.sh
```

### Datasets

The code takes an input graph in a csv file. Every row indicates an edge between two nodes separated by a comma. The first row is a header, and nodes are indexed from 0. A sample graph for the `Facebook Politicians` dataset is included in the  `data/` directory.


### Options

Learning and inference of boosted generative models is handled by the `main.py` script which provides the following command line arguments.

```
  --seed INT                 Random seed for numpy, tensorflow
  --datadir STR              Directory containing dataset files
  --dataset STR              Name of dataset
  --resultdir STR            Directory for saving tf checkpoints
  --run-addbgm BOOL          Runs additive boosting if True
  --addbgm-alpha FLOAT LIST  Space-separated list of model weights for additive boosting
  --run-genbgm BOOL          Runs multiplicative generative boosting if True
  --genbgm-alpha FLOAT LIST  Space-separated list of model weights for multiplicative generative boosting
  --genbgm-beta FLOAT LIST   Space-separated list of reweighting exponents for multiplicative generative boosting
  --run-discbgm BOOL         Runs multiplicative discriminative boosting if True
  --discbgm-alpha FLOAT LIST Space-separated list of model weights for multiplicative generative boosting
  --discbgm-epochs INT       Number of epochs of training for each discriminator
  --discbgm-burn-in INT      Number of discarded burn in samples for Markov chains
  --run-classifier BOOL      Uses generative model for classification if True
```


### Examples

The following commands learns boosted ensembles with two models and evaluates the ensemble for density estimation and classification.

Meta-algorithm: multiplicative generative boosting

```
python src/main.py --dataset nltcs --run-genbgm --genbgm-alpha 0.5 0.5 --genbgm-beta 0.25 0.125 --run-classifier
```

Meta-algorithm: multiplicative discriminative boosting

```
python src/main.py --dataset nltcs --run-discbgm --discbgm-alpha 1. 1. --run-classifier
```

Meta-algorithm: additive boosting

```
python src/main.py --dataset nltcs --run-addbgm --addbgm-alpha 0.5 0.25 --run-classifier
```


You can also run any combination of the meta-algorithms together as shown below.
```
python src/main.py --dataset nltcs --run-genbgm --genbgm-alpha 0.5 0.5 --genbgm-beta 0.25 0.125 --run-discbgm --discbgm-alpha 1. 1. --run-addbgm --addbgm-alpha 0.5 0.25 --run-classifier
```


### Citing

If you find Graph Embeddings with Self Clustering useful in your research, please consider citing the following paper:


>@inproceedings{grover2018boosted,  
  title={Boosted Generative Models},  
  author={Grover, Aditya and Ermon, Stefano},  
  booktitle={AAAI Conference on Artificial Intelligence},  
  year={2018}}

