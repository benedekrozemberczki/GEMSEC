import pandas as pd
import networkx as nx
from texttable import Texttable
import json

def graph_reader(input_path):
    edges = pd.read_csv(input_path)
    graph = nx.from_edgelist(edges.values.tolist())
    return graph

def log_setup(args_in):
    log = dict()
    log["times"] = []
    log["losses"] = []
    log["cluster_quality"] = []
    log["params"] = vars(args_in)
    return log

def json_dumper(data, path):
    with open(path, 'w') as outfile:
        json.dump(data, outfile)

def initiate_dump_gemsec(log, assignments, args, final_embeddings, c_means):
    json_dumper(log, args.log_output)
    json_dumper(assignments, args.assignment_output)
    if args.dump_matrices:
        final_embeddings = pd.DataFrame(final_embeddings)
        final_embeddings.to_csv(args.embedding_output, index = None)
        c_means = pd.DataFrame(c_means)
        c_means.to_csv(args.cluster_mean_output, index = None)

def initiate_dump_dw(log, assignments, args, final_embeddings):
    json_dumper(log, args.log_output)
    json_dumper(assignments, args.assignment_output)
    if args.dump_matrices:
        final_embeddings = pd.DataFrame(final_embeddings)
        final_embeddings.to_csv(args.embedding_output, index = None)

def tab_printer(log):
    t = Texttable() 
    t.add_rows([['Epoch', log["losses"][-1][0]]])
    print t.draw()

    t = Texttable()
    t.add_rows([['Loss', round(log["losses"][-1][1],3)]])
    print t.draw() 

    t = Texttable()
    t.add_rows([['Modularity', round(log["cluster_quality"][-1][1],3)]])
    print t.draw()   

def epoch_printer(repetition):
    print("")
    print("Epoch " + str(repetition+1) + ". initiated.")
    print("")

def log_updater(log, repetition, average_loss, optimization_time, modularity_score):
    index = repetition + 1
    log["losses"] = log["losses"] + [[index, average_loss]]
    log["times"] = log["times"] + [[index, optimization_time]]
    log["cluster_quality"] = log["cluster_quality"] + [[index, modularity_score]]
    return log
