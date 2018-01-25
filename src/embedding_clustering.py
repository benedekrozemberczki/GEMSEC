from helper import parameter_parser, graph_reader
from model import GEMSECWithRegularization, GEMSEC, DWWithRegularization, DW
import networkx as nx

def create_and_run_model(args):
    
    graph = graph_reader(args.input)
    if args.model == "GEMSECWithRegularization":
        model = GEMSECWithRegularization(args, graph)
    elif args.model == "GEMSEC":
        model = GEMSEC(args, graph)
    elif args.model == "DWWithRegularization":
        model = DWWithRegularization(args, graph)
    else:
        model = DW(args, graph)
    model.train()
    model.initiate_dump()

if __name__ == "__main__":
    args = parameter_parser()
    create_and_run_model(args)



