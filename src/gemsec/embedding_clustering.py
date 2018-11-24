from gemsec.parser import parameter_parser
from gemsec.print_and_read import graph_reader
from gemsec.model import GEMSECWithRegularization, GEMSEC, DeepWalkWithRegularization, DeepWalk

def create_and_run_model(args):
    """
    Function to read the graph, create an embedding and train it.
    """
    graph = graph_reader(args.input)
    if args.model == "GEMSECWithRegularization":
        model = GEMSECWithRegularization(args, graph)
    elif args.model == "GEMSEC":
        model = GEMSEC(args, graph)
    elif args.model == "DeepWalkWithRegularization":
        model = DeepWalkWithRegularization(args, graph)
    else:
        model = DeepWalk(args, graph)
    model.train()

if __name__ == "__main__":
    args = parameter_parser()
    create_and_run_model(args)



