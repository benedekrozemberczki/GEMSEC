from parser import parameter_parser
from print_and_read import graph_reader
from model import GEMSECWithRegularization, GEMSEC, DWWithRegularization, DW

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

if __name__ == "__main__":
    args = parameter_parser()
    create_and_run_model(args)



