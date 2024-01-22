import sys, os
sys.path.append(os.pardir)

from framework.ml.passive_party import PassiveParty_LLM
from framework.ml.active_party import ActiveParty_LLM
# from party.server import Server

def load_parties_llm(args):
    # party 0,1,2,...,args.k-2||,args,k-1
    args.parties = [None] * args.k 
    # args.servers = [None] * args.k_server

    assert args.k >= 1
    # for passive party 0,1,2,...,args.k-2
    for ik in range(args.k-1):
        args.parties[ik] = PassiveParty_LLM(args, None)
    # for active party args.k-1
    args.parties[args.k-1] = ActiveParty_LLM(args)

    # # for server party 0,1,2,...,args.k_server-1
    # for ik in range(args.k_server):
    #     args.servers[ik] = Server(args, ik)

    return args