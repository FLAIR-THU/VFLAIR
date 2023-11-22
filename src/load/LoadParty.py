import sys, os
sys.path.append(os.pardir)

from party.passive_party import PassiveParty, PassiveParty_LLM
from party.active_party import ActiveParty, ActiveParty_LLM

def load_parties(args):
    # party 0,1,2,...,args.k-2||,args,k-1
    args.parties = [None] * args.k 

    assert args.k >= 1
    # for passive party 0,1,2,...,args.k-2
    for ik in range(args.k-1):
        args.parties[ik] = PassiveParty(args, ik)

    # for active party args.k-1
    args.parties[args.k-1] = ActiveParty(args, args.k-1)

    return args

def load_parties_llm(args):
    # party 0,1,2,...,args.k-2||,args,k-1
    args.parties = [None] * args.k 

    assert args.k >= 1
    # for passive party 0,1,2,...,args.k-2
    for ik in range(args.k-1):
        args.parties[ik] = PassiveParty_LLM(args, ik)

    # for active party args.k-1
    args.parties[args.k-1] = ActiveParty_LLM(args, args.k-1)

    return args